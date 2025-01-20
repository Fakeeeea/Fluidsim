#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#include "sim_math.cuh"
#include "types.h"

/* Many function do something like:
 * pos = positions[i];
 * //calculations..
 * positions[i] = pos;
 * The reason for this unconventional way of doing things is to reduce the total global memory accesses, as they are slow.
 */

__constant__ float poly6_scaling;
__constant__ float spiky_scaling;
__constant__ float viscosity_scaling;

/*"objects" for thrust (my limited C++ knowledge might be showing here)
 * they are kept extremely simple, C++ was not the main focus of this project.
 */
struct compare_entry
{
    __host__ __device__ bool operator()(const entry &a, const entry &b)
    {
        return a.cell_key < b.cell_key;
    }
};

//Telling thrust to use this custom allocator which caches memory increased performance some more, saving time on repetitive cudaMalloc and cudaFree calls.
struct cached_allocator
{
    typedef char value_type;

    int is_allocated;
    char *allocated_memory;

    cached_allocator() { is_allocated = 0; allocated_memory = nullptr;}

    ~cached_allocator()
    {
        if(is_allocated)
            cudaFree(allocated_memory);
    }

    char *allocate(std::ptrdiff_t num_bytes)
    {
        if(!is_allocated)
        {
            cudaMalloc(&allocated_memory, num_bytes);
            is_allocated = 1;
            return allocated_memory;
        }
        return allocated_memory;
    }

    void deallocate(char *ptr, std::ptrdiff_t num_bytes)
    {
        //exists only to satisfy thrust
    }
};

#ifdef __cplusplus
extern "C" {
#endif

__host__ void malloc_simulation_gpu(int n_particles, particles *sim, obstacles *obs)
{
    cudaMalloc(&(sim->gpu_density), sizeof(float) * n_particles);
    cudaMalloc(&(sim->gpu_pos), sizeof(float_v2) * n_particles);
    cudaMalloc(&(sim->gpu_p_pos), sizeof(float_v2) * n_particles);
    cudaMalloc(&(sim->gpu_vel), sizeof(float_v2) * n_particles);
    cudaMalloc(&(sim->gpu_pressure), sizeof(float_v2) * n_particles);
    cudaMalloc(&(sim->gpu_viscosity), sizeof(float_v2) * n_particles);
    cudaMalloc(&(sim->gpu_near_density), sizeof(float) * n_particles);
    cudaMalloc(&(obs->gpu_obstacles), sizeof(float_b) * obs->n_obstacles);

    cudaMemcpy(sim->gpu_pos, sim->pos, sizeof(float_v2) * n_particles, cudaMemcpyHostToDevice);
    cudaMemcpy(sim->gpu_p_pos, sim->pos, sizeof(float_v2) * n_particles, cudaMemcpyHostToDevice);
    cudaMemcpy(obs->gpu_obstacles, obs->obstacles, sizeof(float_b) * obs->n_obstacles, cudaMemcpyHostToDevice);

    cudaMemset(sim->gpu_vel, 0, sizeof(float_v2) * n_particles);
    cudaMemset(sim->gpu_pressure, 0, sizeof(float_v2) * n_particles);
    cudaMemset(sim->gpu_viscosity, 0, sizeof(float_v2) * n_particles);
    cudaMemset(sim->gpu_density, 0, sizeof(float) * n_particles);
    cudaMemset(sim->gpu_near_density, 0, sizeof(float) * n_particles);
}

__host__  void simulation_step_gpu(int_v2 container, particles *sim, settings s, cells *cell_ll, obstacles *obs)
{
    static cached_allocator alloc;

    const int n_particles = s.ss.n_particles;
    int num_blocks = (n_particles + BLOCK_SIZE - 1) / BLOCK_SIZE;

    update_cells<<<num_blocks, BLOCK_SIZE>>>(cell_ll->entries, cell_ll->start_indices, sim->gpu_p_pos, s);
    cudaDeviceSynchronize();

    //this "strangely!!!" increased performance by a ton compared to sort_entries
    thrust::device_ptr<entry> entries_ptr(cell_ll->entries);
    thrust::sort(thrust::cuda::par(alloc), entries_ptr, entries_ptr + n_particles, compare_entry());
    cudaDeviceSynchronize();

    set_start_indices<<<num_blocks, BLOCK_SIZE>>>(cell_ll->entries, cell_ll->start_indices, n_particles);
    cudaDeviceSynchronize();

    set_predicted_positions<<<num_blocks, BLOCK_SIZE>>>(sim->gpu_pos, sim->gpu_p_pos, sim->gpu_vel, container, s, *obs);
    cudaDeviceSynchronize();

    set_densities<<<num_blocks, BLOCK_SIZE>>>(sim->gpu_p_pos, sim->gpu_density, sim->gpu_near_density, s, cell_ll->entries, cell_ll->start_indices, cell_ll->world_size);
    cudaDeviceSynchronize();

    set_forces<<<num_blocks, BLOCK_SIZE>>>(sim->gpu_pressure, sim->gpu_viscosity, sim->gpu_p_pos, sim->gpu_near_density, sim->gpu_density, sim->gpu_vel, s, container, cell_ll->entries, cell_ll->start_indices);
    cudaDeviceSynchronize();

    set_velocities<<<num_blocks, BLOCK_SIZE>>>(sim->gpu_vel, sim->gpu_pressure, sim->gpu_viscosity, s);
    cudaDeviceSynchronize();

    set_positions<<<num_blocks, BLOCK_SIZE>>>(sim->gpu_pos, sim->gpu_vel, container, s, *obs);
    cudaDeviceSynchronize();
}

__host__ void create_cell_ll_gpu(cells *cell_ll, RECT rect, settings s)
{
    int_v2 size = { (int) ceilf( (float) rect.right / s.ss.smoothing_length), (int) ceilf( (float) rect.bottom / s.ss.smoothing_length)};

    cell_ll->world_size = size;
    cudaMalloc(&(cell_ll->entries), sizeof(entry) * s.ss.n_particles);
    cudaMalloc(&(cell_ll->start_indices), sizeof(int) * s.ss.n_particles);

}

__host__  void initialize_constants(float poly6, float spiky, float viscosity)
{
    cudaMemcpyToSymbol(poly6_scaling, &poly6, sizeof(float));
    cudaMemcpyToSymbol(spiky_scaling, &spiky, sizeof(float));
    cudaMemcpyToSymbol(viscosity_scaling, &viscosity, sizeof(float));
}

/* legacy sort_entries function, meant to be used with the bitonic sort implementation.
 * Entry array was padded to the next power of 2, and filled with INT_MAX values.
 * cells struct had a padded_size variable.
 *
__host__ void sort_entries(entry *entries, int padded_size)
{
    int block_size = 256;
    int num_blocks = (padded_size + block_size - 1) / block_size;

    for(int k = 2; k <= padded_size; k <<= 1)
    {
        for(int j = k >> 1; j > 0; j >>= 1)
        {
            bitonic_sort<<<num_blocks, block_size>>>(entries, padded_size, j, k);
            cudaDeviceSynchronize();
        }
    }
}*/

__host__ void free_simulation_memory(particles *sim)
{
    cudaFree(sim->gpu_density);
    cudaFree(sim->gpu_pos);
    cudaFree(sim->gpu_p_pos);
    cudaFree(sim->gpu_vel);
    cudaFree(sim->gpu_pressure);
    cudaFree(sim->gpu_viscosity);
    cudaFree(sim->gpu_near_density);
}

#ifdef __cplusplus
}
#endif

/* Smoothing kernels used to calculate the influence of particles on each other. Opposed to the poly6 kernel, which smooths out at the
 * peak, the spiky kernel remains spiky: it's used in pressure calculation where we want higher pressures as distance decreases, opposed to
 * how the poly6 kernel calculates similar densities at close distances.
 * Both the poly6 and spiky kernels have a near-density & near-pressure version, which is spikier than the normal version.
 * All kernels (Except the spikier version ones) were taken from:
 * https://www.researchgate.net/publication/221622709_SPH_Based_Shallow_Water_Simulation
 */

__device__ float poly6_smoothing_kernel(float r_2, float h_2)
{
    if(r_2 >= h_2)
        return 0;

    return poly6_scaling * (h_2 - r_2) * (h_2 - r_2) * (h_2 - r_2);
}

__device__ float spiky_smoothing_kernel(float r, float h)
{
    if(r >= h)
        return 0;

    return spiky_scaling * (h - r) * (h - r) * (h - r);
}

__device__ float viscosity_laplacian_kernel(float r, float h)
{
    if(r >= h)
        return 0;

    return viscosity_scaling * (h - r);
}

__device__ float optimized_near_density_kernel(float poly6, float r_2, float h_2)
{
    /* The near density kernel, spikier version of poly6 would be : poly6_scaling * (h_2 - r_2) * (h_2 - r_2) * (h_2 - r_2) * (h_2 - r_2);
     * because we already compute the first (h_2 - r_2) * (h_2 - r_2) * (h_2 - r_2) in the poly6 kernel, we can just multiply it by (h_2 - r_2) again.
     * this saves us some extra multiplications, and we don't have to check again if(r_2 >= h_2) because poly6 already accounts for it.
     */
    return poly6 * (h_2 - r_2);
}

__device__ float optimized_near_pressure_kernel(float spiky, float r, float h)
{
    /* Normally would be: spiky_scaling * ( h - r ) * ( h - r ) * ( h - r ) * ( h - r );
     * See: optimized_near_density_kernel for explanation.
     */
    return spiky * (h - r);
}

__global__ void set_densities(float_v2 *predicted_positions, float *densities, float *near_densities, settings s, entry *entries, int *start_indices, int_v2 size)
{
    int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);

    if(i >= s.ss.n_particles)
        return;

    float_v2 density_pair = calculate_density(predicted_positions, predicted_positions[i], s, entries, start_indices, size);
    densities[i] = density_pair.x;
    near_densities[i] = density_pair.y;
}

/* The calculation of a certain particle density in SPH is pretty simple: multiply the other particle mass * its influence on the
 * main particle.
 */

//Changed up to use position directly instead of particle number
__device__ float_v2 calculate_density(float_v2 *predicted_positions, float_v2 particle_pos, settings s, entry *entries, int *start_indices, int_v2 size)
{
    const float h = s.ss.smoothing_length;
    const float h_2 = h * h;
    const float mass = s.ss.mass;
    const int n_particles = s.ss.n_particles;

    const int_v2 center = cell_coordinate(particle_pos, h);
    const int_v2 max_pos = cell_coordinate({(float) size.x, (float) size.y}, h);

    float_v2 density_pair = {0, 0};

    //pre-calculating bounds instead of checking them inside the loop every time

    const int min_x = max(0, center.x - 1);
    const int max_x = min(max_pos.x, center.x + 1);
    const int min_y = max(0, center.y - 1);
    const int max_y = min(max_pos.y, center.y + 1);

    for(int x = min_x; x <= max_x; x++)
    {
        for(int y = min_y; y <= max_y; y++)
        {
            const int_v2 cell = {x, y};
            const int cell_key = hash_cell(n_particles, cell);
            const int cell_start = start_indices[cell_key];

            if(cell_start == -1)
                continue;

            for(int i = cell_start; i < n_particles; i++)
            {
                const entry current_entry = entries[i];
                //Double-checking hash and location
                if(current_entry.cell_key != cell_key)
                    break;

                const int pi = current_entry.particle_index;
                const float distance_2 = get_distance_2(particle_pos, predicted_positions[pi]);

                if(distance_2 > h_2)
                    continue;

                const float influence = poly6_smoothing_kernel(distance_2, h_2);

                density_pair.x += mass * influence;
                density_pair.y += mass * optimized_near_density_kernel(influence, distance_2, h_2);
            }
        }
    }

    return density_pair;
}

__device__ inline float calculate_symmetric_near_pressure(float near_density1, float near_density2, settings s)
{
    float near_pressure1, near_pressure2;

    near_pressure1 = near_density_to_pressure(near_density1, s);
    near_pressure2 = near_density_to_pressure(near_density2, s);

    return (near_pressure1+near_pressure2)/(2*near_density1);
}

__device__ inline float calculate_symmetric_pressure(float density1, float density2, settings s)
{
    float pressure1, pressure2;

    pressure1 = density_to_pressure(density1, s);
    pressure2 = density_to_pressure(density2, s);

    return (pressure1+pressure2)/(2*density1);
}

__device__ inline float near_density_to_pressure(float near_density, settings s)
{
    return s.ss.near_pressure_multiplier*near_density;
}

__device__ inline float density_to_pressure(float density, settings s)
{
    return s.ss.pressure_multiplier*(density - s.ss.rest_density);
}

__global__ void set_positions(float_v2 *positions, float_v2 *velocities, int_v2 container, settings s, obstacles obs)
{
    int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);

    if(i >= s.ss.n_particles)
        return;

    const float ts = s.ss.time_step;
    const float bm = s.ss.bounce_multiplier;

    float_v2 vel = velocities[i];
    float_v2 pos = positions[i];

    pos.x += vel.x*ts;
    pos.y += vel.y*ts;

    //Resolving collision with walls.
    if(pos.x < 0)
    {
        pos.x = 0;
        vel.x = -vel.x * bm;
    }
    else if(pos.x > (float) container.x)
    {
        pos.x = (float) container.x;
        vel.x = -vel.x * bm;
    }

    if(pos.y < 0)
    {
        pos.y = 0;
        vel.y = -vel.y * bm;
    }
    else if(pos.y > (float) container.y)
    {
        pos.y = (float) container.y;
        vel.y = -vel.y * bm;
    }

    resolve_obstacles(&pos, &vel, &obs, bm);

    velocities[i] = vel;
    positions[i] = pos;
}

/* Using predicted positions instead of the real particle positions in the simulation helps at managing the chaos which the simulation
 * can become.
 */
__global__ void set_predicted_positions(float_v2 *positions, float_v2 *predicted_positions, float_v2 *velocities, int_v2 container, settings s, obstacles obs)
{
    int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);

    if(i >= s.ss.n_particles)
        return;

    float_v2 p_pos = {0, 0};
    float_v2 pos = positions[i];
    float_v2 vel = velocities[i];

    p_pos.x = pos.x + vel.x*s.ss.prediction_time_step;
    p_pos.y = pos.y + vel.y*s.ss.prediction_time_step;

    if(p_pos.x < 0)
    {
        p_pos.x = 0;
    }
    else if(p_pos.x >= (float) container.x)
    {
        p_pos.x = (float) container.x;
    }

    if(p_pos.y < 0)
    {
        p_pos.y = 0;
    }
    else if(p_pos.y >= (float) container.y)
    {
        p_pos.y = (float) container.y;
    }

    resolve_obstacles(&p_pos, NULL, &obs, s.ss.bounce_multiplier);

    predicted_positions[i] = p_pos;
}

__global__ void set_velocities(float_v2 *velocities, float_v2 *pressures, float_v2 *viscosities, settings s)
{
    int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);

    if(i >= s.ss.n_particles)
        return;

    const float ts = s.ss.time_step;
    float_v2 pressure = pressures[i];
    float_v2 viscosity = viscosities[i];
    float_v2 velocity = velocities[i];

    float_v2 total_force = {pressure.x + viscosity.x, pressure.y + viscosity.y};

    velocity.x += (total_force.x / s.ss.mass) * ts;
    velocity.y += (total_force.y / s.ss.mass) * ts;

    velocity.y += s.ss.gravity * ts;

    velocities[i] = velocity;
}

/* Translated world pos into grid pos, used to determine which cell a particle is in.
 * Every cell is smoothing_length wide.
 */
__device__ int_v2 cell_coordinate(float_v2 pos, float h)
{
    int_v2 cell;
    const float h_inv = 1.0f / h;
    cell.x = (int) floorf(pos.x*h_inv);
    cell.y = (int) floorf(pos.y*h_inv);

    return cell;
}

/* Simplest hash function I could come up with, surely has room for improvement.
 * Values multiplied with prime numbers to increase spread, reduce collisions.
 * Lastly, values clamped between 0 and max_hash-1.
 */
__device__ int hash_cell(int max_hash, int_v2 pos)
{
    const unsigned int hashx = pos.x * 11;
    const unsigned int hashy = pos.y * 4079;
    return ((hashx + hashy)*7) % max_hash;
}

__device__ float get_distance_2(float_v2 pos1, float_v2 pos2)
{
    return (pos1.x - pos2.x) * (pos1.x - pos2.x) + (pos1.y - pos2.y) * (pos1.y - pos2.y);
}

/* To avoid checking every particle with every other particle, getting O(n^2) complexity, we divide our world into cells.
 * every cell is smoothing_length wide, this way we have to check the adjacent cells to the one we are in to find all the neighbours.
 * As every iteration the particles position changes, we have to update the cells every iteration.
 */
__global__ void update_cells(entry *entries, int *start_indices, float_v2 *predicted_positions, settings s)
{
    int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);

    if(i >= s.ss.n_particles)
        return;

    int_v2 cell_pos = cell_coordinate(predicted_positions[i], s.ss.smoothing_length);
    int cell_key = hash_cell(s.ss.n_particles, cell_pos);

    entries[i] = {cell_key, i};
    start_indices[i] = -1;
}

/* Old code for sorting, kept for reference. Really slow, took more than the whole step.
 * Probably because of the way I implemented it, and how many times I call cudaDeviceSynchronize.
 * Pretty sure it could have been implemented in a better way, but there is people who spent
 * years working on sorting algorithms, so I'll just use their work.
 *
__global__ void bitonic_sort(entry *entries, int n_entries, int j, int k)
{
    int i = (int) (threadIdx.x + blockDim.x * blockIdx.x);

    if(i >= n_entries)
        return;

    int ij = i ^ j;

    if (ij > i)
    {
        if ((i & k) == 0)
        {
            if (entries[i].cell_key > entries[ij].cell_key)
            {
                entry temp = entries[i];
                entries[i]= entries[ij];
                entries[ij]= temp;
            }
        }
        else
        {
            if(entries[i].cell_key < entries[ij].cell_key)
            {
                entry temp = entries[i];
                entries[i] = entries[ij];
                entries[ij] = temp;
            }
        }
    }
}

__global__ void pad_entries(entry *entries, int n_particles, int padded_size)
{
    int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    int entries_index = n_particles + i;

    if(entries_index >= padded_size)
        return;

    entries[entries_index] = {INT_MAX, -1};
}
*/

/* To rapidly find the start of each cell in the entries array, we set the starting index of each cell
 * in the start_indices array, entry x start location will be stored in start_indices[x].
 * We can also quickly filter out empty cells, as their starting index will be -1.
 * if our entries cell_keys array is:     [ 0, 0, 3, 3, 5, 5, 7, 7]
 * then our start_indices array would be: [ 0,-1,-1, 2,-1, 4,-1, 6]
 */
__global__ void set_start_indices(entry *entries, int *start_indices, int n_particles)
{
    int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);

    if(i >= n_particles)
        return;

    int key = entries[i].cell_key;
    int prev_key = -1;

    if(i != 0)
        prev_key = entries[i-1].cell_key;

    if(key != prev_key)
    {
        start_indices[key] = i;
    }
}

/* Pressure calculation uses symmetric pressure to follow newton's third law.
 * Together with pressure we calculate near pressure: our density to pressure function can
 * return negative values, which means, it can make particles attract each other. This becomes
 * a pretty big problem when we have some messed up settings. Near pressure is used to avoid
 * particles from getting stuck to each other.
 *
 * The initial calculate_pressure and calculate_viscosity were merged in this single function. As both didn't depend
 * on each other, calculating them together is faster as we don't have to search neighbours of a particle twice.
 */
__global__ void set_forces(float_v2 *pressures, float_v2 *viscosities, float_v2 *predicted_positions, float *near_densities, float *densities, float_v2 *velocities, settings s, int_v2 size, entry *entries, int *start_indices)
{
    int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);

    const int n_particles = s.ss.n_particles;

    if(i >= n_particles)
        return;

    const float h = s.ss.smoothing_length;
    const float h_2 = h * h;
    const float mass = s.ss.mass;
    const float viscosity_multiplier = s.ss.viscosity;

    const float density_p1 = densities[i];
    const float ndensity_p1 = near_densities[i];
    const float_v2 pposition_p1 = predicted_positions[i];
    const float_v2 velocity_p1 = velocities[i];

    const int_v2 center = cell_coordinate(pposition_p1, h);
    const int_v2 max_pos = cell_coordinate({(float) size.x, (float) size.y}, h);

    float_v2 pressure = {0, 0};
    float_v2 viscosity = {0, 0};

    const int min_x = max(0, center.x - 1);
    const int max_x = min(max_pos.x, center.x + 1);
    const int min_y = max(0, center.y - 1);
    const int max_y = min(max_pos.y, center.y + 1);

    for(int x = min_x; x <= max_x; x++)
    {
        for(int y = min_y; y <= max_y; y++)
        {
            const int_v2 cell = {x, y};
            const int cell_key = hash_cell(n_particles, cell);
            const int cell_start = start_indices[cell_key];

            if(cell_start == -1)
                continue;

            for(int j = cell_start; j < n_particles; j++)
            {
                const entry current_entry = entries[j];
                //Double-checking hash and location
                if(current_entry.cell_key != cell_key)
                    break;

                const int pi = current_entry.particle_index;

                if(pi == i)
                    continue; //ignore self

                const float_v2 pposition_p2 = predicted_positions[pi];
                const float distance_2 = get_distance_2(pposition_p1, pposition_p2);

                if(distance_2 > h_2)
                    continue;

                //viscosity calculation
                const float density_p2 = densities[pi];
                const float distance = sqrt(distance_2);

                const float influence = viscosity_laplacian_kernel(distance, h);
                const float_v2 velocity_p2 = velocities[pi];

                viscosity.x += (velocity_p2.x - velocity_p1.x) * influence;
                viscosity.y += (velocity_p2.y - velocity_p1.y) * influence;

                /* I am not quite sure how much this density_p2 != 0 here is needed.
                 * It is needed, later on, to avoid division by 0. I placed it here, in hopes of avoiding threads from doing unnecessary work,
                 * but because it creates branching, the threads which won't do this unnecessary work will have to wait for the ones that do.
                 * For clarity, I will leave it here, as I think it makes no change in performance.
                 */

                if(density_p2 != 0)
                {
                    float_v2 dir;

                    if(distance != 0)
                    {
                        const float inv_dst = 1.0f / distance;
                        dir.x = (pposition_p1.x - pposition_p2.x) * inv_dst;
                        dir.y = (pposition_p1.y - pposition_p2.y) * inv_dst;
                    }
                    else
                    {
                        curandState state;
                        curand_init(0, i, 0, &state);

                        /* Generate pseudo-random number if the distance between particles is 0. Often happens when particles
                         * are either spawned on top of each-other, or when they meet the bounds at the same time, and get placed in the same
                         * position afterward.
                         * Number can freely be pseudo-random, what matters is that the 2 particles with same position go into different directions.
                         */

                        dir.x = curand_uniform(&state) * 2.0f - 1.0f;
                        dir.y = curand_uniform(&state) * 2.0f - 1.0f;
                    }

                    const float sp = calculate_symmetric_pressure(density_p1, density_p2, s);
                    const float slope = spiky_smoothing_kernel(distance, h);

                    const float near_sp = calculate_symmetric_near_pressure(ndensity_p1, near_densities[pi], s);
                    const float near_slope = optimized_near_pressure_kernel(slope, distance, h);

                    /*In hopes of reducing the amount of calculations, I compacted pressure and near_pressure calculations.
                     * I leave here what they were before the compacting, for anyone who is trying to learn from this code:
                     * pressure.x += (-mass * sp * slope * dir.x);
                     * pressure.y += (-mass * sp * slope * dir.y);
                     * near_pressure.x += -mass * near_sp * near_slope * dir.x;
                     * near_pressure.y += -mass * near_sp * near_slope * dir.y;
                     * pressure would, at the end, be: {pressure.x + near_pressure.x, pressure.y + near_pressure.y};
                     */
                    pressure.x += (-mass * dir.x) * (sp * slope + near_sp * near_slope);
                    pressure.y += (-mass * dir.y) * (sp * slope + near_sp * near_slope);
                }
            }
        }
    }

    viscosities[i] = {viscosity.x * viscosity_multiplier, viscosity.y * viscosity_multiplier};
    pressures[i] = pressure;
}

__device__ void resolve_obstacles(float_v2 *pos, float_v2 *vel, obstacles *obs, float bm)
{
    for(int i = 0; i < obs->n_obstacles; i++)
    {
        float_b current_obstacle = obs->gpu_obstacles[i];

        /* Collision detection with obstacles: first we check if the particle is actually inside the obstacle, so if it has collided.
         * After that we figure out whenever we push the particle right or left. We do so by calculating the obstacle center and checking what
         * side the particle is closer to. If it's closer to the left side, we push it right, and vice versa.
         */
        if ((pos->x < current_obstacle.min.x || pos->x > current_obstacle.max.x) ||
           (pos->y < current_obstacle.min.y || pos->y > current_obstacle.max.y))
            continue;

        const float_v2 center = {(current_obstacle.min.x + current_obstacle.max.x) * 0.5f, (current_obstacle.min.y + current_obstacle.max.y) * 0.5f};

        const float pen_x = pos->x > center.x ? current_obstacle.max.x - pos->x : current_obstacle.min.x - pos->x;
        const float pen_y = pos->y > center.y ? current_obstacle.max.y - pos->y : current_obstacle.min.y - pos->y;

        if(abs(pen_x) < abs(pen_y))
        {
            pos->x += pen_x;
            if(vel != NULL)
                vel->x = -vel->x * bm;
        }
        else
        {
            pos->y += pen_y;
            if(vel != NULL)
                vel->y = -vel->y * bm;
        }
    }
}