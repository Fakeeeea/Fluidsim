#include <cuda_runtime.h>

#include "sim_math.cuh"
#include "types.h"

__constant__ float poly6_scaling;
__constant__ float spiky_scaling;
__constant__ float viscosity_scaling;

#ifdef __cplusplus
extern "C" {
#endif

__host__ void malloc_simulation_gpu(int n_particles, particles *sim)
{
    cudaMalloc(&(sim->gpu_density), sizeof(float) * n_particles);
    cudaHostAlloc(&(sim->gpu_pos), sizeof(float_v2) * n_particles, cudaHostAllocDefault);
    cudaMalloc(&(sim->gpu_p_pos), sizeof(float_v2) * n_particles);
    cudaMalloc(&(sim->gpu_vel), sizeof(float_v2) * n_particles);
    cudaMalloc(&(sim->gpu_pressure), sizeof(float_v2) * n_particles);
    cudaMalloc(&(sim->gpu_viscosity), sizeof(float_v2) * n_particles);
    cudaMalloc(&(sim->gpu_near_density), sizeof(float) * n_particles);

    cudaMemcpy(sim->gpu_pos, sim->pos, sizeof(float_v2) * n_particles, cudaMemcpyHostToDevice);
    cudaMemcpy(sim->gpu_p_pos, sim->pos, sizeof(float_v2) * n_particles, cudaMemcpyHostToDevice);

    cudaMemset(sim->gpu_vel, 0, sizeof(float_v2) * n_particles);
    cudaMemset(sim->gpu_pressure, 0, sizeof(float_v2) * n_particles);
    cudaMemset(sim->gpu_viscosity, 0, sizeof(float_v2) * n_particles);
    cudaMemset(sim->gpu_density, 0, sizeof(float) * n_particles);
    cudaMemset(sim->gpu_near_density, 0, sizeof(float) * n_particles);
}

__host__  void simulation_step_gpu(int_v2 container, particles *sim, settings s, cells *cell_ll)
{
    const int n_particles = s.ss.n_particles;
    int block_size = 256;
    int num_blocks = (n_particles + block_size - 1) / block_size;

    update_cells<<<num_blocks, block_size>>>(cell_ll->entries, cell_ll->start_indices, sim->gpu_p_pos, s);
    cudaDeviceSynchronize();

    sort_entries(cell_ll->entries, n_particles);
    cudaDeviceSynchronize();

    set_start_indices<<<num_blocks, block_size>>>(cell_ll->entries, cell_ll->start_indices, n_particles);
    cudaDeviceSynchronize();

    set_predicted_positions<<<num_blocks, block_size>>>(sim->gpu_pos, sim->gpu_p_pos, sim->gpu_vel, container, s);
    cudaDeviceSynchronize();

    set_densities<<<num_blocks, block_size>>>(sim->gpu_p_pos, sim->gpu_density, sim->gpu_near_density, s, cell_ll->entries, cell_ll->start_indices, cell_ll->world_size);
    cudaDeviceSynchronize();

    set_pressures<<<num_blocks, block_size>>>(sim->gpu_p_pos, sim->gpu_density, sim->gpu_near_density, sim->gpu_pressure, s, cell_ll->entries, cell_ll->start_indices, cell_ll->world_size);
    cudaDeviceSynchronize();

    set_viscosities<<<num_blocks, block_size>>>(sim->gpu_vel, sim->gpu_p_pos, sim->gpu_viscosity, s, cell_ll->world_size, cell_ll->entries, cell_ll->start_indices);
    cudaDeviceSynchronize();

    set_velocities<<<num_blocks, block_size>>>(sim->gpu_vel, sim->gpu_pressure, sim->gpu_viscosity, s);
    cudaDeviceSynchronize();

    set_positions<<<num_blocks, block_size>>>(sim->gpu_pos, sim->gpu_vel, container, s);
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

__host__ void sort_entries(entry *entries, int n_particles)
{
    int padded_size = 1;

    while(padded_size < n_particles)
        padded_size <<= 1;

    //Pad array with INT_MAX
    entry *cpu_padded_entries = (entry*) malloc(sizeof(entry) * padded_size);

    cudaMemcpy(cpu_padded_entries, entries, sizeof(entry) * n_particles, cudaMemcpyDeviceToHost);

    for(int i = n_particles; i < padded_size; i++)
    {
        cpu_padded_entries[i] = {INT_MAX, -1};
    }

    entry *gpu_padded_entries;

    //Send padded array to GPU
    cudaMalloc(&gpu_padded_entries, sizeof(entry) * padded_size);
    cudaMemcpy(gpu_padded_entries, cpu_padded_entries, sizeof(entry) * padded_size, cudaMemcpyHostToDevice);

    int block_size = 256;
    int num_blocks = (padded_size + block_size - 1) / block_size;

    for(int k = 2; k <= padded_size; k <<= 1)
    {
        for(int j = k >> 1; j > 0; j >>= 1)
        {
            bitonic_sort<<<num_blocks, block_size>>>(gpu_padded_entries, padded_size, j, k);
            cudaDeviceSynchronize();
        }
    }

    //Copy sorted array back to original array
    cudaMemcpy(entries, gpu_padded_entries, sizeof(entry) * n_particles, cudaMemcpyDeviceToDevice);

    //Free
    cudaFree(gpu_padded_entries);
    free(cpu_padded_entries);
}

__host__ void free_simulation_memory(particles *sim)
{
    cudaFree(sim->gpu_density);
    cudaFreeHost(sim->gpu_pos);
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

__device__ float poly6_near_density_kernel(float r_2, float h_2)
{
    if(r_2 >= h_2)
        return 0;

    return poly6_scaling * (h_2 - r_2) * (h_2 - r_2) * (h_2 - r_2) * (h_2 - r_2);
}

__device__ float spiky_near_pressure_kernel(float r, float h)
{
    if(r >= h)
        return 0;

    return spiky_scaling * (h - r) * (h - r) * (h - r) * (h - r);
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

    float_v2 density_pair = {0, 0};

    //pre-calculating bounds instead of checking them inside the loop every time

    const int min_x = center.x - 1 < 0 ? 0 : center.x - 1;
    const int max_x = center.x + 1 >= size.x ? center.x : center.x + 1;
    const int min_y = center.y - 1 < 0 ? 0 : center.y - 1;
    const int max_y = center.y + 1 >= size.y ? center.x : center.y + 1;

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

                density_pair.x += mass * poly6_smoothing_kernel(distance_2, h_2);
                density_pair.y += mass * poly6_near_density_kernel(distance_2, h_2);
            }
        }
    }

    return density_pair;
}

__global__ void set_pressures(float_v2 *predicted_positions, float *densities, float *near_densities, float_v2 *pressures, settings s, entry *entries, int *start_indices, int_v2 size)
{
    int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);

    if(i >= s.ss.n_particles)
        return;

    pressures[i] = calculate_pressure(predicted_positions, densities, near_densities, i, s, entries, start_indices, size);
}

/* Pressure calculation uses symmetric pressure to follow newton's third law.
 * Together with pressure we calculate near pressure: our density to pressure function can
 * return negative values, which means, it can make particles attract each other. This becomes
 * a pretty big problem when we have some messed up settings. Near pressure is used to avoid
 * particles from getting stuck to each other.
 */

__device__ float_v2 calculate_pressure(float_v2 *predicted_positions, float *densities, float *near_densities, int p_number, settings s, entry *entries, int *start_indices, int_v2 size)
{
    const float h = s.ss.smoothing_length;
    const float h_2 = h * h;
    const float mass = s.ss.mass;
    const int n_particles = s.ss.n_particles;

    const int_v2 center = cell_coordinate(predicted_positions[p_number], h);
    float_v2 pressure = {0, 0};
    float_v2 near_pressure = {0, 0};

    //pre-calculating bounds instead of checking them inside the loop every time

    const int min_x = center.x - 1 < 0 ? 0 : center.x - 1;
    const int max_x = center.x + 1 >= size.x ? center.x : center.x + 1;
    const int min_y = center.y - 1 < 0 ? 0 : center.y - 1;
    const int max_y = center.y + 1 >= size.y ? center.x : center.y + 1;

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

                int pi = current_entry.particle_index;

                if(pi == p_number)
                    continue; //Ignore self

                const float distance_2 = get_distance_2(predicted_positions[p_number], predicted_positions[pi]);

                if(distance_2 > h_2)
                    continue;

                float_v2 dir = {0, 0};
                const float distance = sqrt(distance_2);

                if(distance > 1e-10)
                {
                    dir.x = (predicted_positions[p_number].x - predicted_positions[pi].x) / distance;
                    dir.y = (predicted_positions[p_number].y - predicted_positions[pi].y) / distance;
                }

                if(densities[pi] > 1e-10)
                {
                    const float sp = calculate_symmetric_pressure(densities[p_number], densities[pi], s);
                    const float slope = spiky_smoothing_kernel(distance, h);

                    pressure.x += -mass * sp * slope * dir.x;
                    pressure.y += -mass * sp * slope * dir.y;

                    const float near_sp = calculate_symmetric_near_pressure(near_densities[p_number], near_densities[pi], s);
                    const float near_slope = spiky_near_pressure_kernel(distance, h);

                    near_pressure.x += -mass * near_sp * near_slope * dir.x;
                    near_pressure.y += -mass * near_sp * near_slope * dir.y;
                }
            }
        }
    }

    //At the end, just sum the forces up
    pressure.x += near_pressure.x;
    pressure.y += near_pressure.y;

    return pressure;
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
    float near_pressure = s.ss.near_pressure_multiplier*near_density;

    return near_pressure;
}

__device__ inline float density_to_pressure(float density, settings s)
{
    float pressure = s.ss.pressure_multiplier*(density - s.ss.rest_density);

    //negative pressure (particles attracting each-other)
    //if(pressure < 0.0f)
    //  return 0;

    return pressure;
}

/*To avoid particles getting completely stuck on edges when we resize the window, we give them some speed. There
 * are surely better solutions, but this is the simplest work-around I found.
 */
__global__ void set_positions(float_v2 *positions, float_v2 *velocities, int_v2 container, settings s)
{
    int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);

    if(i >= s.ss.n_particles)
        return;


    positions[i].x += velocities[i].x*s.ss.time_step;
    positions[i].y += velocities[i].y*s.ss.time_step;


    if(positions[i].x < 0)
    {
        positions[i].x = 0;

        //if(velocities[i].x > -0.5 && velocities[i].x < 0)
        //    velocities[i].x = 2;
        //else
            velocities[i].x = -velocities[i].x * s.ss.bounce_multiplier;
    }
    else if(positions[i].x >= (float) container.x)
    {
        positions[i].x = (float) container.x;

        //if(velocities[i].x < 0.5 && velocities[i].x > 0)
        //    velocities[i].x = -2;
        //else
            velocities[i].x = -velocities[i].x * s.ss.bounce_multiplier;
    }

    if(positions[i].y < 0)
    {
        positions[i].y = 0;
        velocities[i].y = -velocities[i].y * s.ss.bounce_multiplier;
    }
    else if(positions[i].y >= (float) container.y)
    {
        positions[i].y = (float) container.y;
        velocities[i].y = -velocities[i].y * s.ss.bounce_multiplier;
    }
}

/* Using predicted positions instead of the real particle positions in the simulation helps at managing the chaos which the simulation
 * can become.
 */
__global__ void set_predicted_positions(float_v2 *positions, float_v2 *predicted_positions, float_v2 *velocities, int_v2 container, settings s)
{
    int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);

    if(i >= s.ss.n_particles)
        return;

    predicted_positions[i].x = positions[i].x + velocities[i].x*s.ss.time_step;
    predicted_positions[i].y = positions[i].y + velocities[i].y*s.ss.time_step;

    if(predicted_positions[i].x < 0)
    {
        predicted_positions[i].x = 0;
    }
    else if(predicted_positions[i].x >= (float) container.x)
    {
        predicted_positions[i].x = (float) container.x;
    }

    if(predicted_positions[i].y < 0)
    {
        predicted_positions[i].y = 0;
    }
    else if(predicted_positions[i].y >= (float) container.y)
    {
        predicted_positions[i].y = (float) container.y;
    }
}

__global__ void set_velocities(float_v2 *velocities, float_v2 *pressures, float_v2 *viscosities, settings s)
{
    int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);

    if(i >= s.ss.n_particles)
        return;

    float_v2 total_force = {pressures[i].x + viscosities[i].x, pressures[i].y + viscosities[i].y};

    velocities[i].x += (total_force.x / s.ss.mass)*s.ss.time_step;
    velocities[i].y += (total_force.y / s.ss.mass)*s.ss.time_step;

    velocities[i].y += -9.81f*s.ss.time_step;

}


__device__ int_v2 cell_coordinate(float_v2 pos, float h)
{
    int_v2 cell;

    cell.x = (int) floorf(pos.x/h);
    cell.y = (int) floorf(pos.y/h);

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

__global__ void set_viscosities(float_v2 *velocities, float_v2 *predicted_positions, float_v2 *viscosities, settings s, int_v2 size, entry *entries, int *start_indices)
{
    int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);

    if(i >= s.ss.n_particles)
        return;

    viscosities[i] = calculate_viscosity(velocities, predicted_positions, i, s, size, entries, start_indices);
}

__device__ float_v2 calculate_viscosity(float_v2 *velocities, float_v2 *predicted_positions, int p_number, settings s, int_v2 size, entry *entries, int *start_indices)
{
    const float h = s.ss.smoothing_length;
    const float h_2 = h * h;
    const float viscosity_multiplier = s.ss.viscosity;
    const int n_particles = s.ss.n_particles;

    const int_v2 center = cell_coordinate(predicted_positions[p_number], h);

    float_v2 viscosity = {0, 0};

    const int min_x = center.x - 1 < 0 ? 0 : center.x - 1;
    const int max_x = center.x + 1 >= size.x ? center.x : center.x + 1;
    const int min_y = center.y - 1 < 0 ? 0 : center.y - 1;
    const int max_y = center.y + 1 >= size.y ? center.x : center.y + 1;

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

                if(pi == p_number)
                    continue; //Ignore self

                const float distance_2 = get_distance_2(predicted_positions[p_number], predicted_positions[pi]);

                if(distance_2 > h_2)
                    continue;

                const float influence = viscosity_laplacian_kernel(sqrt(distance_2), h);

                viscosity.x += (velocities[pi].x - velocities[p_number].x) * influence;
                viscosity.y += (velocities[pi].y - velocities[p_number].y) * influence;
            }
        }
    }

    viscosity.x *= viscosity_multiplier;
    viscosity.y *= viscosity_multiplier;

    return viscosity;
}