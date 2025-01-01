#include <cuda_runtime.h>
#include <stdio.h>

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
    cudaMalloc(&(sim->gpu_pos), sizeof(v2) * n_particles);
    cudaMalloc(&(sim->gpu_p_pos), sizeof(v2) * n_particles);
    cudaMalloc(&(sim->gpu_vel), sizeof(v2) * n_particles);
    cudaMalloc(&(sim->gpu_pressure), sizeof(v2) * n_particles);
    cudaMalloc(&(sim->gpu_viscosity), sizeof(v2) * n_particles);
    cudaMalloc(&(sim->gpu_near_density), sizeof(float) * n_particles);

    cudaMemcpy(sim->gpu_pos, sim->pos, sizeof(v2) * n_particles, cudaMemcpyHostToDevice);
    cudaMemcpy(sim->gpu_p_pos, sim->pos, sizeof(v2) * n_particles, cudaMemcpyHostToDevice);

    cudaMemset(sim->gpu_vel, 0, sizeof(v2) * n_particles);
    cudaMemset(sim->gpu_pressure, 0, sizeof(v2) * n_particles);
    cudaMemset(sim->gpu_viscosity, 0, sizeof(v2) * n_particles);
    cudaMemset(sim->gpu_density, 0, sizeof(float) * n_particles);
    cudaMemset(sim->gpu_near_density, 0, sizeof(float) * n_particles);
}

__host__  void simulation_step_gpu(v2 container, particles *sim, int N_PARTICLES, settings s, cells *cell_ll)
{
    int block_size = 256;
    int num_blocks = (N_PARTICLES + block_size - 1) / block_size;

    update_cells<<<num_blocks, block_size>>>(cell_ll->entries, cell_ll->start_indices, sim->gpu_p_pos, N_PARTICLES, s);
    cudaDeviceSynchronize();

    sort_entries(cell_ll->entries, N_PARTICLES);
    cudaDeviceSynchronize();

    set_start_indices<<<num_blocks, block_size>>>(cell_ll->entries, cell_ll->start_indices, N_PARTICLES);
    cudaDeviceSynchronize();

    set_predicted_positions<<<num_blocks, block_size>>>(sim->gpu_pos, sim->gpu_p_pos, sim->gpu_vel, N_PARTICLES, container, s);
    cudaDeviceSynchronize();

    set_densities<<<num_blocks, block_size>>>(sim->gpu_p_pos, sim->gpu_density, sim->gpu_near_density, N_PARTICLES, s, cell_ll->entries, cell_ll->start_indices, cell_ll->size);
    cudaDeviceSynchronize();

    set_pressures<<<num_blocks, block_size>>>(sim->gpu_p_pos, sim->gpu_density, sim->gpu_near_density, sim->gpu_pressure, N_PARTICLES, s, cell_ll->entries, cell_ll->start_indices, cell_ll->size);
    cudaDeviceSynchronize();

    set_viscosities<<<num_blocks, block_size>>>(sim->gpu_vel, sim->gpu_p_pos, sim->gpu_viscosity, N_PARTICLES, s, cell_ll->size, cell_ll->entries, cell_ll->start_indices);
    cudaDeviceSynchronize();

    set_velocities<<<num_blocks, block_size>>>(sim->gpu_vel, sim->gpu_pressure, sim->gpu_viscosity, N_PARTICLES, s);
    cudaDeviceSynchronize();

    set_positions<<<num_blocks, block_size>>>(sim->gpu_pos, sim->gpu_vel, container, N_PARTICLES, s);
    cudaDeviceSynchronize();

    cudaMemcpy(sim->pos, sim->gpu_pos, sizeof(v2) * N_PARTICLES, cudaMemcpyDeviceToHost);
}

__host__ void create_cell_ll_gpu(cells *cell_ll, int n_particles, RECT rect, settings s)
{
    v2 size = {ceilf(rect.right/s.smoothing_length), ceilf(rect.bottom/s.smoothing_length)};

    cell_ll->size = size;
    cudaMalloc(&(cell_ll->entries), sizeof(entry) * n_particles);
    cudaMalloc(&(cell_ll->start_indices), sizeof(int) * n_particles);

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
        cpu_padded_entries[i] = {INT_MAX, {0,0}, -1};
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

__global__ void set_densities(v2 *predicted_positions, float *densities, float *near_densities, int n_particles, settings s, entry *entries, int *start_indices, v2 size)
{
    int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);

    if(i >= n_particles)
        return;

    v2 density_pair = calculate_density(predicted_positions, predicted_positions[i], n_particles, s, entries, start_indices, size);

    densities[i] = density_pair.x;
    near_densities[i] = density_pair.y;
}

/* The calculation of a certain particle density in SPH is pretty simple: multiply the other particle mass * its influence on the
 * main particle.
 */

//Changed up to use position directly instead of particle number
__device__ v2 calculate_density(v2 *predicted_positions, v2 particle_pos, int n_particles, settings s, entry *entries, int *start_indices, v2 size)
{
    v2 center = cell_coordinate(particle_pos, s.smoothing_length);
    v2 density_pair = {0, 0};
    float h_2 = s.smoothing_length * s.smoothing_length;

    for(int offsetX = -1; offsetX <= 1; offsetX++)
    {
        if((center.x + offsetX) < 0 || (center.x + offsetX) >= size.x)
            continue;

        for(int offsetY = -1; offsetY <= 1; offsetY++)
        {
            if((center.y + offsetY) < 0 || (center.y + offsetY) >= size.y)
                continue;

            v2 cell = {center.x + offsetX, center.y + offsetY};
            int cell_key = hash_cell(n_particles, cell);
            int cell_start = start_indices[cell_key];

            if(cell_start == -1)
                continue;

            for(int i = cell_start; i < n_particles; i++)
            {
                //Double-checking hash and location
                if(entries[i].cell_key != cell_key || (entries[i].location.x != center.x + offsetX || entries[i].location.y != center.y + offsetY))
                    break;

                int pi = entries[i].particle_index;

                float distance_2 = get_distance_2(particle_pos, predicted_positions[pi]);

                if(distance_2 > h_2)
                    continue;

                density_pair.x += s.mass * poly6_smoothing_kernel(distance_2, h_2);
                density_pair.y += s.mass * poly6_near_density_kernel(distance_2, h_2);
            }
        }
    }

    return density_pair;
}

__global__ void set_pressures(v2 *predicted_positions, float *densities, float *near_densities, v2 *pressures, int n_particles, settings s, entry *entries, int *start_indices, v2 size)
{
    int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);

    if(i >= n_particles)
        return;

    pressures[i] = calculate_pressure(predicted_positions, densities, near_densities, i, n_particles, s, entries, start_indices, size);
}

/* Pressure calculation uses symmetric pressure to follow newton's third law.
 * Together with pressure we calculate near pressure: our density to pressure function can
 * return negative values, which means, it can make particles attract each other. This becomes
 * a pretty big problem when we have some messed up settings. Near pressure is used to avoid
 * particles from getting stuck to each other.
 */

__device__ v2 calculate_pressure(v2 *predicted_positions, float *densities, float *near_densities, int p_number, int n_particles, settings s, entry *entries, int *start_indices, v2 size)
{
    v2 center = cell_coordinate(predicted_positions[p_number], s.smoothing_length);
    v2 pressure = {0, 0};
    v2 near_pressure = {0, 0};
    float h_2 = s.smoothing_length * s.smoothing_length;

    for(int offsetX = -1; offsetX <= 1; offsetX++)
    {
        if((center.x + offsetX) < 0 || (center.x + offsetX) >= size.x)
            continue;

        for(int offsetY = -1; offsetY <= 1; offsetY++)
        {
            if((center.y + offsetY) < 0 || (center.y + offsetY) >= size.y)
                continue;

            v2 cell = {center.x + offsetX, center.y + offsetY};
            int cell_key = hash_cell(n_particles, cell);
            int cell_start = start_indices[cell_key];

            if(cell_start == -1)
                continue;

            for(int i = cell_start; i < n_particles; i++)
            {
                //Double-checking hash and location
                if(entries[i].cell_key != cell_key || (entries[i].location.x != (center.x + offsetX) || entries[i].location.y != (center.y + offsetY)))
                    break;

                int pi = entries[i].particle_index;

                if(pi == p_number)
                    continue; //Ignore self

                float distance_2 = get_distance_2(predicted_positions[p_number], predicted_positions[pi]);

                if(distance_2 > h_2)
                    continue;

                v2 dir = {0,0};
                float distance = sqrt(distance_2);
                if(distance > 1e-10)
                {
                    dir.x = (predicted_positions[p_number].x - predicted_positions[pi].x) / distance;
                    dir.y = (predicted_positions[p_number].y - predicted_positions[pi].y) / distance;
                }

                if(densities[pi] > 1e-10)
                {
                    float sp = calculate_symmetric_pressure(densities[p_number], densities[pi], s);
                    float slope = spiky_smoothing_kernel(distance, s.smoothing_length);

                    pressure.x += -s.mass * sp * slope * dir.x;
                    pressure.y += -s.mass * sp * slope * dir.y;

                    float near_sp = calculate_symmetric_near_pressure(near_densities[p_number], near_densities[pi], s);
                    float near_slope = spiky_near_pressure_kernel(distance, s.smoothing_length);

                    near_pressure.x += -s.mass * near_sp * near_slope * dir.x;
                    near_pressure.y += -s.mass * near_sp * near_slope * dir.y;
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
    float near_pressure = s.near_pressure_multiplier*near_density;

    return near_pressure;
}

__device__ inline float density_to_pressure(float density, settings s)
{
    float pressure = s.pressure_multiplier*(density - s.rest_density);

    //negative pressure (particles attracting eachother)
    //if(pressure < 0.0f)
    //  return 0;

    return pressure;
}

/*To avoid particles getting completely stuck on edges when we resize the window, we give them some speed. There
 * are surely better solutions, but this is the simplest work-around I found.
 */
__global__ void set_positions(v2 *positions, v2 *velocities, v2 grid_size, int n_particles, settings s)
{
    int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);

    if(i >= n_particles)
        return;


    positions[i].x += velocities[i].x*s.time_step;
    positions[i].y += velocities[i].y*s.time_step;


    if(positions[i].x < 0)
    {
        positions[i].x = 0;

        if(velocities[i].x > -0.5 && velocities[i].x < 0)
            velocities[i].x = 2;
        else
            velocities[i].x = -velocities[i].x * s.bounce_multiplier;
    }
    else if(positions[i].x >= grid_size.x)
    {
        positions[i].x = grid_size.x;

        if(velocities[i].x < 0.5 && velocities[i].x > 0)
            velocities[i].x = -2;
        else
            velocities[i].x = -velocities[i].x * s.bounce_multiplier;
    }

    if(positions[i].y < 0)
    {
        positions[i].y = 0;
        velocities[i].y = -velocities[i].y * s.bounce_multiplier;
    }
    else if(positions[i].y >= grid_size.y)
    {
        positions[i].y = grid_size.y;
        velocities[i].y = -velocities[i].y * s.bounce_multiplier;
    }
}

/* Using predicted positions instead of the real particle positions in the simulation helps at managing the chaos which the simulation
 * can become.
 */
__global__ void set_predicted_positions(v2 *positions, v2 *predicted_positions, v2 *velocities, int n_particles, v2 grid_size, settings s)
{
    int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);

    if(i >= n_particles)
        return;

    predicted_positions[i].x = positions[i].x + velocities[i].x*s.time_step;
    predicted_positions[i].y = positions[i].y + velocities[i].y*s.time_step;

    if(predicted_positions[i].x < 0)
    {
        predicted_positions[i].x = 0;
    }
    else if(predicted_positions[i].x >= grid_size.x)
    {
        predicted_positions[i].x = grid_size.x;
    }

    if(predicted_positions[i].y < 0)
    {
        predicted_positions[i].y = 0;
    }
    else if(predicted_positions[i].y >= grid_size.y)
    {
        predicted_positions[i].y = grid_size.y;
    }
}

__global__ void set_velocities(v2 *velocities, v2 *pressures, v2 *viscosities, int n_particles, settings s)
{
    int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);

    if(i >= n_particles)
        return;

    v2 total_force = {pressures[i].x + viscosities[i].x, pressures[i].y + viscosities[i].y};

    velocities[i].x += (total_force.x / s.mass)*s.time_step;
    velocities[i].y += (total_force.y / s.mass)*s.time_step;

    velocities[i].y += -9.81f*s.time_step;

}


__device__ inline v2 cell_coordinate(v2 pos, float h)
{
    v2 cell;

    cell.x = floorf(pos.x/h);
    cell.y = floorf(pos.y/h);

    return cell;
}

//Simplest hash function I could come up with, surely there is room for improvement
__device__ inline int hash_cell(int max_hash, v2 pos)
{
    unsigned int hashx = (int)pos.x * 11;
    unsigned int hashy = (int)pos.y * 4079;
    return ((hashx + hashy)*7) % max_hash;
}

__device__ float get_distance_2(v2 pos1, v2 pos2)
{
    return (pos1.x - pos2.x) * (pos1.x - pos2.x) + (pos1.y - pos2.y) * (pos1.y - pos2.y);
}

__global__ void update_cells(entry *entries, int *start_indices, v2 *predicted_positions, int n_particles, settings s)
{
    int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);

    if(i >= n_particles)
        return;

    v2 cell_pos = cell_coordinate(predicted_positions[i], s.smoothing_length);
    int cell_key = hash_cell(n_particles, cell_pos);

    entries[i] = {cell_key, cell_pos, i};
    start_indices[i] = -1;
}

__global__ void bitonic_sort(entry *entries, int n_entries, int j, int k)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;

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

__global__ void set_viscosities(v2 *velocities, v2 *predicted_positions, v2 *viscosities, int n_particles, settings s, v2 size, entry *entries, int *start_indices)
{
    int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);

    if(i >= n_particles)
        return;

    viscosities[i] = calculate_viscosity(velocities, predicted_positions, n_particles, i, s, size, entries, start_indices);
}

__device__ v2 calculate_viscosity(v2 *velocities, v2 *predicted_positions, int n_particles, int p_number, settings s, v2 size, entry *entries, int *start_indices)
{
    v2 viscosity = {0, 0};
    v2 center = cell_coordinate(predicted_positions[p_number], s.smoothing_length);

    float h_2 = s.smoothing_length * s.smoothing_length;

    for(int offsetX = -1; offsetX <= 1; offsetX++)
    {
        if((center.x + offsetX) < 0 || (center.x + offsetX) >= size.x)
            continue;

        for(int offsetY = -1; offsetY <= 1; offsetY++)
        {
            if((center.y + offsetY) < 0 || (center.y + offsetY) >= size.y)
                continue;

            v2 cell = {center.x + offsetX, center.y + offsetY};
            int cell_key = hash_cell(n_particles, cell);
            int cell_start = start_indices[cell_key];

            if(cell_start == -1)
                continue;

            for(int i = cell_start; i < n_particles; i++)
            {
                //Double-checking hash and location
                if(entries[i].cell_key != cell_key || (entries[i].location.x != center.x + offsetX || entries[i].location.y != center.y + offsetY))
                    break;

                int pi = entries[i].particle_index;

                float distance_2 = get_distance_2(predicted_positions[p_number], predicted_positions[pi]);

                if(distance_2 > h_2)
                    continue;

                float influence = viscosity_laplacian_kernel(sqrt(distance_2), s.smoothing_length);

                viscosity.x += (velocities[pi].x - velocities[p_number].x) * influence;
                viscosity.y += (velocities[pi].y - velocities[p_number].y) * influence;
            }
        }
    }

    viscosity.x *= s.viscosity;
    viscosity.y *= s.viscosity;

    return viscosity;
}