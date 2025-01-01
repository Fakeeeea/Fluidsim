#ifndef FLUIDSIM_SIM_MATH_CUH
#define FLUIDSIM_SIM_MATH_CUH

#include <cuda_runtime.h>
#include <windows.h>

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

void malloc_simulation_gpu(int n_particles, particles *sim);
void simulation_step_gpu(v2 container, particles *sim, int N_PARTICLES, settings s, cells *cell_ll);
void create_cell_ll_gpu(cells *cell_ll, int n_particles, RECT rect, settings s);
void initialize_constants(float poly6, float spiky, float viscosity);
void sort_entries(entry *entries, int n_particles);
void free_simulation_memory(particles *sim);

#ifdef __cplusplus
}
#endif

__device__ float poly6_smoothing_kernel(float r_2, float h_2);
__device__ float spiky_smoothing_kernel(float r, float h);
__device__ float viscosity_laplacian_kernel(float r, float h);

__device__ float get_distance_2(v2 pos1, v2 pos2);

__device__ int hash_cell(int max_hash, v2 pos);
__device__ v2 cell_coordinate(v2 pos, float h);

__device__ v2 calculate_density(v2 *predicted_positions, v2 particle_pos, int n_particles, settings s, entry *entries, int *start_indices, v2 size);
__global__ void set_densities(v2 *predicted_positions, float *densities, float *near_densities, int n_particles, settings s, entry *entries, int *start_indices, v2 size);

__device__ float density_to_pressure(float density, settings s);
__device__ float calculate_symmetric_pressure(float density1, float density2, settings s);
__device__ float near_density_to_pressure(float near_density, settings s);
__device__ float calculate_symmetric_near_pressure(float near_density1, float near_density2, settings s);

__device__ v2 calculate_pressure(v2 *predicted_positions, float *densities, float *near_densities, int p_number, int n_particles, settings s, entry *entries, int *start_indices, v2 size);
__global__ void set_pressures(v2 *predicted_positions, float *densities, float *near_densities, v2 *pressures, int n_particles, settings s, entry *entries, int *start_indices, v2 size);

__global__ void set_velocities(v2 *velocities, v2 *pressures, v2 *viscosities, int n_particles, settings s);
__global__ void set_predicted_positions(v2 *positions, v2 *predicted_positions, v2 *velocities, int n_particles, v2 grid_size, settings s);
__global__ void set_positions(v2 *positions, v2 *velocities, v2 grid_size, int n_particles, settings s);

__global__ void update_cells(entry *entries, int *start_indices, v2 *predicted_positions, int n_particles, settings s);
__global__ void bitonic_sort(entry *entries, int n_entries, int j, int k);
__global__ void set_start_indices(entry *entries, int *start_indices, int n_particles);

__global__ void set_viscosities(v2 *velocities, v2 *predicted_positions, v2 *viscosities, int n_particles, settings s, v2 size, entry *entries, int *start_indices);
__device__ v2 calculate_viscosity(v2 *velocities, v2 *predicted_positions, int n_particles, int p_number, settings s, v2 size, entry *entries, int *start_indices);

#endif //FLUIDSIM_SIM_MATH_CUH
