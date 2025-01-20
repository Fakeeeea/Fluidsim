#ifndef FLUIDSIM_SIM_MATH_CUH
#define FLUIDSIM_SIM_MATH_CUH

#include <cuda_runtime.h>
#include <windows.h>

#include "types.h"

#define PIXELTOUNIT 0.1f

extern __constant__ float poly6_scaling;
extern __constant__ float spiky_scaling;
extern __constant__ float viscosity_scaling;

#ifdef __cplusplus
extern "C" {
#endif

void malloc_simulation_gpu(int n_particles, particles *sim);
void simulation_step_gpu(int_v2 container, particles *sim, settings s, cells *cell_ll);
void create_cell_ll_gpu(cells *cell_ll, RECT rect, settings s);
void initialize_constants(float poly6, float spiky, float viscosity);
void sort_entries(entry *entries, int n_particles);
void free_simulation_memory(particles *sim);

#ifdef __cplusplus
}
#endif

__device__ float poly6_smoothing_kernel(float r_2, float h_2);
__device__ float spiky_smoothing_kernel(float r, float h);
__device__ float viscosity_laplacian_kernel(float r, float h);

__device__ float get_distance_2(float_v2 pos1, float_v2 pos2);

__device__ int hash_cell(int max_hash, int_v2 pos);
__device__ int_v2 cell_coordinate(float_v2 pos, float h);

__device__ float_v2 calculate_density(float_v2 *predicted_positions, float_v2 particle_pos, settings s, entry *entries, int *start_indices, int_v2 size);
__global__ void set_densities(float_v2 *predicted_positions, float *densities, float *near_densities, settings s, entry *entries, int *start_indices, int_v2 size);

__device__ float density_to_pressure(float density, settings s);
__device__ float calculate_symmetric_pressure(float density1, float density2, settings s);
__device__ float near_density_to_pressure(float near_density, settings s);
__device__ float calculate_symmetric_near_pressure(float near_density1, float near_density2, settings s);

__device__ float_v2 calculate_pressure(float_v2 *predicted_positions, float *densities, float *near_densities, int p_number, settings s, entry *entries, int *start_indices, int_v2 size);
__global__ void set_pressures(float_v2 *predicted_positions, float *densities, float *near_densities, float_v2 *pressures, settings s, entry *entries, int *start_indices, int_v2 size);

__global__ void set_velocities(float_v2 *velocities, float_v2 *pressures, float_v2 *viscosities, settings s);
__global__ void set_predicted_positions(float_v2 *positions, float_v2 *predicted_positions, float_v2 *velocities, int_v2 container, settings s);
__global__ void set_positions(float_v2 *positions, float_v2 *velocities, int_v2 container, settings s);

__global__ void update_cells(entry *entries, int *start_indices, float_v2 *predicted_positions, settings s);
__global__ void bitonic_sort(entry *entries, int n_entries, int j, int k);
__global__ void set_start_indices(entry *entries, int *start_indices, int n_particles);

__global__ void set_viscosities(float_v2 *velocities, float_v2 *predicted_positions, float_v2 *viscosities, settings s, int_v2 size, entry *entries, int *start_indices);
__device__ float_v2 calculate_viscosity(float_v2 *velocities, float_v2 *predicted_positions, int p_number, settings s, int_v2 size, entry *entries, int *start_indices);

#endif //FLUIDSIM_SIM_MATH_CUH
