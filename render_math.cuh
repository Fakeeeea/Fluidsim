#ifndef FLUIDSIM_RENDER_MATH_CUH
#define FLUIDSIM_RENDER_MATH_CUH

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

void allocate_render_memory(int** gpu_bitmap, v2 size);
void realloc_render_memory(int** gpu_bitmap, v2 size);
void free_render_memory(int* gpu_bitmap);
int* get_colored_bitmap(v2 *positions, v2 size, settings s, entry *entries, int *start_indices, int n_particles, int *gpu_bitmap);

#ifdef __cplusplus
}
#endif

__global__ void set_bitmap_colors(int *gpu_bitmap, v2 *positions, v2 size, settings s, entry *entries, int *start_indices, int n_particles);

#endif //FLUIDSIM_RENDER_MATH_CUH
