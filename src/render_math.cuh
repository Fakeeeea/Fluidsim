#ifndef FLUIDSIM_RENDER_MATH_CUH
#define FLUIDSIM_RENDER_MATH_CUH

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

void allocate_render_memory(int** gpu_bitmap, int_v2 size);
void realloc_render_memory(int** gpu_bitmap, int_v2 size);
void free_render_memory(int* gpu_bitmap);
__host__ int* get_colored_bitmap(float_v2 *positions, int_v2 size, settings s, entry *entries, int *start_indices, int *gpu_bitmap, obstacles obs);

#ifdef __cplusplus
}
#endif

__global__ void set_bitmap_colors(int *gpu_bitmap, float_v2 *positions, int_v2 size, settings s, entry *entries, int *start_indices);
__global__ void draw_boundary(int *gpu_bitmap, int_v2 size, int_v2 min, int_v2 max);

#endif //FLUIDSIM_RENDER_MATH_CUH
