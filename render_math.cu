#include <cuda_runtime.h>

#include "render_math.cuh"
#include "sim_math.cuh"

#define RGB_TO_INT(r, g, b) ((r << 16) | (g << 8) | b)

#ifdef __cplusplus
extern "C" {
#endif

__host__ void allocate_render_memory(int** gpu_bitmap, v2 size)
{
    cudaMalloc(gpu_bitmap, size.x * size.y * sizeof(int));
}

__host__ void realloc_render_memory(int** gpu_bitmap, v2 size)
{
    cudaFree(*gpu_bitmap);
    cudaMalloc(gpu_bitmap, size.x * size.y * sizeof(int));
}

__host__ void free_render_memory(int* gpu_bitmap)
{
    cudaFree(gpu_bitmap);
}

__host__ int* get_colored_bitmap(v2 *positions, v2 size, settings s, entry *entries, int *start_indices, int n_particles, int *gpu_bitmap)
{
    cudaMemset(gpu_bitmap, RGB_TO_INT(255,255,255), size.x * size.y * sizeof(int));

    int block_size = ceil(size.x / 10);
    int num_blocks = 10;

    set_bitmap_colors<<<num_blocks, block_size>>>(gpu_bitmap, positions, size, s, entries, start_indices, n_particles);
    cudaDeviceSynchronize();

    int *bitmap = (int*)malloc(size.x * size.y * sizeof(int));
    cudaMemcpy(bitmap, gpu_bitmap, size.x * size.y * sizeof(int), cudaMemcpyDeviceToHost);

    return bitmap;
}

#ifdef __cplusplus
}
#endif

__global__ void set_bitmap_colors(int *gpu_bitmap, v2 *positions, v2 size, settings s, entry *entries, int *start_indices, int n_particles)
{
    /* Main idea: Light comes from above, reaches water. The more fluid the light travels through, less of it is left. Denser areas don't
     * affect light more than less dense areas, for now.
     * At any point, before reaching the viewer, some of the light gets absorbed. The absorption constants decide how much
     * of each wavelength is actually absorbed.
     */

    int x = blockIdx.x * blockDim.x + threadIdx.x;

    v2 ray_start;
    ray_start.x = x/10.0f;

    //Absorption factors
    const float a_red = 3;
    const float a_green = 1.4;
    const float a_blue = 1;

    float ray_light = 1;
    float attenuation_factor = 0.0001;

    for(int i = 0; i < size.y; i++)
    {
        ray_start.y = i/10.0f;
        v2 calculated_density = calculate_density(positions, ray_start, n_particles, s, entries, start_indices, size);

        if(calculated_density.x < 0.04)
        {
            gpu_bitmap[(int)(i * size.x + x)] = RGB_TO_INT(255,255,255);
            continue;
        }
        ray_light -= attenuation_factor;

        float final_light = ray_light * (calculated_density.x/3);

        //The values are clamped between 15*absorption factor and 255
        int red = max((int)(a_red * 15), min(255, (int)(255 * final_light * a_red)));
        int green = max((int)(a_green * 15), min(255, (int)(255 * final_light * a_green)));
        int blue = max((int)(a_blue * 15), min(255, (int)(255 * final_light * a_blue)));

        //Color the pixel in the bitmap
        gpu_bitmap[(int)(i * size.x + x)] = RGB_TO_INT(255-red,255-green,255-blue);

    }
}
