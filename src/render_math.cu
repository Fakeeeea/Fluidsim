#include <cuda_runtime.h>

#include "render_math.cuh"
#include "sim_math.cuh"

#define RGB_TO_INT(r, g, b) ((r << 16) | (g << 8) | b)

#ifdef __cplusplus
extern "C" {
#endif

__host__ void allocate_render_memory(int** gpu_bitmap, int_v2 size)
{
    cudaHostAlloc(gpu_bitmap, size.x * size.y * sizeof(int), cudaHostAllocDefault);
}

__host__ void realloc_render_memory(int** gpu_bitmap, int_v2 size)
{
    cudaFreeHost(*gpu_bitmap);
    cudaHostAlloc(gpu_bitmap, size.x * size.y * sizeof(int), cudaHostAllocDefault);
}

__host__ void free_render_memory(int* gpu_bitmap)
{
    cudaFree(gpu_bitmap);
}

__host__ int* get_colored_bitmap(float_v2 *positions, int_v2 size, settings s, entry *entries, int *start_indices, int *gpu_bitmap, obstacles obs)
{
    static const dim3 block_size(16,16);
    static const dim3 num_blocks((size.x + block_size.x - 1) / block_size.x,
                           (size.y + block_size.y - 1) / block_size.y);

    set_bitmap_colors<<<num_blocks, block_size>>>(gpu_bitmap, positions, size, s, entries, start_indices);
    cudaDeviceSynchronize();

    for(int i = 0; i < obs.n_obstacles; i++)
    {
        const int min_x = (int) (obs.obstacles[i].min.x * UNITTOPIXEL) < 0 ? 0 : (int) (obs.obstacles[i].min.x * UNITTOPIXEL);
        const int min_y = (int) (obs.obstacles[i].min.y * UNITTOPIXEL) < 0 ? 0 : (int) (obs.obstacles[i].min.y * UNITTOPIXEL);
        const int max_x = (int) (obs.obstacles[i].max.x * UNITTOPIXEL) > size.x - 1 ? size.x - 1 : (int) (obs.obstacles[i].max.x * UNITTOPIXEL);
        const int max_y = (int) (obs.obstacles[i].max.y * UNITTOPIXEL) > size.y - 1 ? size.y - 1 : (int) (obs.obstacles[i].max.y * UNITTOPIXEL);

        const int width = max_x - min_x;
        const int height = max_y - min_y;

        if(width <= 0 || height <= 0)
            continue;

        const dim3 num_blocks_boundaries((width + block_size.x - 1) / block_size.x,
                                        (height + block_size.y - 1) / block_size.y);

        draw_boundary<<<block_size, num_blocks_boundaries>>>(gpu_bitmap, size, {min_x, min_y}, {max_x, max_y});
    }

    int *bitmap = (int*)malloc(size.x * size.y * sizeof(int));
    cudaMemcpy(bitmap, gpu_bitmap, size.x * size.y * sizeof(int), cudaMemcpyDeviceToHost);

    return bitmap;
}

#ifdef __cplusplus
}
#endif

__global__ void set_bitmap_colors(int *gpu_bitmap, float_v2 *positions, int_v2 size, settings s, entry *entries, int *start_indices)
{
    int x = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int) (blockIdx.y * blockDim.y + threadIdx.y);

    if(x >= size.x || y >= size.y)
        return;

    const float_v2 pos = {(float) x * PIXELTOUNIT, (float) y * PIXELTOUNIT};

    //Absorption factors
    const float a_red = s.rs.red_absorption;
    const float a_green = s.rs.green_absorption;
    const float a_blue = s.rs.blue_absorption;

    const float h_2 = s.ss.smoothing_length * s.ss.smoothing_length;

    float_v2 calculated_density = calculate_density(positions, pos, s, entries, start_indices, {(int) ceilf( (float) size.x * 0.1f), (int) ceilf( (float) size.y * 0.1f) });

    /* Might want to do this with the normalized density. Didn't bring up any issues as far as I have seen,
     * so I left it like this
     */

    if(calculated_density.x < poly6_smoothing_kernel(h_2 * 0.1f, h_2))
    {
        gpu_bitmap[(int)(y * size.x + x)] = RGB_TO_INT(255,255,255);
        return;
    }

    /* Theoretical maximum density calculation (Really theoretical, extremely theoretical, but gives good results)
     * With this approximation, particles would have an ideal center of maximum density (distance 0), and around another "circle" of 6 particles,
     * with distance h/2 (^2 = h^2/4). Lastly we would have a final "circle" of 12 particles with distance h/sqrt(2) (^2 = h^2/2)
     * (Imagining the particles evenly distributed in the 2d space, 60 degrees apart from each other)
     */
    const float max_theoretical_density = poly6_smoothing_kernel(0.0f, h_2) +
                                          6 * poly6_smoothing_kernel(h_2 * 0.25f, h_2) +
                                          12 * poly6_smoothing_kernel(h_2 * 0.5f, h_2);

    /* Normalize the density relative to maximum theoretical density (Again, really theoretical maximum density)
     * this sadly, makes it way less "fluid-like". But I guess it's a good tradeoff for it working with most smoothing radii (or at least the ones I tested)
     */

    const float normalized_density = calculated_density.x / max_theoretical_density;

    //Damping normalized_density to spread the colors more
    const float final_light = normalized_density * 0.2f;

    //The values are clamped between 15*absorption factor and 255
    const int red = max((int)(a_red * 15), min(255, (int)(255 * final_light * a_red)));
    const int green = max((int)(a_green * 15), min(255, (int)(255 * final_light * a_green)));
    const int blue = max((int)(a_blue * 15), min(255, (int)(255 * final_light * a_blue)));

    //Color the pixel in the bitmap
    gpu_bitmap[(int)(y * size.x + x)] = RGB_TO_INT((255-red),(255-green),(255-blue));
}

__global__ void draw_boundary(int *gpu_bitmap, int_v2 size, int_v2 min, int_v2 max)
{
    const int x = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int) (blockIdx.y * blockDim.y + threadIdx.y);

    const int world_x = x + min.x;
    const int world_y = y + min.y;

    if(world_x >= size.x || world_y >= size.y || world_x < 0 || world_y < 0)
        return;

    if(world_x > max.x || world_y > max.y)
        return;

    gpu_bitmap[(int)(world_y * size.x + world_x)] = RGB_TO_INT(173, 173, 173);
}
