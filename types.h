#ifndef FLUIDSIM_TYPES_H
#define FLUIDSIM_TYPES_H

typedef struct v2{
    float x;
    float y;
}v2;

typedef struct particles{
    v2 *pos;
    v2 *gpu_pos, *gpu_p_pos, *gpu_vel, *gpu_pressure, *gpu_viscosity;
    float *gpu_density, *gpu_near_density;
}particles;

typedef struct settings{
    float smoothing_length;
    float mass;
    float viscosity;
    float time_step;
    float rest_density;
    float pressure_multiplier;
    float bounce_multiplier;
    float near_pressure_multiplier;
}settings;

typedef struct entry{
    int cell_key;
    v2 location;
    int particle_index;
}entry;

typedef struct cells{
    v2 size;
    entry *entries;
    int *start_indices;
}cells;

#endif //FLUIDSIM_TYPES_H
