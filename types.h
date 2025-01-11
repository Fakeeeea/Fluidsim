#ifndef FLUIDSIM_TYPES_H
#define FLUIDSIM_TYPES_H

typedef struct float_v2{
    float x;
    float y;
}float_v2;

typedef struct int_v2 {
    int x;
    int y;
}int_v2;

typedef struct particles{
    float_v2 *pos;
    float_v2 *gpu_pos, *gpu_p_pos, *gpu_vel, *gpu_pressure, *gpu_viscosity;
    float *gpu_density, *gpu_near_density;
}particles;

typedef struct render_settings{
    float red_absorption;
    float blue_absorption;
    float green_absorption;
}render_settings;

typedef struct sim_settings{
    float smoothing_length;
    float mass;
    float viscosity;
    float time_step;
    float rest_density;
    float pressure_multiplier;
    float bounce_multiplier;
    float near_pressure_multiplier;
    int n_particles;
}sim_settings;

typedef struct settings{
    sim_settings ss;
    render_settings rs;
}settings;

typedef struct entry{
    int cell_key;
    int particle_index;
}entry;

typedef struct cells{
    int_v2 world_size;
    entry *entries;
    int *start_indices;
}cells;

typedef struct boundary{
    int_v2 min;
    int_v2 max;
}boundary;

#endif //FLUIDSIM_TYPES_H
