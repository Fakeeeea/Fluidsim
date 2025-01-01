#ifndef FLUIDSIM_SIM_MATH_H
#define FLUIDSIM_SIM_MATH_H

#include "types.h"
#include "cell_ll.h"

#define USECUSTOMPOS -1

float poly6_smoothing_kernel(float r, float smoothing_length);
float spiky_smoothing_kernel_gradient(float r, float smoothing_length);

v2 calculate_pressure(particle *particles, int p_number, int n_particles, settings s, cells *cell_ll);
float calculate_symmetric_pressure(particle p1, particle p2, settings s);
float calculate_density(particle *particles, int p_number, int n_particles, settings s, cells *cell_ll, v2 pos);

void set_densities(particle *particles, int n_particles, settings s, cells *cell_ll);
void set_positions(particle *particles, int n_particles, v2 grid_size, settings s);
void set_velocities(particle *particles, int n_particles, settings s);
void set_pressures(particle *particles, int n_particles, settings s, cells *cell_ll);
void set_predicted_positions(particle *particles, int n_particles, v2 grid_size, settings s);
int* get_neighbours(particle *particles, int p_number, int n_particles, settings s, cells *cell_ll, int *n_neighbours, v2 pos);

float get_point_density(particle *particles, v2 pos, int n_particles, settings s, cells *cell_ll);

float density_to_pressure(float density, settings s);
float get_distance_2(v2 p1, v2 p2);

#endif //FLUIDSIM_SIM_MATH_H
