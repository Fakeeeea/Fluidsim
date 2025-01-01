#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cell_ll.h"
#include "sim_math.h"


/* The main idea behind SPH is about particles having different influence on each-others density
 * based on how far away they are from each-other. These functions' return value is the influence
 * of a particle onto another particle.
 */


float poly6_smoothing_kernel(float r_2, float smoothing_length_2)
{
    static float scaling = 0;

    if(scaling == 0)
    {
        scaling = 4.0f/(M_PI*pow(smoothing_length_2, 4));
    }

    if(r_2 >= smoothing_length_2)
        return 0;

    float delta = smoothing_length_2 - r_2;
    return scaling*delta*delta*delta;
}

//Gradient of the spiky smoothing kernel, from the spiky smoothing kernel, used for pressure calculation
float spiky_smoothing_kernel_gradient(float r, float smoothing_length)
{
    static float scaling = 0;

    if(scaling == 0)
    {
        scaling = -10.0f/(M_PI*pow(smoothing_length, 5));
    }

    if(r >= smoothing_length)
        return 0;

    float delta = smoothing_length - r;
    return scaling*delta*delta*delta;
}

//Laplacian of the viscosity smoothing kernel, used for viscosity calculation
float viscosity_laplacian_smoothing_kernel(float r, float smoothing_length)
{
    static float scaling = 0;

    if(scaling == 0)
    {
        scaling = 40.0f/(M_PI*pow(smoothing_length, 5));
    }

    if(r >= smoothing_length)
        return 0;

    return scaling*(smoothing_length - r);
}

float calculate_density(particle *particles, int p_number, int n_particles, settings s, cells *cell_ll, v2 pos)
{
    float density = 0;
    float h_2 = s.smoothing_length * s.smoothing_length;

    int n_neighbours;
    int *neighbours = NULL;

    if(p_number == USECUSTOMPOS)
        neighbours = get_neighbours(particles, USECUSTOMPOS, n_particles, s, cell_ll, &n_neighbours, pos);
    else
        neighbours = get_neighbours(particles, p_number, n_particles, s, cell_ll, &n_neighbours, (v2) {0,0});

    for(int i = 0; i < n_neighbours; i++)
    {
        int pi = neighbours[i];
        v2 pos1;
        if(p_number == USECUSTOMPOS)
            pos1 = pos;
        else
            pos1 = particles[p_number].predicted_pos;

        float distance_2 = get_distance_2(pos1, particles[pi].predicted_pos);

        density += s.mass * poly6_smoothing_kernel(distance_2, h_2);
    }

    free(neighbours);

    return density;
}


v2 calculate_pressure(particle *particles, int p_number, int n_particles, settings s, cells *cell_ll)
{
    v2 resulting_forces;
    v2 pressure = (v2){0,0};
    v2 viscosity = (v2){0,0};
    v2 dir;

    int n_neighbours;
    int *neighbours = get_neighbours(particles, p_number, n_particles, s, cell_ll, &n_neighbours, (v2) {0,0});
    float h_2 = s.smoothing_length * s.smoothing_length;

    for(int i = 0; i < n_neighbours; i++)
    {
        int pi = neighbours[i];

        if(pi == p_number)
            continue;

        float distance_2 = get_distance_2(particles[p_number].predicted_pos, particles[pi].predicted_pos);

        if(distance_2 >= h_2)
            continue;

        float distance = sqrt(distance_2);

        if(distance > 1e-10)
        {
            dir.x = (particles[pi].predicted_pos.x - particles[p_number].predicted_pos.x)/distance;
            dir.y = (particles[pi].predicted_pos.y - particles[p_number].predicted_pos.y)/distance;
        }
        else
            continue;

        if(particles[pi].density > 1e-10)
        {
            float sp = calculate_symmetric_pressure(particles[p_number], particles[pi], s);
            float slope = spiky_smoothing_kernel_gradient(distance, s.smoothing_length);


            pressure.x += -s.mass * sp * slope * dir.x;
            pressure.y += -s.mass * sp * slope * dir.y;

            viscosity.x += s.mass * s.viscosity * (particles[pi].vel.x - particles[p_number].vel.x) * viscosity_laplacian_smoothing_kernel(distance, s.smoothing_length);
            viscosity.y += s.mass * s.viscosity * (particles[pi].vel.y - particles[p_number].vel.y) * viscosity_laplacian_smoothing_kernel(distance, s.smoothing_length);
        }
    }

    free(neighbours);

    resulting_forces = (v2) {pressure.x + viscosity.x, pressure.y + viscosity.y};

    return resulting_forces;
}

inline float density_to_pressure(float density, settings s)
{
    float pressure = s.pressure_multiplier*(density - s.rest_density);

    //negative pressure (particles attracting eachother)
    //if(pressure < 0.0f)
      //  return 0;

    return pressure;
}

inline float calculate_symmetric_pressure(particle p1, particle p2, settings s)
{
    float pressure1, pressure2;

    pressure1 = density_to_pressure(p1.density, s);
    pressure2 = density_to_pressure(p2.density, s);

    return (pressure1+pressure2)/(2*p2.density);
}

void set_densities(particle *particles, int n_particles, settings s, cells *cell_ll)
{
    for(int i = 0; i<n_particles; i++)
    {
        particles[i].density = calculate_density(particles, i, n_particles, s, cell_ll, (v2) {0,0});
    }
}

void set_positions(particle *particles, int n_particles, v2 grid_size, settings s)
{
    for(int i = 0; i<n_particles; i++)
    {
        particles[i].pos.x += particles[i].vel.x*s.time_step;
        particles[i].pos.y += particles[i].vel.y*s.time_step;

        if(particles[i].pos.x < 0)
        {
            particles[i].pos.x = 0;

            if(particles[i].vel.x < 1)
                particles[i].vel.x = 2;
            else
                particles[i].vel.x = -particles[i].vel.x * s.bounce_multiplier;
        }
        else if(particles[i].pos.x >= grid_size.x)
        {
            particles[i].pos.x = grid_size.x;

            if(particles[i].vel.x < 1)
                particles[i].vel.x = -2;
            else
                particles[i].vel.x = -particles[i].vel.x * s.bounce_multiplier;
        }

        if(particles[i].pos.y < 0)
        {
            particles[i].pos.y = 0;
            particles[i].vel.y = -particles[i].vel.y * s.bounce_multiplier;
        }
        else if(particles[i].pos.y >= grid_size.y)
        {
            particles[i].pos.y = grid_size.y;
            particles[i].vel.y = -particles[i].vel.y * s.bounce_multiplier;
        }
    }
}

void set_velocities(particle *particles, int n_particles, settings s)
{
    for(int i = 0; i<n_particles; i++)
    {
        if(particles[i].density != 0.0f)
        {
            particles[i].vel.x += -(particles[i].pressure.x / s.mass)*s.time_step;
            particles[i].vel.y += -(particles[i].pressure.y / s.mass)*s.time_step;

            particles[i].vel.x += (particles[i].viscosity.x / s.mass)*s.time_step;
            particles[i].vel.y += (particles[i].viscosity.y / s.mass)*s.time_step;
        }
        particles[i].vel.y += -9.81f*s.time_step;
    }
}

void set_pressures(particle *particles, int n_particles, settings s, cells *cell_ll)
{
    for(int i = 0; i<n_particles; i++)
    {
        particles[i].pressure = calculate_pressure(particles, i, n_particles, s, cell_ll);
    }
}

void set_predicted_positions(particle *particles, int n_particles, v2 grid_size, settings s)
{
    for(int i = 0; i<n_particles; i++)
    {
        particles[i].predicted_pos.x = particles[i].pos.x + particles[i].vel.x*s.time_step;
        particles[i].predicted_pos.y = particles[i].pos.y + particles[i].vel.y*s.time_step;

        if(particles[i].predicted_pos.x < 0)
        {
            particles[i].predicted_pos.x = 0;
        }
        else if(particles[i].pos.x >= grid_size.x)
        {
            particles[i].predicted_pos.x = grid_size.x;
        }

        if(particles[i].predicted_pos.y < 0)
        {
            particles[i].predicted_pos.y = 0;
        }
        else if(particles[i].predicted_pos.y >= grid_size.y)
        {
            particles[i].predicted_pos.y = grid_size.y;
        }
    }
}

int* get_neighbours(particle *particles, int p_number, int n_particles, settings s, cells *cell_ll, int *n_neighbours, v2 pos)
{
    v2 center;

    if(p_number == USECUSTOMPOS)
        center = cell_coordinate(pos, s.smoothing_length);
    else
        center = cell_coordinate(particles[p_number].predicted_pos, s.smoothing_length);

    //to avoid calling malloc multiple times with the change of the number of neighbours, we allocate the maximum possible number of neighbours
    int *neighbours = (int*) malloc(sizeof(int) * n_particles);
    *n_neighbours = 0;

    for(int offsetX = -1; offsetX <= 1; offsetX++)
    {
        if(center.x + offsetX < 0 || center.x + offsetX >= cell_ll->size.x)
            continue;

        for(int offsetY = -1; offsetY <= 1; offsetY++)
        {
            if(center.y + offsetY < 0 || center.y + offsetY >= cell_ll->size.y)
                continue;

            int cell_key = hash_cell(n_particles, (v2) {center.x + offsetX, center.y + offsetY});
            int cell_start = cell_ll->start_indices[cell_key];

            if(cell_start == -1)
                continue;

            for(int i = cell_start; i < n_particles ; i++)
            {
                //check if hash of particle is same as computed hash
                if(cell_ll->entries[i].cell_key != cell_key)
                    break;

                //double check if the location is the same as the one we are looking for
                if(cell_ll->entries[i].location.x != center.x + offsetX || cell_ll->entries[i].location.y != center.y + offsetY)
                    break;

                int pi = cell_ll->entries[i].particle_index;

                v2 pos1;

                if(p_number == USECUSTOMPOS)
                    pos1 = pos;
                else
                    pos1 = particles[p_number].predicted_pos;


                if(get_distance_2(pos1, particles[pi].predicted_pos) < s.smoothing_length*s.smoothing_length)
                {
                    neighbours[*n_neighbours] = pi;
                    (*n_neighbours)++;
                }
            }

        }
    }

    return neighbours;
}

float get_point_density(particle *particles, v2 pos, int n_particles, settings s, cells *cell_ll)
{
    return calculate_density(particles, USECUSTOMPOS, n_particles, s, cell_ll, pos);
}

inline float get_distance_2(v2 p1, v2 p2)
{
    return (p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y);
}