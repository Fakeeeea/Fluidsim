#include <stdio.h>
#include <stdlib.h>

#include "sim_parser.h"
#include "types.h"

void parse_settings(settings *s)
{
    FILE *fptr;

    if((fopen_s(&fptr, "../settings.txt", "r")) != 0)
    {
        printf("Could not open settings file");
        exit(1);
    }

    char buf[256];

    while(fgets(buf, sizeof(buf), fptr))
    {
        if(buf[0] == '#' || buf[0] == '\n')
            continue;

        //Sim settings
        if(sscanf_s(buf, "mass: %f", &s->ss.mass)) continue;
        if(sscanf_s(buf, "rest_density: %f", &s->ss.rest_density)) continue;
        if(sscanf_s(buf, "viscosity: %f", &s->ss.viscosity)) continue;
        if(sscanf_s(buf, "smoothing_length: %f", &s->ss.smoothing_length)) continue;
        if(sscanf_s(buf, "time_step: %f", &s->ss.time_step)) continue;
        if((sscanf_s(buf, "prediction_time_step: %f", &s->ss.prediction_time_step))) continue;
        if(sscanf_s(buf, "pressure_multiplier: %f", &s->ss.pressure_multiplier)) continue;
        if(sscanf_s(buf, "bounce_multiplier: %f", &s->ss.bounce_multiplier)) continue;
        if(sscanf_s(buf, "near_pressure_multiplier: %f", &s->ss.near_pressure_multiplier)) continue;
        if(sscanf_s(buf, "n_particles: %d", &s->ss.n_particles)) continue;
        if(sscanf_s(buf, "spawn_random: %d", &s->ss.spawn_random)) continue;
        if(sscanf_s(buf, "gravity: %f", &s->ss.gravity)) continue;

        //Render settings
        if(sscanf_s(buf, "red_absorption: %f", &s->rs.red_absorption)) continue;
        if(sscanf_s(buf, "green_absorption: %f", &s->rs.green_absorption)) continue;
        if(sscanf_s(buf, "blue_absorption: %f", &s->rs.blue_absorption)) continue;
    }

    fclose(fptr);
}

int_b* parse_spawn(int_b max_size, int *n_spawn)
{
    FILE *fptr;

    if((fopen_s(&fptr, "../spawn.txt", "r")) != 0)
    {
        printf("Could not open spawn file");
        exit(1);
    }

    char buf[256];
    int_b *boundaries = NULL;
    int i;
    int is_allocated = 0;

    while(fgets(buf, sizeof(buf), fptr))
    {
        if(buf[0] == '#' || buf[0] == '\n')
            continue;

        if(sscanf_s(buf, "num_boundaries: %d", n_spawn))
        {
            boundaries = (int_b*) malloc(sizeof(int_b) * (*n_spawn));
            is_allocated = 1;
            continue;
        }

        if(is_allocated)
        {
            float_v2 min_percentage, max_percentage;
            if(sscanf_s(buf, "boundary %d: %f %f %f %f", &i, &min_percentage.x, &min_percentage.y, &max_percentage.x, &max_percentage.y))
            {
                if(i > *n_spawn)
                {
                    printf("Number of boundaries in file does not match num_boundaries\n");
                    exit(1);
                }

                boundaries[i].min.x = (int) (( min_percentage.x * (float) max_size.max.x) / 100.0f);
                boundaries[i].min.y = (int) (( min_percentage.y * (float) max_size.max.y) / 100.0f);
                boundaries[i].max.x = (int) (( max_percentage.x * (float) max_size.max.x) / 100.0f);
                boundaries[i].max.y = (int) (( max_percentage.y * (float) max_size.max.y) / 100.0f);
            }
        }
    }

    return boundaries;
}

float_b* parse_obstacles(int_b max_size, int *n_obstacles)
{
    FILE *fptr;

    if((fopen_s(&fptr, "../obstacles.txt", "r")) != 0)
    {
        printf("Could not open obstacles file");
        exit(1);
    }

    char buf[256];
    float_b *obstacles = NULL;
    int i;
    int is_allocated = 0;

    while(fgets(buf, sizeof(buf), fptr))
    {
        if(buf[0] == '#' || buf[0] == '\n')
            continue;

        if(sscanf_s(buf, "num_obstacles: %d", n_obstacles))
        {
            obstacles= (float_b*) malloc(sizeof(float_b) * (*n_obstacles));
            is_allocated = 1;
            continue;
        }

        if(is_allocated)
        {
            float_v2 min_percentage, max_percentage;
            if(sscanf_s(buf, "obstacle %d: %f %f %f %f", &i, &min_percentage.x, &min_percentage.y, &max_percentage.x, &max_percentage.y))
            {
                if(i > *n_obstacles)
                {
                    printf("Number of obstacles in file does not match num_obstacles\n");
                    exit(1);
                }

                obstacles[i].min.x = ((min_percentage.x * (float) max_size.max.x) / 100.0f);
                obstacles[i].min.y = ((min_percentage.y * (float) max_size.max.y) / 100.0f);
                obstacles[i].max.x = ((max_percentage.x * (float) max_size.max.x) / 100.0f);
                obstacles[i].max.y = ((max_percentage.y * (float) max_size.max.y) / 100.0f);
            }
        }
    }

    return obstacles;


}