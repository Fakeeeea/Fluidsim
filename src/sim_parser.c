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
        if(sscanf_s(buf, "pressure_multiplier: %f", &s->ss.pressure_multiplier)) continue;
        if(sscanf_s(buf, "bounce_multiplier: %f", &s->ss.bounce_multiplier)) continue;
        if(sscanf_s(buf, "near_pressure_multiplier: %f", &s->ss.near_pressure_multiplier)) continue;
        if(sscanf_s(buf, "n_particles: %d", &s->ss.n_particles)) continue;

        //Render settings
        if(sscanf_s(buf, "red_absorption: %f", &s->rs.red_absorption)) continue;
        if(sscanf_s(buf, "green_absorption: %f", &s->rs.green_absorption)) continue;
        if(sscanf_s(buf, "blue_absorption: %f", &s->rs.blue_absorption)) continue;
    }

    fclose(fptr);
}

boundary* parse_spawn(boundary screen, int *n_spawn)
{
    FILE *fptr;

    if((fopen_s(&fptr, "../spawn.txt", "r")) != 0)
    {
        printf("Could not open spawn file");
        exit(1);
    }

    char buf[256];
    boundary *boundaries = NULL;
    int i;
    int is_allocated = 0;

    while(fgets(buf, sizeof(buf), fptr))
    {
        if(buf[0] == '#' || buf[0] == '\n')
            continue;

        if(sscanf_s(buf, "num_boundaries: %d", n_spawn))
        {
            boundaries = (boundary*) malloc(sizeof(boundary) * (*n_spawn));
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

                boundaries[i].min.x = (min_percentage.x * screen.max.x)/100;
                boundaries[i].min.y = (min_percentage.y * screen.max.y)/100;
                boundaries[i].max.x = (max_percentage.x * screen.max.x)/100;
                boundaries[i].max.y = (max_percentage.y * screen.max.y)/100;
            }
        }
    }

    return boundaries;
}
