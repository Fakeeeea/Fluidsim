#ifndef FLUIDSIM_SIM_PARSER_H
#define FLUIDSIM_SIM_PARSER_H

#include "types.h"

void parse_settings(settings *s);
int_b* parse_spawn(int_b max_size, int *n_spawn);
float_b* parse_obstacles(int_b max_size, int *n_obstacles);

#endif //FLUIDSIM_SIM_PARSER_H
