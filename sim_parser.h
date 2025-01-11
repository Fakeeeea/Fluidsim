#ifndef FLUIDSIM_SIM_PARSER_H
#define FLUIDSIM_SIM_PARSER_H

#include "types.h"

void parse_settings(settings *s);
boundary* parse_spawn(boundary screen, int *n_spawn);

#endif //FLUIDSIM_SIM_PARSER_H
