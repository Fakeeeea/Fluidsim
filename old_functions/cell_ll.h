#ifndef FLUIDSIM_CELL_LL_H
#define FLUIDSIM_CELL_LL_H

#include <windows.h>
#include "types.h"

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

int hash_cell(int max_hash, v2 pos);
v2 cell_coordinate(v2 pos, float h);

void create_cells_grid(cells *grid, int n_particles, settings s);
void update_cells(cells *grid, particle *particles, int n_particles, settings s);

int compare(const void *a, const void*b);

#endif //FLUIDSIM_CELL_LL_H
