#include <math.h>
#include <windows.h>

#include "cell_ll.h"

inline int hash_cell(int max_hash, v2 pos)
{
    unsigned int hashx = (int)pos.x * 11;
    unsigned int hashy = (int)pos.y * 4079;
    return ((hashx + hashy)*7) % max_hash;
}

inline v2 cell_coordinate(v2 pos, float h)
{
    v2 cell;

    cell.x = floorf(pos.x/h);
    cell.y = floorf(pos.y/h);

    return cell;
}

void create_cells_grid(cells *grid, int n_particles, settings s)
{
    grid->size.x = ceil( (float) GetSystemMetrics(SM_CXSCREEN)/ s.smoothing_length);
    grid->size.y = ceil( (float) GetSystemMetrics(SM_CYSCREEN)/ s.smoothing_length);

    grid->entries = (entry*) malloc(sizeof(entry) * n_particles);
    grid->start_indices = (int*) malloc(sizeof(int) * n_particles);
}

void update_cells(cells *grid, particle *particles, int n_particles, settings s)
{
    for(int i = 0; i < n_particles; i++)
    {
        v2 cell_pos = cell_coordinate(particles[i].predicted_pos, s.smoothing_length);
        int cell_key = hash_cell(n_particles, cell_pos);
        //printf("placed particle %d in hash %d, coordinates: %f-%f\n", i, cell_key,cell_pos.x, cell_pos.y);
        grid->entries[i] = (entry) {cell_key,cell_pos,i};
        grid->start_indices[i] = -1;
    }

    qsort(grid->entries, n_particles, sizeof(entry), compare);

    int current_key = -1;
    int prev_key;

    for(int i = 0; i < n_particles; i++)
    {
        prev_key = current_key;
        current_key = grid->entries[i].cell_key;

        if(prev_key != current_key)
        {
            grid->start_indices[current_key] = i;
        }
    }
}

int compare(const void *a, const void*b)
{
    const entry A = *((entry*)a);
    const entry B = *((entry*)b);
    return (A.cell_key > B.cell_key) - (A.cell_key < B.cell_key);
}
