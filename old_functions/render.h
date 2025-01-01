#ifndef FLUIDSIM_RENDER_H
#define FLUIDSIM_RENDER_H

#include <windows.h>
#include "types.h"
#include "cell_ll.h"


void draw_scene(HDC hdc, RECT rect, v2 *positions, int n_particles, int UNITTOPIXEL);

#endif //FLUIDSIM_RENDER_H
