#include <windows.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "render.h"


//This is not slow, its worse than slow. It's a whole disaster
void draw_scene(HDC hdc, RECT rect, v2 *positions, int n_particles, int UNITTOPIXEL)
{
    for(int i = 0; i < n_particles; i++)
    {
        v2 pos = {positions[i].x * UNITTOPIXEL, rect.bottom - positions[i].y * UNITTOPIXEL};
        Ellipse(hdc, pos.x + 2, pos.y + 2, pos.x - 2, pos.y - 2);
    }
}