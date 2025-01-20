#define _USE_MATH_DEFINES

#include <stdlib.h>
#include <windows.h>
#include <time.h>
#include <math.h>

#include "types.h"
#include "sim_math.cuh"
#include "render_math.cuh"
#include "sim_parser.h"

void summon_particles(int n_boundaries, int_b *boundaries);

char * szAppName = "Fluid";

particles sim;

settings s;
RECT rect;
cells cell_ll;
obstacles obs;

LRESULT CALLBACK WndProc (HWND, UINT, WPARAM, LPARAM);

int WINAPI WinMain (HINSTANCE hInstance, HINSTANCE hPrevInstance, PSTR szCmdLine, int iCmdShow){
    HWND hwnd;
    MSG msg;
    WNDCLASS wndclass;

    wndclass.style = CS_HREDRAW | CS_VREDRAW ;
    wndclass.lpfnWndProc = WndProc;
    wndclass.cbClsExtra = 0;
    wndclass.cbWndExtra = 0;
    wndclass.hInstance = hInstance;
    wndclass.hIcon = NULL;
    wndclass.hCursor = NULL;
    wndclass.hbrBackground = (HBRUSH) GetStockObject (WHITE_BRUSH) ;
    wndclass.lpszMenuName = NULL;
    wndclass.lpszClassName = szAppName;

    if(!RegisterClass(&wndclass)){
        MessageBox(NULL, TEXT("Window class registration failed!"), szAppName, MB_ICONERROR);
    }

    hwnd = CreateWindow (szAppName,
                         TEXT ("Fluid"),
                         WS_OVERLAPPEDWINDOW | WS_MAXIMIZE,
                         CW_USEDEFAULT,
                         CW_USEDEFAULT,
                         GetSystemMetrics(SM_CXSCREEN),
                         GetSystemMetrics(SM_CYSCREEN),
                         NULL,
                         NULL,
                         hInstance,
                         NULL);

    ShowWindow (hwnd, iCmdShow);
    UpdateWindow (hwnd);

    while (1) {
        while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT)
                return msg.wParam;
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        simulation_step_gpu( (int_v2){ rect.right * PIXELTOUNIT, rect.bottom * PIXELTOUNIT}, &sim, s, &cell_ll, &obs);
        InvalidateRect(hwnd, &rect, 0);
    }

    return msg.wParam;
}

LRESULT CALLBACK WndProc (HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    HDC hdc;
    PAINTSTRUCT ps;

    static HDC memDC = NULL;
    static HBITMAP memBitmap = NULL;
    static HBITMAP oldBitmap = NULL;
    static int *gpu_bitmap = NULL;

    switch (message)
    {
        case WM_SIZE:
            GetClientRect(hwnd, &rect);

            cell_ll.world_size = (int_v2) { (int) ceilf( (float) rect.right * PIXELTOUNIT),
                                            (int) ceilf( (float) rect.bottom * PIXELTOUNIT)};

            realloc_render_memory(&gpu_bitmap, (int_v2) { rect.right, rect.bottom });

            if (memBitmap) {
                DeleteObject(memBitmap);
            }

            memBitmap = CreateCompatibleBitmap(GetDC(hwnd), rect.right, rect.bottom);
            SelectObject(memDC, memBitmap);

            return 0;
        case WM_CREATE:
            GetClientRect(hwnd, &rect);

            //Initialize and parse simulation settings.
            memset(&s, 0, sizeof(settings));
            parse_settings(&s);

            const int n_particles = s.ss.n_particles;
            const float h = s.ss.smoothing_length;

            //Allocate memory for particle creation.
            sim.pos = (float_v2*) malloc(sizeof(float_v2) * n_particles);

            //Parse boundaries and spawn particles.
            int_v2 temp_world_size = (int_v2) { (int) ceilf( (float) rect.right * PIXELTOUNIT),
                                                (int) ceilf( (float) rect.bottom * PIXELTOUNIT)};
            int n_spawn;
            int_b *spawn = parse_spawn((int_b) {{0, 0}, temp_world_size}, &n_spawn);
            summon_particles(n_spawn, spawn);

            //Parse obstacles.
            obs.obstacles = parse_obstacles((int_b) {{0, 0}, temp_world_size}, &obs.n_obstacles);

            //Allocate gpu memory for the simulation.
            malloc_simulation_gpu(s.ss.n_particles, &sim, &obs);

            //Pre-calculate kernels used in simulation
            const float poly6 =  4.0f/( (float) M_PI*powf(h*h, 4));
            const float spiky =  -10.0f/( (float) M_PI*powf(h, 5));
            const float viscosity = 40.0f/( (float) M_PI*powf(h, 5));

            //Set the calculated kernels in the __constant__ memory.
            initialize_constants(poly6, spiky, viscosity);

            //Allocate memory for the cell linked list.
            create_cell_ll_gpu(&cell_ll, rect, s);

            //Allocate memory for rendering.
            allocate_render_memory(&gpu_bitmap, (int_v2) { rect.right, rect.bottom });

            //Create the dc for double buffering.
            memDC = CreateCompatibleDC(GetDC(hwnd));
            memBitmap = CreateCompatibleBitmap(GetDC(hwnd), rect.right, rect.bottom);
            oldBitmap = (HBITMAP)SelectObject(memDC, memBitmap);

            return 0;
        case WM_PAINT:
            hdc = BeginPaint(hwnd, &ps);

            BITMAPINFO bmi;

            ZeroMemory(&bmi, sizeof(BITMAPINFO));
            bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
            bmi.bmiHeader.biWidth = rect.right;
            bmi.bmiHeader.biHeight = rect.bottom;
            bmi.bmiHeader.biPlanes = 1;
            bmi.bmiHeader.biBitCount = 32;
            bmi.bmiHeader.biCompression = BI_RGB;

            int *colored_bitmap;
            colored_bitmap = get_colored_bitmap(sim.gpu_pos, (int_v2) {rect.right, rect.bottom}, s, cell_ll.entries, cell_ll.start_indices, gpu_bitmap, obs);

            SetDIBits(memDC, memBitmap, 0, rect.bottom, colored_bitmap, &bmi, DIB_RGB_COLORS);
            BitBlt(hdc, 0, 0, rect.right, rect.bottom, memDC, 0, 0, SRCCOPY);

            free(colored_bitmap);

            EndPaint(hwnd, &ps);
            return 0 ;
        case WM_DESTROY:

            free_simulation_memory(&sim);
            free_render_memory(gpu_bitmap);
            free(sim.pos);

            if (memBitmap) {
                SelectObject(memDC, oldBitmap);
                DeleteObject(memBitmap);
                DeleteDC(memDC);
            }

            PostQuitMessage (0) ;
            return 0 ;
    }
    return DefWindowProc (hwnd, message, wParam, lParam) ;
}

void summon_particles(int n_boundaries, int_b *boundaries)
{
    srand(time(NULL));

    const int particles_boundary = s.ss.n_particles/n_boundaries;
    const int particles_last_boundary = s.ss.n_particles - particles_boundary*(n_boundaries-1);

    if(s.ss.spawn_random)
    {
        for(int i = 0; i < n_boundaries; i++)
        {
            for(int j = 0; j < (i == n_boundaries - 1 ? particles_last_boundary : particles_boundary); j++)
            {
                const float px = boundaries[i].min.x + ((float) rand() / RAND_MAX) * (boundaries[i].max.x - boundaries[i].min.x);
                const float py = boundaries[i].min.y + ((float) rand() / RAND_MAX) * (boundaries[i].max.y - boundaries[i].min.y);
                sim.pos[i*particles_boundary + j] = (float_v2) {px, py};
            }
        }
    }
    else
    {
        for(int i = 0; i < n_boundaries; i++)
        {
            const int current_particles = (i == n_boundaries - 1) ? particles_last_boundary : particles_boundary;

            const float width = (float) (boundaries[i].max.x - boundaries[i].min.x);
            const float height = (float) (boundaries[i].max.y - boundaries[i].min.y);

            const int cols = (int)sqrt(current_particles);
            const int rows = (current_particles + cols - 1) / cols;

            const float dx = width / ( (float) cols + 1);
            const float dy = height / ( (float) rows + 1);

            for(int j = 0; j < current_particles; j++)
            {
                const int row = j / cols;
                const int col = j % cols;

                const float px = (float) boundaries[i].min.x + dx * (float) (col + 1);
                const float py = (float) boundaries[i].min.y + dy * (float) (row + 1);

                sim.pos[i*particles_boundary + j] = (float_v2){px, py};
            }
        }
    }
}