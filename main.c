#define _USE_MATH_DEFINES

#include <stdlib.h>
#include <windows.h>
#include <time.h>
#include <math.h>

#include "types.h"
#include "sim_math.cuh"
#include "render_math.cuh"

#define PIXELTOUNIT (1.0f/10)

#define N_PARTICLES 10000

void initialize_settings();
void create_particles(int min_x, int min_y, int max_x, int max_y);

char * szAppName = "Fluid";

particles sim;

settings s;
RECT rect;
cells cell_ll;

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
                         WS_OVERLAPPEDWINDOW,
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
            if (msg.message == WM_QUIT) return msg.wParam;
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        simulation_step_gpu((v2) {rect.right*PIXELTOUNIT, rect.bottom*PIXELTOUNIT}, &sim, N_PARTICLES, s, &cell_ll);
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

            cell_ll.size = (v2) {rect.right * PIXELTOUNIT, rect.bottom * PIXELTOUNIT};
            realloc_render_memory(&gpu_bitmap, (v2){rect.right, rect.bottom});

            if (memBitmap) {
                DeleteObject(memBitmap);
            }

            memBitmap = CreateCompatibleBitmap(GetDC(hwnd), rect.right, rect.bottom);
            SelectObject(memDC, memBitmap);

            return 0;
        case WM_CREATE:
            GetClientRect(hwnd, &rect);

            sim.pos = (v2*) malloc(sizeof(v2)*N_PARTICLES);

            create_particles((rect.right/2 - rect.right/4)*PIXELTOUNIT, (rect.bottom/2 - 300)*PIXELTOUNIT, (rect.right/2 + rect.right/4)*PIXELTOUNIT, (rect.bottom/2 + 300)*PIXELTOUNIT);

            malloc_simulation_gpu(N_PARTICLES, &sim);

            initialize_settings();

            float poly6 = 4.0f/(M_PI*pow(s.smoothing_length*s.smoothing_length, 4));
            float spiky = -10.0f/(M_PI*pow(s.smoothing_length, 5));
            float viscosity = 40.0f/(M_PI*pow(s.smoothing_length, 5));

            initialize_constants(poly6, spiky, viscosity);

            create_cell_ll_gpu(&cell_ll, N_PARTICLES, rect, s);

            allocate_render_memory(&gpu_bitmap, (v2){rect.right, rect.bottom});

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
            colored_bitmap = get_colored_bitmap(sim.gpu_pos, (v2) {rect.right, rect.bottom}, s, cell_ll.entries, cell_ll.start_indices, N_PARTICLES, gpu_bitmap);

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


void initialize_settings()
{
    s.mass = 1;
    s.rest_density = 0.1f;
    s.viscosity = 0.3f;
    s.smoothing_length = 5;
    s.time_step = 1/30.f;
    s.pressure_multiplier = 30;
    s.bounce_multiplier = 0.4f;
    s.near_pressure_multiplier = 10.0f;
}

void create_particles(int min_x, int min_y, int max_x, int max_y)
{
    srand(time(NULL));

    for(int i = 0; i < N_PARTICLES; i++)
    {
        sim.pos[i].x = min_x + ((float)rand() / RAND_MAX) * (max_x - min_x);
        sim.pos[i].y = min_y + ((float)rand() / RAND_MAX) * (max_y - min_y);
    }
}
