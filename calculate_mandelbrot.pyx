# cython: boundscheck=False
# cython: wraparound=False

from libc.math cimport sqrt, sin, log

cpdef void calculate_mandelbrot(double[:, ::1] pixels_a, double[::1] cx, double[::1] cy, int max_iters, int start_line, int end_line):
    cdef int W = pixels_a.shape[0]
    cdef int x, y

    cdef double radius_squared = 100.0
    cdef double real_part = 0.0
    cdef double imag_part = 0.0
    cdef double log_radius = log(radius_squared)
    cdef double log_2 = log(2.0)
    cdef double dist_squared = 0.0
    cdef double temp = 0.0
    cdef double m = 0.0
    cdef int r, g, b

    for y in range(start_line, end_line + 1):
        for x in range(W):
            real_part, imag_part = 0.0, 0.0
            r, g, b = 0, 0, 0
            
            for iteration in range(max_iters):
                real_part, imag_part = real_part * real_part - imag_part * imag_part + cx[x], 2 * real_part * imag_part + cy[y]
                dist_squared = real_part * real_part + imag_part * imag_part

                if dist_squared > radius_squared:
                    temp = iteration - log(log(dist_squared) / log_radius) / log_2
                    m = sqrt(temp / max_iters)
                    m *= 30

                    r = <int>((0.5 * sin(0.3 * m) + 0.5) * 200)
                    g = <int>((0.5 * sin(0.4 * m) + 0.5) * 200)
                    b = <int>((0.5 * sin(0.5 * m) + 0.5) * 255)

                    break

            pixels_a[x, y] = r << 16 | g << 8 | b