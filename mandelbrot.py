import numba
import numpy as np

ESCAPE_RADIUS_SQ = np.float32(100.0)


@numba.njit(fastmath=True)
def compute_mandelbrot_pixel(x, y, max_iters):
    real = np.float32(0.0)
    imag = np.float32(0.0)
    for iteration in range(max_iters):
        real_sq = real * real
        imag_sq = imag * imag
        if real_sq + imag_sq > ESCAPE_RADIUS_SQ:
            # Simple grayscale color based on iteration count
            color = 255 - int(255 * iteration / max_iters)
            return color, color, color
        imag = 2.0 * real * imag + y
        real = real_sq - imag_sq + x
    # Return black for points inside the Mandelbrot set
    return 0, 0, 0


@numba.njit(fastmath=True)
def generate_mandelbrot_set(pixels, cx, cy, max_iters, start_line, end_line):
    W = pixels.shape[0]
    for y in range(start_line, end_line + 1):
        cy_y = cy[y]
        for x in range(W):
            r, g, b = compute_mandelbrot_pixel(cx[x], cy_y, max_iters)
            pixels[x, y] = (r << 16) | (g << 8) | b
