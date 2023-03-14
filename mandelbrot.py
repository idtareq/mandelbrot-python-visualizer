import math
import numba


@numba.jit
def clac_pixel(
    max_iters, real_part, imag_part, x, y, radius_squared, log_radius, log_2
):
    for iteration in range(max_iters):
        new_real_part = real_part * real_part - imag_part * imag_part + x
        new_imag_part = 2 * real_part * imag_part + y
        real_part, imag_part = new_real_part, new_imag_part
        dist_squared = real_part * real_part + imag_part * imag_part

        if dist_squared > radius_squared:
            temp = iteration - math.log(math.log(dist_squared) / log_radius) / log_2
            m = math.sqrt(temp / max_iters)
            m *= 30

            r = int((0.5 * math.sin(0.3 * m) + 0.5) * 200)
            g = int((0.5 * math.sin(0.4 * m) + 0.5) * 200)
            b = int((0.5 * math.sin(0.5 * m) + 0.5) * 255)

            return r, b, g
    return 0, 0, 0


def calculate_mandelbrot(pixels_a, cx, cy, max_iters, start_line, end_line):
    W = pixels_a.shape[0]

    radius_squared = 100.0
    log_radius = math.log(radius_squared)
    log_2 = math.log(2.0)

    for y in range(start_line, end_line + 1):
        for x in range(W):
            real_part = imag_part = 0.0
            r, b, g = clac_pixel(
                max_iters,
                real_part,
                imag_part,
                cx[x],
                cy[y],
                radius_squared,
                log_radius,
                log_2,
            )

            pixels_a[x, y] = (r << 16) | (g << 8) | b
