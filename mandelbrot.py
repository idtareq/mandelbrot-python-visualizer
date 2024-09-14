import numba
import numpy as np

ESCAPE_RADIUS_SQ = 100.0


@numba.njit
def hsv_to_rgb(h, s, v):
    """
    Converts HSV color space to RGB.
    h, s, v should be in the range [0, 1].
    Returns r, g, b in the range [0, 1].
    """
    h = h % 1.0  # Ensure h is in [0, 1)
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6

    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:  # i == 5
        r, g, b = v, p, q

    return r, g, b


@numba.njit
def compute_color(iteration, dist_sq, max_iters):
    """
    Computes the color based on the smoothed iteration count.
    """
    if iteration < max_iters:
        # Smooth the iteration count to reduce banding
        log_zn = np.log(dist_sq) / 2
        nu = np.log(log_zn / np.log(2)) / np.log(2)
        smoothed_iter = iteration + 1 - nu
    else:
        smoothed_iter = iteration

    # Map the smoothed iteration to a hue value
    hue = smoothed_iter / max_iters
    saturation = 1.0
    value = (
        1.0 if iteration < max_iters else 0.0
    )  # Set value to 0 for points inside the set

    # Convert HSV to RGB
    r_f, g_f, b_f = hsv_to_rgb(hue, saturation, value)

    # Convert to 8-bit RGB values
    r = int(r_f * 255)
    g = int(g_f * 255)
    b = int(b_f * 255)

    return r, g, b


@numba.njit
def compute_mandelbrot_pixel(x, y, max_iters):
    """
    Computes the color of a single pixel in the Mandelbrot set.
    """
    real = imag = 0.0
    for iteration in range(max_iters):
        real_sq = real * real
        imag_sq = imag * imag
        dist_sq = real_sq + imag_sq
        if dist_sq > ESCAPE_RADIUS_SQ:
            return compute_color(iteration, dist_sq, max_iters)
        imag = 2 * real * imag + y
        real = real_sq - imag_sq + x
    # Return black for points inside the Mandelbrot set
    return 0, 0, 0


def generate_mandelbrot_set(pixels, cx, cy, max_iters, start_line, end_line):
    """
    Generates the Mandelbrot set image by computing each pixel's color.
    """
    W = pixels.shape[0]

    for y in range(start_line, end_line + 1):
        for x in range(W):
            r, g, b = compute_mandelbrot_pixel(cx[x], cy[y], max_iters)
            pixels[x, y] = (r << 16) | (g << 8) | b
