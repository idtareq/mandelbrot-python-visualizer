"""
Visualize the mandelbrot in Python using Multiprocessing or Multithreading

Due to the Python Global Interpreter Lock (GIL) multithreading has worst performance than multiprocessing.

Author: Tareq Ibrahim (idtareq@gmail.com)
"""

import numba
import numpy as np
import multiprocessing as mp
from threading import Thread
from multiprocessing import Process


@numba.jit
def mandlebrot_pixel(x, y, max_iters: int) -> int:
    r = 10
    z_real = 0
    z_imag = 0

    for n in range(max_iters):
        z_real, z_imag = (z_real**2 - z_imag**2 + x), (2 * z_real * z_imag + y)

        dist = np.sqrt(z_real**2 + z_imag**2)
        if dist > r:
            n -= np.log2(np.log(dist) / np.log(r))
            m = np.sqrt(n / max_iters)

            r, g, b = (
                int((np.sin(0.6 * m * 20) * 0.5 + 0.5) * 200),
                int((np.sin(0.7 * m * 20) * 0.5 + 0.5) * 200),
                int((np.sin(0.8 * m * 20) * 0.5 + 0.5) * 255),
            )

            return r << 16 | g << 8 | b

    return 0


def render_worker(id, a_barrier, b_barrier, N_WORKERS, W, H, pixels, cx, cy, max_iters):
    pixels_a = np.frombuffer(pixels, dtype=np.double).reshape(W, H)

    while True:
        a_barrier.wait()
        lines_per_thread = H // N_WORKERS
        start_line, end_line = (id * lines_per_thread), (
            (id + 1) * lines_per_thread
        ) - 1

        for y in range(start_line, end_line + 1):
            for x in range(W):
                pixels_a[x, y] = mandlebrot_pixel(cx[x], cy[y], max_iters.value)
        b_barrier.wait()


class Methods:
    multiprocessing = "Multiprocessing"
    threading = "Threading"


if __name__ == "__main__":
    import pygame as pg

    W = 800
    ratio = 9 / 16
    control_speed = 0.08
    method = Methods.multiprocessing

    centerX = 0
    centerY = 0
    zoom = 2
    H = int(ratio * W)

    N_WORKERS = mp.cpu_count()
    a_proc_barrier = mp.Barrier(N_WORKERS + 1)
    b_proc_barrier = mp.Barrier(N_WORKERS + 1)
    a_thread_barrier = mp.Barrier(N_WORKERS + 1)
    b_thread_barrier = mp.Barrier(N_WORKERS + 1)
    cx = mp.RawArray("d", W)
    cx_a = np.frombuffer(cx, dtype=np.double)
    cy = mp.RawArray("d", H)
    cy_a = np.frombuffer(cy, dtype=np.double)
    pixels = mp.RawArray("d", W * H)
    pixels_a = np.frombuffer(pixels, dtype=np.double).reshape(W, H)
    max_iters = mp.RawValue("i", 80)

    # create workers
    for id in range(N_WORKERS):
        worker_args = N_WORKERS, W, H, pixels, cx, cy, max_iters

        Process(
            target=render_worker,
            daemon=True,
            args=(id, a_proc_barrier, b_proc_barrier, *worker_args),
        ).start()

        Thread(
            target=render_worker,
            daemon=True,
            args=(id, a_thread_barrier, b_thread_barrier, *worker_args),
        ).start()

    pg.init()
    screen = pg.display.set_mode((W, H), pg.SCALED)
    pg.display.set_caption("Mandelbrot python visualizer by github.com/idtareq")

    clock = pg.time.Clock()
    font = pg.font.Font(None, 25)

    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                exit()

        pressed_key = pg.key.get_pressed()
        if pressed_key[pg.K_a]:
            centerX -= zoom * control_speed
        if pressed_key[pg.K_d]:
            centerX += zoom * control_speed
        if pressed_key[pg.K_w]:
            centerY -= zoom * control_speed
        if pressed_key[pg.K_s]:
            centerY += zoom * control_speed
        if pressed_key[pg.K_UP]:
            zoom -= zoom * control_speed
        if pressed_key[pg.K_DOWN]:
            zoom += zoom * control_speed
        if pressed_key[pg.K_RIGHT]:
            max_iters.value += 2
        if pressed_key[pg.K_LEFT]:
            max_iters.value -= 2
        if pressed_key[pg.K_ESCAPE]:
            pg.quit()
            exit()
        if pressed_key[pg.K_p]:
            method = Methods.multiprocessing
        if pressed_key[pg.K_t]:
            method = Methods.threading

        centerX = np.clip(centerX, -2, 2)
        centerY = np.clip(centerY, -2, 2)
        max_iters.value = np.clip(max_iters.value, 50, 500)
        zoom = np.clip(zoom, -2, 2)
        zoomX = zoom + zoom * ratio
        cx_a[:] = np.linspace(centerX - zoomX, centerX + zoomX, W)
        cy_a[:] = np.linspace(centerY - zoom, centerY + zoom, H)

        # synchronize main thread with worker threads
        if method == Methods.multiprocessing:
            a_proc_barrier.wait()
            b_proc_barrier.wait()
        elif method == Methods.threading:
            a_thread_barrier.wait()
            b_thread_barrier.wait()

        pixels_r = pg.surfarray.pixels2d(screen)
        pixels_r[:] = pixels_a
        del pixels_r

        fps = round(clock.get_fps(), 1)
        c = (255, 255, 255)
        screen.blit(
            font.render(
                f"controls: (w, a, s, d, up, down), (t or p) to switch method, (right or left) to change iters",
                True,
                c,
            ),
            (10, 10),
        )
        screen.blit(font.render(f"FPS: {fps}", True, c), (10, 30))
        screen.blit(font.render(f"ITERS: {max_iters.value}", True, c), (10, 50))
        screen.blit(font.render(f"METHOD: {method}", True, c), (10, 70))

        pg.display.flip()

        clock.tick(30)
