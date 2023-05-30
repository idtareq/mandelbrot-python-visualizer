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


class WorkerType:
    multiprocessing = "Multiprocessing"
    threading = "Threading"


@numba.jit
def mandlebrot_calc_pixel(x, y, max_iters: int) -> int:
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


class MandelbrotVisualizer:
    def __init__(self, W, H, N_WORKERS, calc_fn):
        self.W = W
        self.H = H
        self.N_WORKERS = N_WORKERS
        self.calc_fn = calc_fn

        # to sync workers
        self.a_proc_barrier = mp.Barrier(N_WORKERS + 1)
        self.b_proc_barrier = mp.Barrier(N_WORKERS + 1)
        self.a_thread_barrier = mp.Barrier(N_WORKERS + 1)
        self.b_thread_barrier = mp.Barrier(N_WORKERS + 1)

        # to share memory between workers
        self.cx = mp.RawArray("d", W)
        self.cx_a = np.frombuffer(self.cx, dtype=np.double)
        self.cy = mp.RawArray("d", H)
        self.cy_a = np.frombuffer(self.cy, dtype=np.double)
        self.pixels = mp.RawArray("d", W * H)
        self.pixels_a = np.frombuffer(self.pixels, dtype=np.double).reshape(W, H)
        self.max_iters = mp.RawValue("i", 80)

        # create render workers and wait for work
        for id in range(self.N_WORKERS):
            Process(
                target=self.render_worker,
                daemon=True,
                args=(id, self.a_proc_barrier, self.b_proc_barrier),
            ).start()
            Thread(
                target=self.render_worker,
                daemon=True,
                args=(id, self.a_thread_barrier, self.b_thread_barrier),
            ).start()

    def render_worker(self, id, a_barrier, b_barrier):
        pixels_a = np.frombuffer(self.pixels, dtype=np.double).reshape(self.W, self.H)
        lines_per_thread = self.H // self.N_WORKERS
        start_line, end_line = (id * lines_per_thread), (
            (id + 1) * lines_per_thread
        ) - 1

        while True:
            a_barrier.wait()
            for y in range(start_line, end_line + 1):
                for x in range(self.W):
                    pixels_a[x, y] = self.calc_fn(
                        self.cx[x], self.cy[y], self.max_iters.value
                    )
            b_barrier.wait()

    def update(self, method: WorkerType, centerX, centerY, zoom):
        centerX = np.clip(centerX, -2, 2)
        centerY = np.clip(centerY, -2, 2)
        self.max_iters.value = np.clip(self.max_iters.value, 50, 500)
        zoom = np.clip(zoom, -2, 2)
        zoomX = zoom + zoom * ratio
        self.cx_a[:] = np.linspace(centerX - zoomX, centerX + zoomX, self.W)
        self.cy_a[:] = np.linspace(centerY - zoom, centerY + zoom, self.H)

        if method == WorkerType.multiprocessing:
            self.a_proc_barrier.wait()
            self.b_proc_barrier.wait()
        elif method == WorkerType.threading:
            self.a_thread_barrier.wait()
            self.b_thread_barrier.wait()


def handle_input(pg, max_iters, centerX, centerY, zoom, method):
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
        method = WorkerType.multiprocessing
    if pressed_key[pg.K_t]:
        method = WorkerType.threading

    return centerX, centerY, zoom, method


def display(pg, screen, viz):
    # display mandelbrot set
    pixels_r = pg.surfarray.pixels2d(screen)
    pixels_r[:] = viz.pixels_a
    del pixels_r

    # display text info
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
    screen.blit(font.render(f"ITERS: {viz.max_iters.value}", True, c), (10, 50))
    screen.blit(font.render(f"METHOD: {method}", True, c), (10, 70))


if __name__ == "__main__":
    # import here so the multiprocessing library doesn't import it again unnecessarily per worker process
    import pygame as pg

    W = 800
    ratio = 9 / 16
    control_speed = 0.08
    method = WorkerType.multiprocessing

    centerX = 0
    centerY = 0
    zoom = 2
    H = int(ratio * W)
    N_WORKERS = mp.cpu_count()

    viz = MandelbrotVisualizer(W, H, N_WORKERS, mandlebrot_calc_pixel)

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

        centerX, centerY, zoom, method = handle_input(
            pg, viz.max_iters, centerX, centerY, zoom, method
        )

        viz.update(method, centerX, centerY, zoom)
        display(pg, screen, viz)

        pg.display.flip()

        clock.tick(30)
