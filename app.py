import numpy as np
import multiprocessing as mp
from threading import Thread
from multiprocessing import Process
import time
from mandelbrot import calculate_mandelbrot


class WorkerType:
    multiprocessing = "Multiprocessing"
    threading = "Threading"


class MandelbrotVisualizer:
    def __init__(
        self,
        screen_width,
        screen_ratio,
        number_of_workers,
        worker_type,
        max_iters,
    ):
        self.screen_width = screen_width
        self.screen_ratio = screen_ratio
        self.screen_height = int(screen_ratio * screen_width)
        self.number_of_workers = number_of_workers
        self.worker_type = worker_type

        # used to sync workers
        self.a_proc_barrier = mp.Barrier(number_of_workers + 1)
        self.b_proc_barrier = mp.Barrier(number_of_workers + 1)
        self.a_thread_barrier = mp.Barrier(number_of_workers + 1)
        self.b_thread_barrier = mp.Barrier(number_of_workers + 1)

        # used to share memory between workers
        self.cx = mp.RawArray("d", self.screen_width)
        self.cx_a = np.frombuffer(self.cx, dtype=np.double)
        self.cy = mp.RawArray("d", self.screen_height)
        self.cy_a = np.frombuffer(self.cy, dtype=np.double)
        self.pixels = mp.RawArray("d", self.screen_width * self.screen_height)
        self.pixels_a = np.frombuffer(self.pixels, dtype=np.double).reshape(
            self.screen_width, self.screen_height
        )
        self.max_iters = mp.RawValue("i", max_iters)

        # create render workers and wait for work
        for id in range(self.number_of_workers):
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
        pixels_a = np.frombuffer(self.pixels, dtype=np.double).reshape(
            self.screen_width, self.screen_height
        )
        lines_per_thread = self.screen_height // self.number_of_workers
        start_line, end_line = (
            (id * lines_per_thread),
            ((id + 1) * lines_per_thread) - 1,
        )

        while True:
            a_barrier.wait()
            calculate_mandelbrot(
                pixels_a, self.cx, self.cy, self.max_iters.value, start_line, end_line
            )
            b_barrier.wait()

    def update(self, centerX, centerY, zoom):
        centerX = np.clip(centerX, -2, 2)
        centerY = np.clip(centerY, -2, 2)
        self.max_iters.value = np.clip(self.max_iters.value, 50, 500)
        zoom = np.clip(zoom, -2, 2)
        zoomX = zoom + zoom * self.screen_ratio
        self.cx_a[:] = np.linspace(centerX - zoomX, centerX + zoomX, self.screen_width)
        self.cy_a[:] = np.linspace(centerY - zoom, centerY + zoom, self.screen_height)

        if self.worker_type == WorkerType.multiprocessing:
            self.a_proc_barrier.wait()
            self.b_proc_barrier.wait()
        elif self.worker_type == WorkerType.threading:
            self.a_thread_barrier.wait()
            self.b_thread_barrier.wait()

    def draw(self, screen, font, fps):
        # display mandelbrot set
        pixels_r = pg.surfarray.pixels2d(screen)
        pixels_r[:] = self.pixels_a
        del pixels_r

        # display text info
        c = (255, 255, 255)
        screen.blit(
            font.render(
                f"controls: (w, a, s, d, up, down)",
                True,
                c,
            ),
            (10, 10),
        )
        screen.blit(
            font.render(f"Worker type: {self.worker_type} (1/2 to change)", True, c),
            (10, 30),
        )
        screen.blit(
            font.render(
                f"Iters: {self.max_iters.value} (right/left arrow to change)", True, c
            ),
            (10, 50),
        )
        screen.blit(font.render(f"FPS: {fps}", True, c), (10, 70))


def handle_input1(viz: MandelbrotVisualizer):
    while True:
        time.sleep(0.1)
        pressed_key = pg.key.get_pressed()
        if pressed_key[pg.K_RIGHT]:
            viz.max_iters.value += 2
        if pressed_key[pg.K_LEFT]:
            viz.max_iters.value -= 2
        if pressed_key[pg.K_1]:
            viz.worker_type = WorkerType.multiprocessing
        if pressed_key[pg.K_2]:
            viz.worker_type = WorkerType.threading


def handle_input2(centerX, centerY, zoom, delta_time):
    pressed_key = pg.key.get_pressed()
    if pressed_key[pg.K_a]:
        centerX -= zoom * delta_time
    if pressed_key[pg.K_d]:
        centerX += zoom * delta_time
    if pressed_key[pg.K_w]:
        centerY -= zoom * delta_time
    if pressed_key[pg.K_s]:
        centerY += zoom * delta_time
    if pressed_key[pg.K_UP]:
        zoom -= zoom * delta_time
    if pressed_key[pg.K_DOWN]:
        zoom += zoom * delta_time
    if pressed_key[pg.K_ESCAPE]:
        pg.quit()
        exit()

    return centerX, centerY, zoom


def main():
    viz = MandelbrotVisualizer(
        screen_width=800,
        screen_ratio=9 / 16,
        number_of_workers=mp.cpu_count(),
        worker_type=WorkerType.multiprocessing,
        max_iters=80,
    )

    pg.init()
    screen = pg.display.set_mode((viz.screen_width, viz.screen_height), pg.SCALED)
    pg.display.set_caption("Mandelbrot python visualizer by github.com/idtareq")

    clock = pg.time.Clock()
    font = pg.font.Font(None, 25)

    Thread(target=handle_input1, args=(viz,), daemon=True).start()

    centerX = 0
    centerY = 0
    zoom = 2

    while True:
        delta_time = clock.tick(30) / 1000.0

        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                exit()

        centerX, centerY, zoom = handle_input2(centerX, centerY, zoom, delta_time)

        viz.update(centerX, centerY, zoom)
        viz.draw(screen, font, round(clock.get_fps(), 1))

        pg.display.flip()


if __name__ == "__main__":
    # Import pygame within this scope to prevent the multiprocessing library from redundantly importing it for each worker process.
    import pygame as pg

    main()
