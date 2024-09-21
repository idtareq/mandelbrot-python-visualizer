import logging
from controls import Controls
from mandelbrot import generate_mandelbrot_set
from mandelbrot_visualizer import MandelbrotVisualizer
import multiprocessing as mp
from pygame_renderer import PygameRenderer
from worker import WorkerType
# from pyglet_renderer import PygletRenderer

numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def main():
    controls = Controls(WorkerType.PROCESS)
    renderer = PygameRenderer(controls)

    viz = MandelbrotVisualizer(
        screen_width=renderer.screen_width,
        screen_height=renderer.screen_height,
        number_of_workers=1 + mp.cpu_count() // 2,
        max_iters=80,
        controls=controls,
        worker_function=generate_mandelbrot_set,
    )

    while True:
        renderer.handle_input()
        viz.update()
        renderer.render_texts(viz.get_texts())
        with viz.get_pixels() as pixels:
            renderer.render_pixels(pixels)
        renderer.display()
        if controls.quit:
            logger.debug("Quitting.")
            viz.terminate()
            renderer.quit()
            break


if __name__ == "__main__":
    main()
