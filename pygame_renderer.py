import pygame as pg
import numpy as np

from renderer import Renderer
from controls import Controls


class PygameRenderer(Renderer):
    def __init__(self, controls: Controls):
        super().__init__(controls)
        pg.init()
        pg.display.set_caption("Mandelbrot Python Visualizer")
        info = pg.display.Info()
        screen_ratio = info.current_h / info.current_w
        self.screen_width = int(info.current_w // 1.3)
        self.screen_height = int(screen_ratio * self.screen_width)
        self.screen = pg.display.set_mode((self.screen_width, self.screen_height))
        self.mandelbrot_surface = pg.Surface(
            (self.screen.get_width(), self.screen.get_height())
        )
        self.text_surface = pg.Surface(
            (self.screen.get_width(), self.screen.get_height()), pg.SRCALPHA
        )
        self.clock = pg.time.Clock()
        self.font = pg.font.Font(None, 25)

    def render_pixels(self, pixels: np.ndarray | None):
        if pixels is not None:
            pixels_r = pg.surfarray.pixels2d(self.mandelbrot_surface)
            pixels_r[:] = pixels

    def _text_drop_shadow(self, message, offset):
        text_color = 255, 255, 255
        shadow_color = 0, 0, 100

        base = self.font.render(message, True, text_color)
        shadow = self.font.render(message, True, shadow_color)
        shadow.set_alpha(127)
        size = base.get_width() + offset, base.get_height() + offset
        img = pg.Surface(size, pg.SRCALPHA)
        img.blit(shadow, (offset, offset))
        img.blit(base, (0, 0))
        return img

    def _render_text(self, text: str, x: int, y: int):
        self.text_surface.blit(
            self._text_drop_shadow(
                text,
                3,
            ),
            (x, y),
        )

    def render_texts(self, texts: list[str]):
        y = 10
        for text in texts:
            self._render_text(text, 10, y)
            y += 30

    def display(self):
        self.screen.fill((0, 0, 0))
        self.screen.blit(self.mandelbrot_surface, (0, 0))
        self.screen.blit(self.text_surface, (0, 0))
        self.text_surface.fill((0, 0, 0, 0))
        pg.display.flip()

    def handle_input(self):
        pressed_key = pg.key.get_pressed()

        if pressed_key[pg.K_RIGHT]:
            self.controls.increase_iters()
        if pressed_key[pg.K_LEFT]:
            self.controls.decrease_iters()
        if pressed_key[pg.K_a]:
            self.controls.left()
        if pressed_key[pg.K_d]:
            self.controls.right()
        if pressed_key[pg.K_w]:
            self.controls.up()
        if pressed_key[pg.K_s]:
            self.controls.down()
        if pressed_key[pg.K_UP]:
            self.controls.zoomin()
        if pressed_key[pg.K_DOWN]:
            self.controls.zoomout()
        if pressed_key[pg.K_ESCAPE]:
            self.controls.quit = True

        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.controls.quit = True
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_c:
                    self.controls.switch_worker()
            elif event.type == pg.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.controls.start_pan(event.pos[0], event.pos[1])
            elif event.type == pg.MOUSEBUTTONUP:
                if event.button == 1:
                    self.controls.stop_pan()
            elif event.type == pg.MOUSEMOTION:
                self.controls.move_pan(
                    event.pos[0], event.pos[1], self.screen_width, self.screen_height
                )
            elif event.type == pg.MOUSEWHEEL:
                if event.y > 0:
                    self.controls.zoomin(10)
                else:
                    self.controls.zoomout(10)

    def quit(self):
        pg.quit()
