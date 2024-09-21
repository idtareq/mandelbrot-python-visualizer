import pyglet
import numpy as np

from renderer import Renderer
from controls import Controls


class PygletRenderer(Renderer):
    def __init__(self, controls: Controls):
        super().__init__(controls)
        display = pyglet.canvas.get_display()
        screen = display.get_default_screen()
        screen_ratio = screen.height / screen.width
        self.screen_width = int(screen.width // 1.3)
        self.screen_height = int(screen_ratio * self.screen_width)
        self.window = pyglet.window.Window(
            width=self.screen_width,
            height=self.screen_height,
            caption="Mandelbrot Python Visualizer",
            resizable=False,
        )
        self.image = pyglet.image.ImageData(
            self.screen_width,
            self.screen_height,
            "RGBA",
            pitch=self.screen_width * 4,
            data=None,
        )

        self.batch = pyglet.graphics.Batch()
        self.labels = []
        self.font_size = 12

        self.keys = pyglet.window.key.KeyStateHandler()
        self.window.push_handlers(self.keys)
        self.window.push_handlers(self)

        self.quit_flag = False
        self.labels = []

        for _ in range(2):
            self.labels.append(
                pyglet.text.Label(
                    "",
                    font_name="Arial",
                    font_size=self.font_size,
                    x=0,
                    y=0,
                    color=(255, 255, 255, 255),
                    batch=self.batch,
                )
            )

    def render_pixels(self, pixels: np.ndarray):
        if pixels is not None:
            pixels = pixels.astype(np.uint32)

            red = ((pixels >> 16) & 0xFF).astype(np.uint8)
            green = ((pixels >> 8) & 0xFF).astype(np.uint8)
            blue = (pixels & 0xFF).astype(np.uint8)
            alpha = np.full_like(red, 255, dtype=np.uint8)

            colored_pixels = np.stack((red, green, blue, alpha), axis=-1)
            colored_pixels = np.transpose(colored_pixels, (1, 0, 2))

            self.image.set_data("RGBA", self.screen_width * 4, colored_pixels.tobytes())

    def _render_text(self, idx, text: str, x: int, y: int):
        self.labels[idx].text = text
        self.labels[idx].x = x
        self.labels[idx].y = y

    def render_texts(self, texts: list):
        y = self.screen_height - 30
        for idx, text in enumerate(texts):
            self._render_text(idx, text, 10, y)
            y -= 30

    def display(self):
        pyglet.clock.tick()
        self.window.dispatch_events()
        self.on_draw()
        self.window.flip()

    def handle_input(self):
        if self.keys[pyglet.window.key.RIGHT]:
            self.controls.increase_iters()
        if self.keys[pyglet.window.key.LEFT]:
            self.controls.decrease_iters()
        if self.keys[pyglet.window.key.A]:
            self.controls.left()
        if self.keys[pyglet.window.key.D]:
            self.controls.right()
        if self.keys[pyglet.window.key.W]:
            self.controls.down()
        if self.keys[pyglet.window.key.S]:
            self.controls.up()
        if self.keys[pyglet.window.key.UP]:
            self.controls.zoomin()
        if self.keys[pyglet.window.key.DOWN]:
            self.controls.zoomout()
        if self.keys[pyglet.window.key.ESCAPE]:
            self.controls.quit = True

    def on_draw(self):
        self.window.clear()
        self.image.blit(0, 0)
        self.batch.draw()

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.C:
            self.controls.switch_worker()

    def on_mouse_press(self, x, y, button, modifiers):
        if button == pyglet.window.mouse.LEFT:
            self.controls.start_pan(x, y)

    def on_mouse_release(self, x, y, button, modifiers):
        if button == pyglet.window.mouse.LEFT:
            self.controls.stop_pan()

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if buttons & pyglet.window.mouse.LEFT:
            self.controls.move_pan(x, y, self.screen_width, self.screen_height)

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        if scroll_y > 0:
            self.controls.zoomin(10)
        else:
            self.controls.zoomout(10)

    def quit(self):
        self.window.close()
        pyglet.app.exit()

    def on_close(self):
        self.controls.quit = True
