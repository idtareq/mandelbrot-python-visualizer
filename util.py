import pygame as pg


def text_drop_shadow(font, message, offset, fontcolor, shadowcolor):
    base = font.render(message, True, fontcolor)
    shadow = font.render(message, True, shadowcolor)
    shadow.set_alpha(127)
    size = base.get_width() + offset, base.get_height() + offset
    img = pg.Surface(size, pg.SRCALPHA)
    img.blit(shadow, (offset, offset))
    img.blit(base, (0, 0))
    return img
