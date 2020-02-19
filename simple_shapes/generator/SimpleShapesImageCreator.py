import numpy as np
import cairo as co
import math

image_width = 50

data = np.zeros((image_width,image_width,4), dtype=np.uint8)
surface = co.ImageSurface.create_for_data(
    data, co.FORMAT_ARGB32, image_width, image_width)


cr = co.Context(surface)

cr.set_source_rgb(1.0, 1.0, 1.0)
cr.paint()

cr.arc(image_width/2, image_width/2, image_width/4, 0,2*math.pi)
cr.set_source_rgb(1.0, 0.0, 0.0)
cr.fill()

surface.write_to_png("circle_red.png")


data = np.zeros((image_width,image_width,4), dtype=np.uint8)
surface = co.ImageSurface.create_for_data(
    data, co.FORMAT_ARGB32, image_width, image_width)


cr = co.Context(surface)

cr.set_source_rgb(1.0, 1.0, 1.0)
cr.paint()

cr.arc(image_width/2, image_width/2, image_width/4, 0,2*math.pi)
cr.set_source_rgb(0.0, 0.0, 1.0)
cr.fill()

surface.write_to_png("circle_blue.png")

data = np.zeros((image_width,image_width,4), dtype=np.uint8)
surface = co.ImageSurface.create_for_data(
    data, co.FORMAT_ARGB32, image_width, image_width)


cr = co.Context(surface)

cr.set_source_rgb(1.0, 1.0, 1.0)
cr.paint()

cr.arc(image_width/2, image_width/2, image_width/4, 0,2*math.pi)
cr.set_source_rgb(0.0, 1.0, 0.0)
cr.fill()
surface.write_to_png("circle_green.png")


data = np.zeros((image_width,image_width,4), dtype=np.uint8)
surface = co.ImageSurface.create_for_data(
    data, co.FORMAT_ARGB32, image_width, image_width)

cr = co.Context(surface)

cr.set_source_rgb(1.0, 1.0, 1.0)
cr.paint()

cr.rectangle(image_width/4, image_width/4, image_width/2, image_width/2)
cr.set_source_rgb(1.0, 0.0, 0.0)
cr.fill()

surface.write_to_png("square_red.png")


data = np.zeros((image_width,image_width,4), dtype=np.uint8)
surface = co.ImageSurface.create_for_data(
    data, co.FORMAT_ARGB32, image_width, image_width)

cr = co.Context(surface)

cr.set_source_rgb(1.0, 1.0, 1.0)
cr.paint()

cr.rectangle(image_width/4, image_width/4, image_width/2, image_width/2)
cr.set_source_rgb(0.0, 1.0, 0.0)
cr.fill()
surface.write_to_png("square_green.png")


data = np.zeros((image_width,image_width,4), dtype=np.uint8)
surface = co.ImageSurface.create_for_data(
    data, co.FORMAT_ARGB32, image_width, image_width)

cr = co.Context(surface)

cr.set_source_rgb(1.0, 1.0, 1.0)
cr.paint()

cr.rectangle(image_width/4, image_width/4, image_width/2, image_width/2)
cr.set_source_rgb(0.0, 0.0, 1.0)
cr.fill()
surface.write_to_png("square_blue.png")


data = np.zeros((image_width,image_width,4), dtype=np.uint8)
surface = co.ImageSurface.create_for_data(
    data, co.FORMAT_ARGB32, image_width, image_width)

cr = co.Context(surface)

cr.set_source_rgb(1.0, 1.0, 1.0)
cr.paint()

cr.move_to(image_width/2, image_width/2 - (image_width/4)*np.sqrt(3)/2)
cr.line_to(image_width/2 - image_width/4, image_width/2 + (image_width/4)*np.sqrt(3)/2)
cr.line_to(image_width/2 + image_width/4, image_width/2 + (image_width/4)*np.sqrt(3)/2)

cr.set_source_rgb(1.0, 0.0, 0.0)
cr.fill()

surface.write_to_png("triangle_red.png")

data = np.zeros((image_width,image_width,4), dtype=np.uint8)
surface = co.ImageSurface.create_for_data(
    data, co.FORMAT_ARGB32, image_width, image_width)

cr = co.Context(surface)

cr.set_source_rgb(1.0, 1.0, 1.0)
cr.paint()

cr.move_to(image_width/2, image_width/2 - (image_width/4)*np.sqrt(3)/2)
cr.line_to(image_width/2 - image_width/4, image_width/2 + (image_width/4)*np.sqrt(3)/2)
cr.line_to(image_width/2 + image_width/4, image_width/2 + (image_width/4)*np.sqrt(3)/2)

cr.set_source_rgb(0.0, 1.0, 0.0)
cr.fill()
surface.write_to_png("triangle_green.png")

data = np.zeros((image_width,image_width,4), dtype=np.uint8)
surface = co.ImageSurface.create_for_data(
    data, co.FORMAT_ARGB32, image_width, image_width)

cr = co.Context(surface)

cr.set_source_rgb(1.0, 1.0, 1.0)
cr.paint()

cr.move_to(image_width/2, image_width/2 - (image_width/4)*np.sqrt(3)/2)
cr.line_to(image_width/2 - image_width/4, image_width/2 + (image_width/4)*np.sqrt(3)/2)
cr.line_to(image_width/2 + image_width/4, image_width/2 + (image_width/4)*np.sqrt(3)/2)

cr.set_source_rgb(0.0, 0.0, 1.0)
cr.fill()

surface.write_to_png("triangle_blue.png")
