"""
Generate synthetic "video" date, which is really a bunch of colors boxes
jumping around. The label for this dataset consists of five classes:

0 - box got much smaller
1 - box got smaller
3 - box stayed the same size
4 - box got larger
5 - box got much larger

It's a toy dataset that can help us validate that models are working at
least to some degree.
"""
import cairo
import csv
import numpy as np
import random
import sys
import time
import uuid
from math import sin, cos, radians, pi
from keras import utils

class MainRect():
    """Class that defines the main object in our generator."""

    def __init__(self):
        """Construct."""
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0
        self.color = (0.2, 0.2, 0.9)
        self.angle = 0  # (-0.01, 0.01)

    def set_random_pos(self):
        """Set a random position (used to start.)"""
        self.x = random.uniform(0, 1 - self.w)
        self.y = random.uniform(0, 1 - self.h)

    def set_pos(self, x, y):
        """Set the position of the rect."""
        self.x = x
        self.y = y

    def get_pos(self):
        """Get the position."""
        return self.x, self.y

    def set_size(self, w, h):
        """Set the width and height."""
        self.w = w
        self.h = h

    def get_size(self):
        """Get the size."""
        return self.w, self.h

    def set_color(self, new_color):
        """Set the color."""
        self.color = new_color

    def get_color(self):
        """Get the color."""
        return self.color

    def get_angle(self):
        """Get the current angle."""
        return self.angle

    def resize(self, amount):
        """Increase or decrease the size by some amount."""
        # Change size.
        self.w += amount
        self.h += amount

        # Change location.
        self.x -= amount / 2
        self.y -= amount / 2

    def move_random(self):
        """Teleport the rectangle randomly."""
        self.x = random.uniform(0, 1 - self.w)
        self.y = random.uniform(0, 1 - self.h)

    def move_linear(self, amount, theta):
        """Return new coordinates, given some direction."""
        theta_rad = pi/2 - radians(theta)
        self.x = self.x + amount * cos(theta_rad)
        self.y = self.y + amount * sin(theta_rad)

    def move_jitter(self, amount):
        """Move in a random direction by a random-ish amount.
        
        We want to move between +/- px_per_frame * 2 in each direction.
        """
        self.x = self.x + amount * random.uniform(-1, 1) * 4
        self.y = self.y + amount * random.uniform(-1, 1) * 4

    def set_angle(self, angle):
        """Rotate the object."""
        self.angle = angle


class SyntheticBoxes():
    """Class that defines a spatiotemporal dataset generator."""

    def __init__(self, batch_size, surface_width, surface_height,
                 nb_frames,
                 background_shapes=False,
                 jitter_move=False,
                 linear_move=False,
                 random_angle=False,
                 random_angle_per_frame=False,
                 random_background_color=False,
                 random_bg_per_frame=False,
                 random_fg_per_frame=False,
                 random_foreground_color=False,
                 random_move=False
        ):
        """Construct."""
        # Build the canvas.
        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, surface_width,
                                          surface_height)
        self.ctx = cairo.Context(self.surface)
        self.ctx.scale(surface_width, surface_height)

        # Set sequence and general params.
        self.batch_size = batch_size
        self.nb_frames = nb_frames
        self.px_per_frame = 0
        self.surface_width = surface_width
        self.surface_height = surface_height

        # Set more detailed params.
        self.background_shapes = background_shapes
        self.jitter_move = jitter_move
        self.linear_move = linear_move
        self.random_angle = random_angle
        self.random_angle_per_frame = random_angle_per_frame
        self.random_background_color = random_background_color
        self.random_bg_per_frame = random_bg_per_frame
        self.random_fg_per_frame = random_fg_per_frame
        self.random_foreground_color = random_foreground_color
        self.random_move = random_move

        # Init the main rect for later use.
        self.main_rect = None

        # Initialize background colors.
        self.bg_r1, self.bg_g1, self.bg_b1, self.bg_r2, self.bg_g2, \
            self.bg_b2 = (0.1, 0.8, 0.9, 0.9, 0.2, 0.1)

    def gen_random_colors(self, num):
        return tuple([random.random() for _ in range(num)])

    def gen_frame(self):
        """Create the elements of a single frame."""
        self.gen_background()
        self.gen_rect()

    def gen_background(self):
        """Build a background for the frame."""
        pat = cairo.LinearGradient(0.0, 0.0, 0.0, 1.0)

        pat.add_color_stop_rgba(1, self.bg_r1, self.bg_g1, self.bg_b1, 1)
        pat.add_color_stop_rgba(0, self.bg_r2, self.bg_g2, self.bg_b2, 1)
        self.ctx.rectangle(0, 0, 1, 1) # fill the whole screen.
        self.ctx.set_source(pat)
        self.ctx.fill()

        # Do we want shapes?
        if self.background_shapes:
            # Random circles!
            for _ in range(2):
                self.ctx.arc(
                    random.uniform(0, 1),
                    random.uniform(0, 1),
                    random.uniform(0, .5),
                    0,
                    2*3.14159)
                r, g, b = self.gen_random_colors(3)
                self.ctx.set_source_rgb(r, g, b)
                self.ctx.fill()

    def gen_rect(self):
        """Generate a rectangle. This is the star of the show."""
        # Get the current stats.
        x, y = self.main_rect.get_pos()
        w, h = self.main_rect.get_size()
        angle = self.main_rect.get_angle()

        # Draw it.
        self.ctx.save()
        self.ctx.set_source_rgb(*self.main_rect.get_color())
        self.ctx.rotate(angle)
        self.ctx.rectangle(x, y, w, h)
        self.ctx.set_line_width(0.03)
        self.ctx.fill()
        self.ctx.close_path()
        self.ctx.restore()

    def write_frame(self, ftype='png', fname='tmp.png'):
        """Utility to save out a single frame.

        Args:
            ftype (str): Type of file to save
            fname (str): Path to the output file
        """
        if ftype == 'png':
            self.surface.write_to_png(fname) # Output to PNG
        else:
            raise ValueError("Invalid file type, %d", ftype)

    def get_random_w_h(self):
        """Generate initial rectangle parameters."""
        # Width and height shouldn't be less than the amount they can shrink.
        w = max(random.random() / 2, self.nb_frames * abs(self.px_per_frame))
        h = max(random.random() / 2, self.nb_frames * abs(self.px_per_frame))

        return w, h

    def resize_rect(self, delta):
        """Return a resized rectangle."""
        self.main_rect.resize(delta)

    def get_sequence(self, delta):
        """Generate a sequence in a batch."""
        sequence_X = np.zeros((self.nb_frames, self.surface_height,
            self.surface_width, 3))

        # Set the change amount.
        self.px_per_frame = delta

        # Create a new main rect for this sequence.
        self.main_rect = MainRect()
        w, h = self.get_random_w_h()
        self.main_rect.set_size(w, h)
        self.main_rect.set_random_pos()

        # Set a theta at the sequence level.
        if self.linear_move:
            theta = random.randint(0, 360)

        # If we're only generating things per sequence, do them here.
        if self.random_background_color:
            # Generating background once.
            self.bg_r1, self.bg_rg1, self.bg_rb1, self.bg_rr2, self.bg_rg2, \
                self.bg_rb2 = self.gen_random_colors(6)

        if self.random_foreground_color:
            # Generate foreground once.
            self.main_rect.set_color(self.gen_random_colors(3))

        if self.random_angle:
            # Generate random starting angle.
            self.main_rect.set_angle(random.uniform(-0.01, 0.01))

        # Build our sequence.
        for i in range(self.nb_frames):
            # Generate a frame at current state.
            self.gen_frame()

            # Now add the new frame to our sequence as an array.
            surface_arr = np.frombuffer(self.surface.get_data(), np.uint8)
            surface_arr.shape = (self.surface_height, self.surface_width, 4)
            surface_arr = surface_arr[:,:,:3]  # remove alpha channel

            # Add it to the array.
            sequence_X[i] = surface_arr

            # Change color for the next frame.
            if self.random_background_color and self.random_bg_per_frame:
                self.bg_r1, self.bg_g1, self.bg_b1, self.bg_r2, self.bg_g2, \
                    self.bg_b2 = self.gen_random_colors(6)

            if self.random_foreground_color and self.random_fg_per_frame:
                self.main_rect.set_color(self.gen_random_colors(3))

            # Change angle?
            if self.random_angle and self.random_angle_per_frame:
                self.main_rect.set_angle(random.uniform(-0.01, 0.01))

            # Resize the rect for the next frame.
            self.resize_rect(delta)

            # Move it?
            if self.linear_move:
                self.main_rect.move_linear(self.px_per_frame * 2, theta)
            elif self.random_move:
                self.main_rect.move_random()
            elif self.jitter_move:
                self.main_rect.move_jitter(self.px_per_frame / 2)

        return sequence_X

    def delta_to_label(self, delta):
        """Given a change, bin it into one of N classes.

        Args:
            delta (float): Amount to change image size
        """
        upper_bounds = [
            -0.0084253935123,
            -0.00256442209132,
            0.00257423721249,
            0.0084253935123,
        ]
        bin_class = np.digitize(delta, upper_bounds)

        return bin_class

    def data_gen(self):
        """A generator that returns sequences of frames."""
        while True:
            batch_X = np.zeros((self.batch_size, self.nb_frames, self.surface_height,
            self.surface_width, 3))
            batch_y = np.zeros((self.batch_size, 5))

            for i in range(self.batch_size):
                # Sample from a normal distribution.
                delta = np.random.normal(0, .01)

                # Get the label vector corresponding to the delta.
                bin_class = self.delta_to_label(delta)
                label = utils.to_categorical(bin_class, 5)

                batch_X[i] = self.get_sequence(delta)
                batch_y[i] = label

            yield batch_X, batch_y

if __name__ == '__main__':
    #Just for testing.
    from PIL import Image
    toy = SyntheticBoxes(1, 80, 80, nb_frames=10)
    """
        background_shapes=False,
        jitter_move=False,
        linear_move=False,
        nb_frames=16,
        random_background_color=False,
        random_bg_per_frame=False,
        random_fg_per_frame=False,
        random_foreground_color=False,
        random_move=False,
        random_angle=False,
        random_angle_per_frame=False)
    """
    start = time.time()
    i = 0
    for x, y in toy.data_gen():
        print(x.shape, y.shape)

        # Checkout the first batch.
        for sequence in x:
            for i, image in enumerate(sequence):
                #image *= 255
                image = Image.fromarray(image.astype('uint8'))
                image.save('tmp/' + str(i) + '.png')
            print(y[0])
            sys.exit()

        break

