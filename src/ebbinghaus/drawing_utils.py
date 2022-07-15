from functools import partial
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class DrawShape():
    def __init__(self, background='black', img_size=(224, 224), width=None, antialiasing=False, borders=False, line_col=None, resize_up_resize_down=False, borders_width=None):
        self.resize_up_resize_down = resize_up_resize_down
        self.antialiasing = antialiasing
        self.borders = borders
        self.background = background
        self.img_size = np.array(img_size)
        if width is None:
            width = img_size[0] * 0.022

        # random means random pixels,
        # random uni means random uniform color every creation
        if background == 'random':
            self.background_type = 'rnd-pixels'
            self.line_col = 0 if line_col is None else line_col

        elif background == 'black':
            self.background_type = 'white-on-black'
            self.line_col = 255 if line_col is None else line_col

        elif background == 'white':
            self.background_type = 'black-on-white'
            self.line_col = 0 if line_col is None else line_col

        elif background == 'random_uni':
            self.background_type = 'random_uniform'
            self.line_col = 255 if line_col is None else line_col

        self.fill = (*[self.line_col] * 3, 255)
        self.line_args = dict(fill=self.fill, width=width, joint='curve')
        if borders:
            self.borders_width = self.line_args['width'] if borders_width is None else borders_width
        else:
            self.borders_width = 0

    def create_canvas(self, img=None, borders=None):
        if img is None:
            img = self.img_size

        if self.background == 'random':
            img = Image.fromarray(np.random.randint(0, 255, (img[1], img[0], 3)).astype(np.uint8), mode='RGB')
        elif self.background == 'black':
            img = Image.new('RGB', tuple(img), 'black')  # x and y
        elif self.background == 'white':
            img = Image.new('RGB', tuple(img), 'white')  # x and y
        elif self.background == 'random_uni':
            img = Image.new('RGB', tuple(img), (np.random.randint(256), np.random.randint(256), np.random.randint(256)))  # x and y


        borders = self.borders if borders is None else borders
        if borders:
            draw = ImageDraw.Draw(img)
            draw.line([(0, 0), (0, img.size[0])], fill=self.line_args['fill'], width=self.borders_width)
            draw.line([(0, 0), (img.size[0], 0)], fill=self.line_args['fill'], width=self.borders_width)
            draw.line([(img.size[0] - 1, 0), (img.size[0] - 1, img.size[1] - 1)], fill=self.line_args['fill'], width=self.borders_width)
            draw.line([(0, img.size[0] - 1), (img.size[0] - 1, img.size[1] - 1)], fill=self.line_args['fill'], width=self.borders_width)

        return img

    def resize_up_down(fun):
        def wrap(self, *args, **kwargs):
            if self.resize_up_resize_down:
                ori_self_width = self.line_args['width']
                self.line_args['width'] = self.line_args['width'] * 2
                original_img_size = self.img_size
                self.img_size = np.array(self.img_size) * 2

            im1 = fun(self, *args, **kwargs)

            if self.resize_up_resize_down:
                im1 = im1.resize(original_img_size)
                self.line_args['width'] = ori_self_width
                self.img_size = original_img_size
            return im1

        return wrap

    def circle(self, draw, center, radius, fill=None):
        if not fill:
            fill = self.fill
        draw.ellipse((center[0] - radius + 1,
                      center[1] - radius + 1,
                      center[0] + radius - 1,
                      center[1] + radius - 1), fill=fill, outline=None)

    # @resize_up_down
    def create_ebbinghaus(self, r_c, d=0, r2=0, n=0, shift=0, colour_center_circle=(255, 255, 255), img=None):
        """
        Parameters r_c, d, and r2, are relative to the total image size.
        If you only want to generate the center circle, leave d to 0.
        r_c : radius of the center circle
        d : distance to flankers
        r2 : radius flankers
        n : 0 number flankers
        shift : flankers rotations around center circle [0, pi]
        """
        if img is None:
            img = self.create_canvas()
        draw = ImageDraw.Draw(img)
        if d != 0:
            thetas = np.linspace(0, np.pi*2, n, endpoint=False) + shift
            dd = img.size[0]*d
            vect = [[np.cos(t)*dd, np.sin(t)*dd] for t in thetas]
            [self.circle(draw, np.array(vv) + img.size[0]/2, img.size[0]*r2) for vv in vect]
        # img = self.apply_antialiasing(img)
        self.circle(draw, np.array(img.size)/2, img.size[0]*r_c, fill=colour_center_circle)

        return img

    def create_random_ebbinghaus(self, r_c, n=0, flankers_size_range=(0.1, 0.5), colour_center_circle=(255, 255, 255)):
        gen_rnd = lambda r: np.random.uniform(*r)

        img = self.create_canvas()
        draw = ImageDraw.Draw(img)

        for i in range(n):
            random_points = [np.random.randint(self.img_size[0]), np.random.randint(self.img_size[1])]
            random_size = self.img_size[0]*gen_rnd(flankers_size_range)
            self.circle(draw, np.array(random_points), random_size)
        self.circle(draw, self.img_size/2, self.img_size[0]*r_c, fill=colour_center_circle)

        return img

    def apply_antialiasing(self, im):
        im = im.resize((im.size[0] * 2, im.size[1] * 2))
        im = im.resize((im.size[0] // 2, im.size[1] // 2), resample=Image.ANTIALIAS)
        return im


