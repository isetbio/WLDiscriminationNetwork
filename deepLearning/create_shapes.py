import cv2 as cv
import numpy as np
from PIL import Image
import os

black = np.full((64,64), 0.)
# cv.imshow('white image', white)
out_dir = '/share/wandell/data/reith/matlabData/circles/'
# create logspace of radians, use of 128.001, as with 128, the function creates an array of 20 logarithmically spaced
# numbers
radii = np.unique(np.geomspace(1, 32.001, 33).astype(np.int))
for rad in radii:
    painting = cv.circle(black, center=(32,32), radius=rad, color=255, thickness=-1)
    img = Image.fromarray(painting)
    out_name = os.path.join(out_dir, f"white_circle_rad_{rad}.bmp")
    img.convert("L").save(out_name)
print('nice')
