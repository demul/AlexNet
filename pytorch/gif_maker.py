import numpy as np
import cv2
import os
from array2gif import write_gif


def run(max_epoch):
    gif = np.empty((max_epoch, 800, 1600, 3), dtype=np.uint8)
    for i in range(max_epoch):
        gif[i] = cv2.imread(os.path.join('first_kernel_visualization', 'result%04d.png' % i))

    np.maximum(np.minimum(gif, 255), 0)
    write_gif(gif, 'first_kernel_visualization.gif', fps=2)
