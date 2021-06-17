from matplotlib import image
import numpy as np
from skimage.io import imread
from skimage.color import rgb2lab, lab2rgb
from skimage.transform import resize

def L_AB2RGB(l, ab, dim):
    image = np.empty(dim + (3,))
    image[:,:,0] = l.reshape(dim)
    image[:,:,1:] = ab * 128
    return lab2rgb(image)

def RGB2L_AB(rgb, dim):
    rgb = rgb.reshape(dim+(3,))
    # /255 not required as resize returns in [0,1] range
    # rgb = rgb / 255
    lab = rgb2lab(rgb)
    l = np.empty(dim+(1,))
    ab = np.empty(dim+(2,))
    l = np.expand_dims(lab[:,:,0], axis=2)
    ab = lab[:,:,1:] / 128
    return l, ab