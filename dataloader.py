import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.image import adjust_saturation
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2hsv, hsv2rgb
from skimage.util import img_as_ubyte
import utils

class SatelliteImagesGenerator(Sequence):

    def __init__(self, image_files, batch_size=32, dim=(128, 128), saturation_factor=0.5, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.image_files = image_files
        self.saturation_factor = saturation_factor
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.image_files) / self.batch_size)


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.image_files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[ index*self.batch_size : (index+1)*self.batch_size]
        # Find list of image_files
        image_files_temp = [self.image_files[i] for i in indexes]
        # Generate data
        X, y = self.__data_generation(image_files_temp)
        return X, y

    def __data_generation(self, image_files_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, 1))
        y = np.empty((self.batch_size, *self.dim, 2))

        # Generate data
        for i, file in enumerate(image_files_temp):
            # Read image
            image = imread(file)            
            hsv = rgb2hsv(image)
            hsv = img_as_ubyte(hsv)
            hsv[:,:,1] = hsv[:,:,1] * self.saturation_factor
            hsv[:,:,1] = np.clip(hsv[:,:,1],0,255)
            image = hsv2rgb(hsv.astype('uint8'))
            image = resize(image, self.dim)
            l, ab = utils.RGB2L_AB(image, self.dim)
            X[i,] = l
            y[i,] = ab
        return X, y