from PIL import ImageGrab
import numpy as np
import cv2
from skimage.io import imread
from skimage.transform import resize
from skimage.util import img_as_ubyte
from tensorflow.keras.models import load_model
import utils

SCREEN_RESOLUTION = (1920, 1080)
IMAGE_RESOLUTION = (128, 128)
width, height  = 128, 128
bbox = ((1920 // 2) - width , (1080 // 2) - height, (1920 // 2) + width , (1080 // 2) + height) 

model = load_model('model/colorizer')

def sliding_window(image):
	for y in range(0, image.shape[0], IMAGE_RESOLUTION[0]):
		for x in range(0, image.shape[1], IMAGE_RESOLUTION[0]):
			yield image[y:y + IMAGE_RESOLUTION[1], x:x + IMAGE_RESOLUTION[0]]

def stich_windows(windows):
    n = len(windows)
    ns = np.ceil(np.sqrt(n)).astype(int)
    windows_2d_list = []
    i=0
    for _ in range(ns):
        windows_h = []
        for _ in range(ns):
            windows_h.append(windows[i])
            i+=1
        windows_2d_list.append(windows_h)
    return cv2.vconcat([cv2.hconcat(h_windows) for h_windows in windows_2d_list])

while True:
    frame = ImageGrab.grab(bbox=bbox) 
    frame = np.array(frame)
    frame = img_as_ubyte(frame)
    
    colorized_windows = []
    for window in sliding_window(frame):
    
        image = resize(window, IMAGE_RESOLUTION)
        l, ab = utils.RGB2L_AB(image, IMAGE_RESOLUTION)
        predicted_ab = model.predict(np.expand_dims(l, axis=0))
        colorized_window = utils.L_AB2RGB(l, predicted_ab[0], IMAGE_RESOLUTION)
        colorized_window = img_as_ubyte(colorized_window)
        colorized_window = cv2.cvtColor(colorized_window, cv2.COLOR_RGB2BGR)
        colorized_windows.append(colorized_window)
    
    colorized_frame = stich_windows(colorized_windows)
    
    cv2.imshow("Input", cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
    cv2.imshow("Prediction", colorized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()