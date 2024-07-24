import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential

# from matplotlib import pyplot as plt

for image in os.listdir('./70148-025 - FECHADURAS E MANIPULOS PORTAS POSTERIORES PARA VEICULOS COMERCIAIS/PROCESSADAS'):
    img = cv2.imread('./70148-025 - FECHADURAS E MANIPULOS PORTAS POSTERIORES PARA VEICULOS COMERCIAIS/PROCESSADAS/' + image)
    resize = tf.image.resize(img, (256, 256))
    new_model = load_model('./models/imageclassifierTraços.h5')
    yhat = new_model.predict(np.expand_dims(resize/255, 0))
    
    if yhat > 0.5:
        print(f'Predicted class is Errada')
        os.remove('./70148-025 - FECHADURAS E MANIPULOS PORTAS POSTERIORES PARA VEICULOS COMERCIAIS/PROCESSADAS/' + image)
    else:
        print(f'Predicted class is Certa')
