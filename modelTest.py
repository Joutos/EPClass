import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential

# from matplotlib import pyplot as plt

img = cv2.imread('./img_833_1074_543_502.png')
# plt.imshow(img)
# plt.show()
resize = tf.image.resize(img, (256, 256))
new_model = load_model('./models/imageclassifier2.h5')
yhat = new_model.predict(np.expand_dims(resize/255, 0))

if yhat > 0.5:
    print(f'Predicted class is Errada')
else:
    print(f'Predicted class is Certa')
