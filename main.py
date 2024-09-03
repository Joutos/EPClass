import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

data_dir = 'data'

# Utilize o ImageDataGenerator para carregar e pré-processar as imagens
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.3  # Porcentagem para validação
)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary',
    subset='training'  # Para o conjunto de treinamento
)

val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary',
    subset='validation'  # Para o conjunto de validação
)

# Criação do modelo
model = Sequential()
model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(32, (3, 3), 1, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(16, (3, 3), 1, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))  # Corrigindo o erro, incluindo model.add
model.add(Dense(1, activation='sigmoid'))

model.compile('adam', loss=tf.losses.BinaryCrossentropy(),
              metrics=['accuracy'])
model.summary()

# Callback para TensorBoard
logdir = 'logs'
tensorboard_callback = TensorBoard(log_dir=logdir)

# Treinamento do modelo
hist = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=[tensorboard_callback]
)

# Plotando métricas de treinamento
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

# Avaliação do modelo
loss, accuracy = model.evaluate(val_generator)
print(f'Validation accuracy: {accuracy}')

# Exemplo de predição com uma imagem específica
img_path = './1_imagem_5.png'
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(256, 256))
plt.imshow(img)
plt.show()

img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Adiciona dimensão do batch
img_array /= 255.0  # Normalização
prediction = model.predict(img_array)

if prediction > 0.5:
    print(f'Predicted class: Errada')
else:
    print(f'Predicted class: Certa')

# Salvando o modelo
model.save(os.path.join('models', 'imageclassifierTraços2500.keras'))
