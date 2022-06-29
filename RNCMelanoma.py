"""
DEEP LEARNING CLASIFICADOR MELANOMAS
CARLOS MIR MARTÍNEZ
"""
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator # Preprocesa las images y rescala
import warnings
import cv2
import os
warnings.filterwarnings('ignore')

# Visualizacion
labels = ['NotMelanoma','Melanoma']
img_size = 150

def get_data(data_dir):
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)

train = get_data('C:/Users/USUARIO/Desktop/FotosPyQT6/train_sep')
test = get_data('C:/Users/USUARIO/Desktop/FotosPyQT6/test')
"""
### 
plt.figure(figsize = (5,5))
plt.imshow(train[0][0], cmap='gray')
plt.title(labels[train[0][1]]) # 0 imagen pneumonia

### 
plt.figure(figsize = (5,5))
plt.imshow(train[-1][0], cmap='gray')
plt.title(labels[train[-1][1]]) # -1 imagen normal
"""

# Preprocesado
# Rescala las imagenes del Train
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
# Rescala las imagenes del test
test_datagen = ImageDataGenerator(rescale = 1./255)

# Creando el DF Training SET
training_set = train_datagen.flow_from_directory('C:/Users/USUARIO/Desktop/FotosPyQT6/train_sep',
                                                 target_size = (64, 64),
                                                 class_mode = 'binary')

# Creando el DF Test SET
test_set = test_datagen.flow_from_directory('C:/Users/USUARIO/Desktop/FotosPyQT6/test',
                                            target_size = (64, 64),
                                            class_mode = 'binary')

# Creamos la red RNC, Convolucion --> Pooling --> Flattenin --> Full Connect
RNC = tf.keras.models.Sequential()
# 1º Capa Convolucion2D, entrada de datos
RNC.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[64, 64, 3]))
# 2º Capa - Pooling, Simplifica los problemas y reduce las operaciones
RNC.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
# 3º Capa de Convolucion y Pooling
RNC.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
RNC.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
# 4º Capa - Flattening, adapta la estructura de forma vertical en una columna
RNC.add(tf.keras.layers.Flatten())
# Full Connection, añadimos la red neuronal totalmentne conectada
RNC.add(tf.keras.layers.Dense(units=128, activation='relu'))
# Capa de Salida
RNC.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) # Funcion sigmoide

# Compilamos el modelos con el optimizador Adam y entropia cruzada binaria
RNC.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Entrenamos el modelo
RNC.fit_generator(training_set,
                  steps_per_epoch = 100,
                  epochs = 20,
                  validation_data = test_set
                  ) 

#RNC.save('C:/Users/USUARIO/Desktop/FotosPyQT6/ClasificadorMelanomaRNC02.h5')
"""
Epoch 99/100
100/100 [==============================] - 70s 704ms/step - loss: 0.2005 - accuracy: 0.9189 - val_loss: 0.2550 - val_accuracy: 0.8950
Epoch 100/100
100/100 [==============================] - 69s 696ms/step - loss: 0.1837 - accuracy: 0.9274 - val_loss: 0.2696 - val_accuracy: 0.8863
RNC.save('C:/Users/USUARIO/Desktop/FotosPyQT6/ClasificadorMelanomaRNC.h5')
20/20 [==============================] - 46s 2s/step - loss: 0.4868 - accuracy: 0.7875 - val_loss: 0.4366 - val_accuracy: 0.8060
"""