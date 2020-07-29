import numpy as np
import sys
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator#preprocesar las imagenes 
from tensorflow.python.keras import optimizers#optimizador con el cual se entrena el algoritmo
from tensorflow.python.keras.models import Sequential#hacer redes neuronales secuenciales(capas ordenadas)
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation

from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D
#capas de convoluciones y pooling
from tensorflow.python.keras import backend as K#mata sesiones anteriores para comenzar entrenamiento
from sklearn.metrics import classification_report, confusion_matrix


K.clear_session()



data_entrenamiento = './data/entrenamiento'#directorios de imagenes de valida y entre
data_validacion = './data/validacion'

"""
Parametros
"""
epocas=1000#20 numero de iteraciones durante el entrenamiento---- 1
longitud, altura = 150, 150#(100,100)tamaño de las imagenes de entrada
batch_size = 32#32numero de imagenes a procesar en cada paso 32 ------3
pasos = 1000#(1000)numero de veces q se procesa la info en cada epoca---50 o 500
pasos_validacion = 300#(300)(validation_steps = 200) al finalizar cada epoca se corre 300 pasos con el set de validacion----2 o 20
filtrosConv1 = 32#32 numero de filtros por convolucion luego de la una convo------ 3
0                 #la imagen tendra una prof de 32
filtrosConv2 = 64#64lueo de la 2da convo una prof de 64    ---------   6
tamano_filtro1 = (3, 3)#tamaño filtro usado en convolucion(altura,longitud)
tamano_filtro2 = (2, 2)
tamano_pool = (2, 2)#tamaño de filtro para el maxpooling
clases = 4 #numero de clases en este caso 2 perro y gato
lr = 0.0004#ajustes de la red neuronal para acercarce a una solucion optima


##Pre procesamiento de imagenes

#datos de entrenamiento
entrenamiento_datagen = ImageDataGenerator(#reescalamiento de imagenes
    rescale=1. / 255,#rango normal 1 a 255, reescalado de 0 a  1
    shear_range=0.2,#inclina la imagen para k el algo aprendar a detectar en esa posicion
    zoom_range=0.2,#hace zoom para q aprenda puede se ma grande e incompleto
    horizontal_flip=True)#invertir imagen para q aprenda a reconocer inveridas 

#(validacion_datagen) datos de validacion

test_datagen = ImageDataGenerator(rescale=1. / 255)#no se modifican

#entra al directoria data/entrenamiento procesa a una alura y longitus especificada
entrenamiento_generador = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')#tipo de clasificacion categorica

#entra al directoria data/validacion procesa a una alura y longitus especificada
validacion_generador = test_datagen.flow_from_directory(
    data_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')#tipo de clasificacion categorica

#crear la red neuronal convolucional

cnn = Sequential()#tipo secuencial(varias capas apiladas)
#adicionar primera capa convilucion y pooling,input_shane recibe las imagenes 
cnn.add(Convolution2D(filtrosConv1, tamano_filtro1, padding ="same", input_shape=(longitud, altura, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))
#adicionar segunda capa convilucion y pooling
cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding ="same"))
cnn.add(MaxPooling2D(pool_size=tamano_pool))
#empezar clasificacion 
cnn.add(Flatten())#imagen procesada muy profunda y pequeña la volvemos plana(1 dimension q contiene toda la informacion)
#cnn.add(Dense(256, activation='relu'))
cnn.add(Dense(256, activation='relu'))#conexion con una capa de 256 neuronas y funcion de activacion relu 
cnn.add(Dropout(0.5))#a la capa dense durante el entrenamiento desactivamos la mitaad de las neuronas cada paso(evitar sobre ajuste)
                     #aprende caminos alternos para clasificar la informacion ya que las neuronas de desactivan aleatoriamente
cnn.add(Dense(clases, activation='softmax'))#sofmax devuelve las probabilidades q sea una clase

#parametros de optimizacion
cnn.compile(loss='categorical_crossentropy',#funcion perdida
            optimizer=optimizers.Adam(lr=lr),#optimizador lr=0.004
            metrics=['accuracy'])#porcentaje de que tan bien aprende

pasos

#Entrenar algoritmo
cnn.fit_generator(
    entrenamiento_generador,#imagnes de entrenamiento
    steps_per_epoch=pasos,#nro de pasos
    epochs=epocas,#numero epocas
    validation_data=validacion_generador,#imagenes de validacion
    validation_steps=pasos_validacion)#pasos validacion


#guardar modelo en un archivo

target_dir = './modelo/'
if not os.path.exists(target_dir):# si no existe crea
  os.mkdir(target_dir)
cnn.save('./modelo/modelo1.h5')#guarda modelo
cnn.save_weights('./modelo/pesos1.h5')#guarda pesos

#OBTENER MATRIZ DE CONFUSION Y OTRAS METRICAS
#====================================================
"""Y_pred = cnn.predict_generator(validacion_generador,1348)#batch_size
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print(Y_pred)
print('Confusion Matrix')
print(confusion_matrix(validacion_generador.classes, y_pred))
print('Classification Report')
target_names = ['Conejo','Gato','Loro','Perro',]
print(classification_report(validacion_generador.classes, y_pred, target_names=target_names))"""
#======================================================================

