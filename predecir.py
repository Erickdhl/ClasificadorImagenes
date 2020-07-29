import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from tensorflow.python.keras import layers, models 

import os #como buscar listar los archivos de un directorio
import tkinter
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk

longitud, altura = 150, 150
modelo = './modelo/modelo2.h5'
pesos_modelo = './modelo/pesos2.h5'
cnn = load_model(modelo)#cargar modelo
cnn.load_weights(pesos_modelo)#cargar pesos

def predict(file):
  x = load_img(file, target_size=(longitud, altura))#cargar imagen
  x = img_to_array(x)#convertir arrglo la imagen 
  x = np.expand_dims(x, axis=0)#agregar dimension extra para procesar informa
  array = cnn.predict(x)#llamamo a predecir [[1=prediccion correcta,0,0] (tiene 2 dimesions)
  result = array[0]#solo trae una dimension q es la q nos interesa [1,0,0]
  answer = np.argmax(result)#0 posicionde resultado correcto (1 en pos 0) 
  if answer == 0:
    return("se predice que es un Conejo")
  elif answer == 1:
    return("se predice que es un Gato")
  elif answer == 2:
    return("se predice que es un Loro")
  elif answer == 3:
    return("se predice que es un Perro")

  return answer

#=====================================================
#predict('cat.4063.jpg')
#predict('cat.4019.jpg')
#predict('cat.4052.jpg')
#predict('cat.4028.jpg')
#predict('dog.2.jpg')
#predict('dog.4011.jpg')
#predict('dog.4012.jpg')


#===================================================
   
def cargar_imagen():
  nombre = filedialog.askopenfilename()

  """imagen = ImageTk.PhotoImage(Image.open(nombre))
  contenedor.image = imagen
  contenedor.create_image(50, 50, image=imagen, anchor=NW)
  contenedor.place(relx=0.05, rely=0.25)"""
  imagen = Image.open(nombre)
  r = imagen.resize((300, 300))
  r.save("r.jpg")
  imagen = ImageTk.PhotoImage(Image.open("r.jpg"))
  contenedor.image = imagen
  contenedor.create_image(50, 50, image=imagen, anchor=NW)
  contenedor.place(relx=0.05, rely=0.25)
  #predict(nombre)
  txbDireccion.delete(0, END)
  pred=predict(nombre)
  txbDireccion.insert(0, pred)
  #try:
   # imagen = Image.open(nombre)
  #except:
   # messagebox.showinfo("Error", "La imagen no puede ser leida")
	

cuadro = tkinter.Tk()
cuadro.title("Deteccion Mascotas")
cuadro.geometry('700x600')

txtTitulo = Label(cuadro,text='DETECCION DE MASCOTAS',font=("calibri", 16))
txtTitulo.place(relx=0.3, rely=0.028)



txbDireccion = Entry(cuadro, width=45)
txbDireccion.place(relx=0.25, rely=0.116)


btnImagen = tkinter.Button(cuadro, text = "Seleccionar Imagen", command=cargar_imagen)
btnImagen.place(relx=0.25, rely=0.18)


#btnRed = tkinter.Button(cuadro, text = "Predecir", command=Predecir())
#btnRed.place(relx=0.05, rely=0.18)

contenedor = Canvas(cuadro,width=400,height=400)
contenedor.place(relx=0.1, rely=0.25)

txtPredicciones = Label(cuadro,text='',font=("calibri", 14))
txtPredicciones.place(relx=0.65, rely=0.43)

cuadro.mainloop()
