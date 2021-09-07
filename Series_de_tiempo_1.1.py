# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 15:41:43 2020
PROGRAMA SMART UNIFICADO
@author: Carlos Quinchia
"""

########## LIBRERÍAS A UTILIZAR ##########
#Se importan la librerias a utilizar
# import
import os
import shutil

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
#import statsmodels.formula.api as sfm
#import seaborn as sns 
import time
import math
#import pandas_datareader as web
#----------------------------------------------------------------------------------------------------------

# from
from os import scandir
from os import remove
from os.path import isfile, isdir

from matplotlib.font_manager import FontProperties
#from statsmodels.sandbox.regression.predstd import wls_prediction_std
from scipy import stats 

from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
from sklearn.linear_model import LinearRegression, Ridge 
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

#Manejo de fuentes-------------------------------------------------------------
font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font.set_style('italic')


#PROGRAMA 05 REGRESION SERIE DE TIEMPO VENTAS ===============================================================================
#import P5
"""
Calcula el pronostico de la serie de ventas en el tiempo del ingreso genera las graficas comparativas
"""

plt.style.use('fivethirtyeight')

"""
Obtendré la cotización de las acciones de la empresa 'Apple Inc.' utilizando el 
ticker bursátil de las empresas (AAPL) desde el 1 de enero de 2012 hasta el 30 
de septiembre de 2020.
"""

#Obtenga la cotización bursátil 
#df = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2020-09-30') 
 #Muestre los datos 

file= "D:\Mis documentos\Escritorio\PYLAB\Quote.csv"
#df = pd.read_csv(file, start='2015-07-30', end='2021-04-28')
#df = pd.read_csv(file,data_source=[[0]],start='2015-07-30', end='2021-04-28') 
df = pd.read_csv(file,header=0) 
#print(df)

#x=df[df.columns[-1]]
#print(x)

#y=df[df.columns[-0]]
#print(y)

"""
A continuación, mostraré el número de filas y columnas en el conjunto de datos. 
El resultado muestra que tenemos 2003 filas o días en que se registró el precio 
de las acciones y 6 columnas.
"""
#Dias de predicción (Dp)
Dp=120

df.shape

#Crear el grafico de visualización de datos

#Visualize the closing price history
plt.figure(figsize=(16,8))
plt.title('Histórico de ventas')
plt.plot(df['Close'])
plt.xlabel('fecha',fontsize=18)
plt.ylabel('Valor de las ventas diarias ($)',fontsize=18)
#plt.show()
plt.savefig("D:\Mis documentos\Escritorio\PYLAB\Sales.jpg")
#plt.savefig("Sales.jpg")

"""
Cree un nuevo marco de datos con solo el precio de cierre y conviértalo en una 
matriz. Luego, cree una variable para almacenar la longitud del conjunto de 
datos de entrenamiento. Quiero que el conjunto de datos de entrenamiento contenga 
aproximadamente el 80% de los datos.
"""

#Cree un nuevo marco de datos con solo la columna 'Cerrar'
data = df.filter (['Close'])
#Convertir el marco de datos a un conjunto de
dataset = data.values
#Obtener / Calcular el número de filas para entrenar el modelo en
training_data_len = math.ceil( len(dataset) *.8) 

"""
Ahora escale el conjunto de datos para que tenga valores entre 0 y 1 inclusive, 
lo hago porque generalmente es una buena práctica escalar sus datos antes de 
entregarlos a la red neuronal.
"""

#Escala todos los datos para que sean valores entre 0 y 1 
scaler = MinMaxScaler(feature_range=(0, 1)) 
scaled_data = scaler.fit_transform(dataset)

"""
Cree un conjunto de datos de entrenamiento que contenga los valores del precio 
de cierre de los últimos 60 días que queremos usar para predecir el valor del 
precio de cierre número 61.

Por tanto, la primera columna del conjunto de datos ' x_train ' contendrá valores
 del conjunto de datos del índice 0 al índice 59 (60 valores en total) y la 
 segunda columna contendrá valores del conjunto de datos del índice 1 al índice 
 60 (60 valores) y así sucesivamente y así sucesivamente.
 
El conjunto de datos ' y_train ' contendrá el valor 61 ubicado en el índice 60 
para su primera columna y el valor 62 ubicado en el índice 61 del conjunto de 
datos para su segundo valor y así sucesivamente.
"""

#Crea el conjunto de datos de entrenamiento escalados
train_data = scaled_data[0:training_data_len  , : ]
#Dividir los datos en conjuntos de datos x_train y y_train
x_train=[]
y_train = []
for i in range(Dp,len(train_data)):
    x_train.append(train_data[i-Dp:i,0])
    y_train.append(train_data[i,0])


"""
Ahora convierta el conjunto de datos de tren independiente ' x_train ' y el 
conjunto de datos de tren dependiente ' y_train ' en matrices numpy para que 
puedan usarse para entrenar el modelo LSTM.
"""

#Convertir x_train y y_train en matrices
x_train, y_train = np.array(x_train), np.array(y_train)

"""
Cambie la forma de los datos para que sean tridimensionales en la forma 
[número de muestras , número de pasos de tiempo y número de características ]. 
El modelo LSTM espera un conjunto de datos tridimensionales.
"""

# Cambie la forma de los datos a la forma aceptada por LSTM
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

"""
Construya el modelo LSTM para tener dos capas LSTM con 50 neuronas y dos 
capas densas, una con 25 neuronas y la otra con 1 neurona.
"""

#Construir el modelo de red LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

"""
Compile el modelo usando la función de pérdida de error cuadrático medio (MSE) 
y el optimizador de Adam.
"""

#Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

"""
Entrene el modelo con los conjuntos de datos de entrenamiento. Tenga en cuenta 
que el ajuste es otro nombre para tren. El tamaño del lote es el número total 
de ejemplos de entrenamiento presentes en un solo lote, y la época es el número 
de iteraciones cuando un conjunto de datos completo se pasa hacia adelante y hacia 
atrás a través de la red neuronal.
"""

#Entrenar el modelo
model.fit(x_train, y_train, batch_size=1, epochs=1)



#Crear un conjunto de datos de prueba.- Test data set
test_data = scaled_data[training_data_len - Dp: , : ]

#Cree los conjuntos de datos x_test y y_test
x_test = []
y_test =  dataset[training_data_len : , : ] #Get all of the rows from index 1603 to the rest and all 
#of the columns (in this case it's only column 'Close'), so 2003 - 1603 = 400 rows of data
for i in range(Dp,len(test_data)):
    x_test.append(test_data[i-Dp:i,0])

"""
Luego, convierta el conjunto de datos de prueba independiente ' x_test ' en una 
matriz numérica para que pueda usarse para probar el modelo LSTM.
"""
#Convertir x_test a una matriz
x_test = np.array(x_test)

"""
Cambie la forma de los datos para que sean tridimensionales en la forma 
[número de muestras , número de pasos de tiempo y número de características ]. 
Esto debe hacerse, porque el modelo LSTM espera un conjunto de datos 
tridimensionales.
"""
# Cambie la forma de los datos a la forma aceptada por LSTM
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))


"""
Ahora obtenga los valores predichos del modelo usando los datos de prueba.
"""

#Obtener los modelos de valores de precio predichos
predictions = model.predict(x_test) 
predictions = scaler.inverse_transform(predictions)#Undo scaling

#Realizo una predicción---------------------------------------------------------------
#print(predictions)
#Convertir x_test a una matriz
y_pred = np.array(predictions)
#df = pd.DataFrame(dict(y_pred=y_pred))
#print(y_pred)
df=pd.DataFrame(y_pred)
df.to_csv('D:\Mis documentos\Escritorio\PYLAB\param3.csv',index=False)
#------------------------------------------------------------------------------------

"""
Obtenga la raíz del error cuadrático medio (RMSE), que es una buena medida de 
la precisión del modelo. Un valor de 0 indicaría que los valores predichos de 
los modelos coinciden perfectamente con los valores reales del conjunto de datos 
de prueba.

Cuanto menor sea el valor, mejor se desempeñó el modelo. Pero, por lo general, 
es mejor utilizar también otras métricas para tener una idea real del rendimiento 
del modelo.
"""

# Calcular / Obtener el valor de RMSE
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
#print(rmse)

#Tracemos y visualicemos los datos.
#Plot/Create the data for the graph
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Modelo')
plt.xlabel('fecha', fontsize=18)
plt.ylabel('Valor de las ventas diarias ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Values', 'Predictions'], loc='lower right')
#plt.show()
plt.savefig("D:\Mis documentos\Escritorio\PYLAB\Sales_adj.jpg")
#plt.savefig("Sales_adj.jpg")


#Muestre los precios válidos y previstos.
#print(valid)

"""
Quiero probar el modelo un poco más y obtener el valor del precio de cierre 
previsto de Apple Inc. para el 1 de octubre de 2020 (01/10/2020).

Entonces obtendré la cotización, convertiré los datos en una matriz que contiene 
solo el precio de cierre. Luego obtendré el precio de cierre de los últimos 60 
días y escalaré los datos para que sean valores entre 0 y 1 inclusive.

Después de eso, crearé una lista vacía y le agregaré el precio de los últimos 
60 días, y luego lo convertiré en una matriz numpy y le daré una nueva forma para 
poder ingresar los datos en el modelo.

Por último, pero no menos importante, ingresaré los datos en el modelo y obtendré 
el precio previsto.
"""

#Obtenga el Quote
#apple_quote = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', 
#                             end='2020-09-30')
apple_quote = pd.read_csv(file,header=0)

#Cree un nuevo marco de datos
new_df = apple_quote.filter(['Close'])

#Obtener el precio de cierre de los últimos 120 días
last_120_days = new_df[-Dp:].values

#Escale los datos para que sean valores entre 0 y 1
last_120_days_scaled = scaler.transform(last_120_days)

#Crear una lista vacía
X_test = []

#Añadir los últimos 120 días
X_test.append(last_120_days_scaled)

#Convierta el conjunto de datos X_test en una matriz
X_test = np.array(X_test)

#Redimensione los datos
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#Obtenga el precio escalado previsto
pred_price = model.predict(X_test)

#hacer el escalado
pred_price = scaler.inverse_transform(pred_price)
print('Ventas proyectadas ', pred_price)

"""
Ahora veamos cuál fue el precio real de ese día.
"""
#Obtenga el Quote
df2 =  pd.read_csv(file,header=0)
print(df2[-1:])

#FIN PROGRAMA 05 REGRESION SERIE DE TIEMPO VENTAS  ==========================================================================

#PROGRAMA 06 REGRESION SERIE DE TIEMPO MARGEN ===============================================================================
#import P6
"""
Calcula el pronostico de la serie de Margen en el tiempo del ingreso genera las graficas comparativas

Descripción: Este programa utiliza una red neuronal recurrente artificial llamada 
Long Short Term Memory (LSTM) para predecir el precio de cierre de las acciones de una 
corporación (Apple Inc.) utilizando el precio de las acciones de los últimos 60 días.
A continuación, cargaré / importaré las bibliotecas que se utilizarán en este programa.
"""
#Manejo de fuentes
font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font.set_style('italic')

plt.style.use('fivethirtyeight')

"""
Obtendré la cotización de las acciones de la empresa 'Apple Inc.' utilizando el 
ticker bursátil de las empresas (AAPL) desde el 1 de enero de 2012 hasta el 30 
de septiembre de 2020.
"""

#Obtenga la cotización bursátil 
#df = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2020-09-30') 
 #Muestre los datos 

file= "D:\Mis documentos\Escritorio\PYLAB\Quote1.csv"
#df = pd.read_csv(file, start='2015-07-30', end='2021-04-28')
#df = pd.read_csv(file,data_source=[[0]],start='2015-07-30', end='2021-04-28') 
df = pd.read_csv(file,header=0) 
#print(df)

#x=df[df.columns[-1]]
#print(x)

#y=df[df.columns[-0]]
#print(y)

"""
A continuación, mostraré el número de filas y columnas en el conjunto de datos. 
El resultado muestra que tenemos 2003 filas o días en que se registró el precio 
de las acciones y 6 columnas.
"""
#Dias de predicción (Dp)
Dp=120

df.shape

#Crear el grafico de visualización de datos

#Visualize the closing price history
plt.figure(figsize=(16,8))
plt.title('Histórico de Margen')
plt.plot(df['Close'])
plt.xlabel('fecha',fontsize=18)
plt.ylabel('Valor del Margen diario (%)',fontsize=18)
#plt.show()
plt.savefig("D:\Mis documentos\Escritorio\PYLAB\Margin.jpg")
#plt.savefig("Margin.jpg")

"""
Cree un nuevo marco de datos con solo el precio de cierre y conviértalo en una 
matriz. Luego, cree una variable para almacenar la longitud del conjunto de 
datos de entrenamiento. Quiero que el conjunto de datos de entrenamiento contenga 
aproximadamente el 80% de los datos.
"""

#Cree un nuevo marco de datos con solo la columna 'Cerrar'
data = df.filter (['Close'])
#Convertir el marco de datos a un conjunto de
dataset = data.values
#Obtener / Calcular el número de filas para entrenar el modelo en
training_data_len = math.ceil( len(dataset) *.8) 

"""
Ahora escale el conjunto de datos para que tenga valores entre 0 y 1 inclusive, 
lo hago porque generalmente es una buena práctica escalar sus datos antes de 
entregarlos a la red neuronal.
"""

#Escala todos los datos para que sean valores entre 0 y 1 
scaler = MinMaxScaler(feature_range=(0, 1)) 
scaled_data = scaler.fit_transform(dataset)

"""
Cree un conjunto de datos de entrenamiento que contenga los valores del precio 
de cierre de los últimos 60 días que queremos usar para predecir el valor del 
precio de cierre número 61.

Por tanto, la primera columna del conjunto de datos ' x_train ' contendrá valores
 del conjunto de datos del índice 0 al índice 59 (60 valores en total) y la 
 segunda columna contendrá valores del conjunto de datos del índice 1 al índice 
 60 (60 valores) y así sucesivamente y así sucesivamente.
 
El conjunto de datos ' y_train ' contendrá el valor 61 ubicado en el índice 60 
para su primera columna y el valor 62 ubicado en el índice 61 del conjunto de 
datos para su segundo valor y así sucesivamente.
"""

#Crea el conjunto de datos de entrenamiento escalados
train_data = scaled_data[0:training_data_len  , : ]
#Dividir los datos en conjuntos de datos x_train y y_train
x_train=[]
y_train = []
for i in range(Dp,len(train_data)):
    x_train.append(train_data[i-Dp:i,0])
    y_train.append(train_data[i,0])


"""
Ahora convierta el conjunto de datos de tren independiente ' x_train ' y el 
conjunto de datos de tren dependiente ' y_train ' en matrices numpy para que 
puedan usarse para entrenar el modelo LSTM.
"""

#Convertir x_train y y_train en matrices
x_train, y_train = np.array(x_train), np.array(y_train)

"""
Cambie la forma de los datos para que sean tridimensionales en la forma 
[número de muestras , número de pasos de tiempo y número de características ]. 
El modelo LSTM espera un conjunto de datos tridimensionales.
"""

# Cambie la forma de los datos a la forma aceptada por LSTM
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

"""
Construya el modelo LSTM para tener dos capas LSTM con 50 neuronas y dos 
capas densas, una con 25 neuronas y la otra con 1 neurona.
"""

#Construir el modelo de red LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

"""
Compile el modelo usando la función de pérdida de error cuadrático medio (MSE) 
y el optimizador de Adam.
"""

#Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

"""
Entrene el modelo con los conjuntos de datos de entrenamiento. Tenga en cuenta 
que el ajuste es otro nombre para tren. El tamaño del lote es el número total 
de ejemplos de entrenamiento presentes en un solo lote, y la época es el número 
de iteraciones cuando un conjunto de datos completo se pasa hacia adelante y hacia 
atrás a través de la red neuronal.
"""

#Entrenar el modelo
model.fit(x_train, y_train, batch_size=1, epochs=1)

#Crear un conjunto de datos de prueba.- Test data set
test_data = scaled_data[training_data_len - Dp: , : ]

#Cree los conjuntos de datos x_test y y_test
x_test = []
y_test =  dataset[training_data_len : , : ] #Get all of the rows from index 1603 to the rest and all of the columns 
#(in this case it's only column 'Close'), so 2003 - 1603 = 400 rows of data
for i in range(Dp,len(test_data)):
    x_test.append(test_data[i-Dp:i,0])


"""
Luego, convierta el conjunto de datos de prueba independiente ' x_test ' en una 
matriz numérica para que pueda usarse para probar el modelo LSTM.
"""
#Convertir x_test a una matriz
x_test = np.array(x_test)

"""
Cambie la forma de los datos para que sean tridimensionales en la forma 
[número de muestras , número de pasos de tiempo y número de características ]. 
Esto debe hacerse, porque el modelo LSTM espera un conjunto de datos 
tridimensionales.
"""
# Cambie la forma de los datos a la forma aceptada por LSTM
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))


"""
Ahora obtenga los valores predichos del modelo usando los datos de prueba.
"""

#Obtener los modelos de valores de precio predichos
predictions = model.predict(x_test) 
predictions = scaler.inverse_transform(predictions)#Undo scaling

#Realizo una predicción---------------------------------------------------------------
#print(predictions)
#Convertir x_test a una matriz
y_pred = np.array(predictions)
#df = pd.DataFrame(dict(y_pred=y_pred))
#print(y_pred)
df=pd.DataFrame(y_pred)
df.to_csv('D:\Mis documentos\Escritorio\PYLAB\param4.csv',index=False)
#-------------------------------------------------------------------------------------

"""
Obtenga la raíz del error cuadrático medio (RMSE), que es una buena medida de 
la precisión del modelo. Un valor de 0 indicaría que los valores predichos de 
los modelos coinciden perfectamente con los valores reales del conjunto de datos 
de prueba.

Cuanto menor sea el valor, mejor se desempeñó el modelo. Pero, por lo general, 
es mejor utilizar también otras métricas para tener una idea real del rendimiento 
del modelo.
"""

# Calcular / Obtener el valor de RMSE
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
#print(rmse)

#Tracemos y visualicemos los datos.
#Plot/Create the data for the graph
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Modelo')
plt.xlabel('fecha', fontsize=18)
plt.ylabel('Valor del Margen diario (%)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Values', 'Predictions'], loc='lower right')
#plt.show()
plt.savefig("D:\Mis documentos\Escritorio\PYLAB\Margin_adj.jpg")
#plt.savefig("Margin_adj.jpg")

#Muestre los precios válidos y previstos.
#print(valid)

"""
Quiero probar el modelo un poco más y obtener el valor del precio de cierre 
previsto de Apple Inc. para el 1 de octubre de 2020 (01/10/2020).

Entonces obtendré la cotización, convertiré los datos en una matriz que contiene 
solo el precio de cierre. Luego obtendré el precio de cierre de los últimos 60 
días y escalaré los datos para que sean valores entre 0 y 1 inclusive.

Después de eso, crearé una lista vacía y le agregaré el precio de los últimos 
60 días, y luego lo convertiré en una matriz numpy y le daré una nueva forma para 
poder ingresar los datos en el modelo.

Por último, pero no menos importante, ingresaré los datos en el modelo y obtendré 
el precio previsto.
"""

#Obtenga el Quote
#apple_quote = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', 
#                             end='2020-09-30')
apple_quote = pd.read_csv(file,header=0)

#Cree un nuevo marco de datos
new_df = apple_quote.filter(['Close'])

#Obtener el precio de cierre de los últimos 120 días
last_120_days = new_df[-Dp:].values

#Escale los datos para que sean valores entre 0 y 1
last_120_days_scaled = scaler.transform(last_120_days)

#Crear una lista vacía
X_test = []

#Añadir los últimos 120 días
X_test.append(last_120_days_scaled)

#Convierta el conjunto de datos X_test en una matriz
X_test = np.array(X_test)

#Redimensione los datos
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#Obtenga el precio escalado previsto
pred_price = model.predict(X_test)

#hacer el escalado
pred_price = scaler.inverse_transform(pred_price)
print('Margen proyectado ', pred_price)

"""
Ahora veamos cuál fue el margen real de ese día.
"""
#Obtenga el Quote
df2 =  pd.read_csv(file,header=0)
print(df2[-1:])

print('¡Proceso completo!')

#===========================================================================================================================

