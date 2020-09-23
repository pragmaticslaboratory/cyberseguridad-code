""" K-NEIBHORG CLASE BINARY RANSOMWARE"""

import pandas as pd
import matplotlib.pyplot as plt

"""matplotlib inline"""
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

from sklearn import datasets, metrics

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import os
import glob

# Primero especificamos un patrón del archivo y lo pasamos como parámetro en la función glob
os.chdir("C:\\DatosChinos\\ransomware")
os.getcwd()
csv_files = glob.glob('*.csv')
# Mostrar el archivo csv_files, el cual es una lista de nombres
print(csv_files)


list_data = []
  
# Escribimos un loop que irá a través de cada uno de los nombres de archivo a través de globbing y el resultado final será la lista dataframes

for filename in csv_files:
    data = pd.read_csv(filename)
    list_data.append(data)

#Para chequear que todo está bien, mostramos la list_data por consola
#list_data
 
df = pd.concat(list_data,ignore_index=True)

 #tipos de datos de campos

"""información datos"""
print(df.head(5))
print('Cantidad de Filas y columnas:',df.shape)
print('Nombre columnas:',df.columns)

"""Columnas, nulos y tipo de datos"""
print(df.info())

"""descripción estadística de los datos numéricos"""
print(df.describe())

"""correlación entre los datos"""
corr = df.set_index('Label').corr()
#sm.graphics.plot_corr(corr, xnames=list(corr.columns))
#plt.show()

df2=df[['Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts',
       'TotLen Bwd Pkts', 'Fwd Pkt Len Min', 'Bwd Pkt Len Min',
       'Fwd Pkt Len Max', 'Bwd Pkt Len Max', 'Fwd Pkt Len Mean',
       'Bwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Std', 'Flow Pkts/s',
       'Flow Byts/s']]
df3=df[['Label']]




from sklearn.preprocessing import KBinsDiscretizer
bines = 2 # Elegir el número de bines
cabecera = list(df2) # Guardamos los nombres de las columnas.
ind = 0 # Contador para iterar por columnas.
while (ind < len(cabecera)):
    disc = df2.iloc[:,ind] 
    disc = disc.to_frame() 
    disc = KBinsDiscretizer(n_bins=bines, encode='ordinal',
                            strategy = "quantile").fit_transform(disc)
    df2[cabecera[ind]] = disc 
    ind = ind + 1
    del(disc)   
    
    
    
"""convertir columna tipo dato str a int"""
#df2['Label']=df3[['Label']].astype(object)
df2['Label']=df3[['Label']]
#print(df2.groupby('Label').size())

"""desordena el dataset"""

df2=df2.sort_values('Fwd Pkt Len Min')
df2 = df2[df2['Label'].isin([1,4])]

""" convertir valores número la clase"""
df2['Label'] = df2['Label'].replace(1,"Benigno")
df2['Label'] = df2['Label'].replace(4,"Ransomware")

#print(df2.head(10))
print(df2.groupby('Label').size())

#df2.drop(['Label'], axis=1).hist()
#plt.show()

X=df2.drop(['Label'], axis=1)
y=df2['Label']


  
# dividing X, y into train and test data 
X_train, X_test, y_train, y_test = train_test_split(df2[['Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts',
       'TotLen Bwd Pkts', 'Fwd Pkt Len Min', 'Bwd Pkt Len Min',
       'Fwd Pkt Len Max', 'Bwd Pkt Len Max', 'Fwd Pkt Len Mean',
       'Bwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Std', 'Flow Pkts/s',
       'Flow Byts/s']],df2['Label'], test_size=0.2, random_state=6) 
   
# training a KNN classifier 
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier().fit(X_train, y_train) 
accuracy = knn.score(X_train, y_train) *100
print ("Precisión Entrenamiento " + str(accuracy))
  
# accuracy on X_test 
accuracy = knn.score(X_test, y_test) *100
print ("Precisión Validado " + str(accuracy))
  
# creating a confusion matrix 
knn_predictions = knn.predict(X_test)  
cm = confusion_matrix(y_test, knn_predictions) 
print(cm)
print("Métricas")
print(classification_report(y_test, knn_predictions))

"""
dtype: int64
Precisión Entrenamiento 91.46361521652294
Precisión Validado 88.76383373669495
[[24811  2715]
 [ 2067 12966]]
Métricas
              precision    recall  f1-score   support

     Benigno       0.92      0.90      0.91     27526
  Ransomware       0.83      0.86      0.84     15033

    accuracy                           0.89     42559
   macro avg       0.87      0.88      0.88     42559
weighted avg       0.89      0.89      0.89     42559


con
Label
Benigno       137907
Ransomware     74888
dtype: int64
Precisión Entrenamiento 91.02363777344392
Precisión Validado 91.0500716652177
[[25420  2143]
 [ 1666 13330]]
Métricas
              precision    recall  f1-score   support

     Benigno       0.94      0.92      0.93     27563
  Ransomware       0.86      0.89      0.87     14996

    accuracy                           0.91     42559
   macro avg       0.90      0.91      0.90     42559
weighted avg       0.91      0.91      0.91     42559

"""
