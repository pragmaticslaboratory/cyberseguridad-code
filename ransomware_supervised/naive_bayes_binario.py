"""analisis de malware y benigno con dataset y naive
preprocesado por chinos con binario
paper 244802 en df 293333"""


import pandas as pd

import matplotlib.pyplot as plt



"""matplotlib inline"""
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

from sklearn import datasets, metrics

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
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

"""
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
"""

"""convertir columna tipo dato str a int"""
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

# Split dataset in training and test datasets
X_train, X_test, y_train, y_test = train_test_split(df2[['Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts',
       'TotLen Bwd Pkts', 'Fwd Pkt Len Min', 'Bwd Pkt Len Min',
       'Fwd Pkt Len Max', 'Bwd Pkt Len Max', 'Fwd Pkt Len Mean',
       'Bwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Std', 'Flow Pkts/s',
       'Flow Byts/s']],df2['Label'], test_size=0.2, random_state=6) 
 
# Instantiate the classifier
gnb = GaussianNB()  #MultinomialNB()
# Train classifier
gnb.fit(X_train.values,y_train)
y_pred = gnb.predict(X_test)

print('Precisión en el set de Entrenamiento: {:.2f}'
     .format(gnb.score(X_train, y_train)*100))
print('Precisión en el set de Test: {:.2f}'
     .format(gnb.score(X_test, y_test)*100))

matriz = confusion_matrix(y_test, y_pred)
print('Matriz de Confusión Validad :')
print(matriz)
print(classification_report(y_test, y_pred))

"""
dtype: int64
Precisión en el set de Entrenamiento: 82.67
Precisión en el set de Test: 82.89
Matriz de Confusión Validad :
[[22421  5142]
 [ 2138 12858]]
              precision    recall  f1-score   support

     Benigno       0.91      0.81      0.86     27563
  Ransomware       0.71      0.86      0.78     14996

    accuracy                           0.83     42559
   macro avg       0.81      0.84      0.82     42559
weighted avg       0.84      0.83      0.83     42559


sin
Label
Benigno       137907
Ransomware     74888
dtype: int64
Precisión en el set de Entrenamiento: 64.16
Precisión en el set de Test: 63.93
Matriz de Confusión Validad :
[[26122  1404]
 [13945  1088]]
              precision    recall  f1-score   support

     Benigno       0.65      0.95      0.77     27526
  Ransomware       0.44      0.07      0.12     15033

    accuracy                           0.64     42559
   macro avg       0.54      0.51      0.45     42559
weighted avg       0.58      0.64      0.54     42559

"""