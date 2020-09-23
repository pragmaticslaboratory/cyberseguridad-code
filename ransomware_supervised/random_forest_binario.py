"""RANDOM FOREST CLASE BINARIA RANSOMWARE"""


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report 
import os
import glob

"""crea entorno grÃ¡fico"""
plt.rcParams['figure.figsize'] = (40, 50)
plt.style.use('ggplot')


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


"""convertir columna tipo dato str a int"""
df2['Label']=df3[['Label']]

#print(df2.groupby('Label').size())
df2 = df2[df2['Label'].isin([1,4])]


""" convertir valores número la clase"""
df2['Label'] = df2['Label'].replace(1,"Benigno")
df2['Label'] = df2['Label'].replace(4,"Ransomware")

"""desordena el dataset"""

df2=df2.sort_values('Fwd Pkt Len Min')

print(df2.groupby('Label').size())



"""random forest"""

#Splitting the data into independent and dependent variables
X = df2.iloc[:,0:15].values
y = df2.iloc[:,15].values

# Creating the Training and Test set from data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 21)

# Fitting Random Forest Classification to the Training set
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
classifier.fit(X_train, y_train)

#Realizo una predicción
y_pred2 = classifier.predict(X_test)

acc_random_tree = round(classifier.score(X_train, y_train) * 100, 2)
print("Precisión Modelo entrenado (Accuracy) : ", acc_random_tree, "%")

acc_random_tree = round(classifier.score(X_test, y_test) * 100, 2)
print("Precisión Modelo validado (Accuracy) : ", acc_random_tree, "%")

matriz = confusion_matrix(y_test, y_pred2)

print('Matriz de Confusión RF Validado :')
print(matriz)

print("Métricas RF  Validadas")
print(classification_report(y_test, y_pred2))

"""
dtype: int64
Precisión Modelo entrenado (Accuracy) :  99.67 %
Precisión Modelo validado (Accuracy) :  95.73 %
Matriz de Confusión RF Validado :
[[26350  1388]
 [  430 14391]]
Métricas RF  Validadas
              precision    recall  f1-score   support

     Benigno       0.98      0.95      0.97     27738
  Ransomware       0.91      0.97      0.94     14821

    accuracy                           0.96     42559
   macro avg       0.95      0.96      0.95     42559
weighted avg       0.96      0.96      0.96     42559

con
dtype: int64
Precisión Modelo entrenado (Accuracy) :  90.92 %
Precisión Modelo validado (Accuracy) :  91.02 %
Matriz de Confusión AD Validada :
[[25505  1958]
 [ 1865 13231]]
Métricas AD Validadas
              precision    recall  f1-score   support

     Benigno       0.93      0.93      0.93     27463
  Ransomware       0.87      0.88      0.87     15096

    accuracy                           0.91     42559
   macro avg       0.90      0.90      0.90     42559
weighted avg       0.91      0.91      0.91     42559
"""
