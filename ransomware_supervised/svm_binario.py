"""analisis de malware y benigno con dataset y SVM
preprocesado por chinos con binary
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
#df2['Label']=df3[['Label']].astype(object)
df2['Label']=df3[['Label']]
print(df2.groupby('Label').size())

"""desordena el dataset"""

df2=df2.sort_values('Fwd Pkt Len Min')
df2 = df2[df2['Label'].isin([1,4])]

""" convertir valores número la clase"""
df2['Label'] = df2['Label'].replace(1,"Benigno")
df2['Label'] = df2['Label'].replace(4,"Ransomware")
#print(df2.head(10))
#print(df2.groupby('Label').size())

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
   
# training a linear SVM classifier 
from sklearn.svm import SVC 
svm_model_linear = SVC(kernel='rbf', random_state=6, gamma=.01, C=1).fit(X_train, y_train) 
svm_predictions = svm_model_linear.predict(X_test) 
  
# model accuracy for X_test   
accuracy = svm_model_linear.score(X_train, y_train)*100
print("precisión entrenamiento : " + str(accuracy))
accuracy = svm_model_linear.score(X_test, y_test)*100
print("precisión validado : " + str(accuracy))
# creating a confusion matrix 
cm = confusion_matrix(y_test, svm_predictions) 
print(cm)
print(classification_report(y_test, svm_predictions))





"""
con
precisión entrenamiento : 89.00350102211048
precisión validado : 89.32540708193332
[[25730  1833]
 [ 2710 12286]]
              precision    recall  f1-score   support

     Benigno       0.90      0.93      0.92     27563
  Ransomware       0.87      0.82      0.84     14996

    accuracy                           0.89     42559
   macro avg       0.89      0.88      0.88     42559
weighted avg       0.89      0.89      0.89     42559


sin


"""