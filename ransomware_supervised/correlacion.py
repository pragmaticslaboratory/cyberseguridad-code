""" ARBOL CON CLASE BINARIA BENIGNOS Y RANSOMWARE"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix 
from sklearn import datasets, metrics
from sklearn.metrics import classification_report 
import os

import glob

"""crea entorno grÃ¡fico"""
plt.rcParams['figure.figsize'] = (10, 20)
plt.style.use('ggplot')

# Primero especificamos un patrón del archivo y lo pasamos como parámetro en la función glob
#os.chdir("C:\\DatosChinos\\ransomware")
os.chdir("C:\\Users\\dieku\\Documents\\GitHub\\cyberseguridad-code\\ransomware_supervised")
os.getcwd()
csv_files = glob.glob('*.csv')
# Mostrar el archivo csv_files, el cual es una lista de nombres
print(csv_files)#No es un archivo son varios ? 

list_data = []
  
# Escribimos un loop que irá a través de cada uno de los nombres de archivo a través de globbing y el resultado final será la lista dataframes

for filename in csv_files:
    data = pd.read_csv(filename)
    list_data.append(data)

#Para chequear que todo está bien, mostramos la list_data por consola
#list_data
 
df = pd.concat(list_data,ignore_index=True)

tipos=df.dtypes  #tipos de datos de campos

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
"""
import seaborn as sb
pearsoncorr = df2.corr(method='pearson')
sb.heatmap(pearsoncorr, 
            xticklabels=pearsoncorr.columns,
            yticklabels=pearsoncorr.columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=1)
"""
import seaborn as sb
pearsoncorr = df2.corr(method='kendall')
sb.heatmap(pearsoncorr, 
            xticklabels=pearsoncorr.columns,
            yticklabels=pearsoncorr.columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=1)

print(df2.corr('pearson'))
pear=df2.corr('pearson')
print(df2.corr('kendall'))
ken=df2.corr('kendall')


data = df2[['Bwd Pkt Len Max','Bwd Pkt Len Max']]
correlation = data.corr(method='pearson')


from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE


svm = LinearSVC()
# create the RFE model for the svm classifier 
# and select attributes
rfe = RFE(svm, 3)
rfe = rfe.fit(df2, df2)
# print summaries for the selection of attributes
print(rfe.support_)
print(rfe.ranking_)






"""convertir columna tipo dato str a int"""
df2['Label']=df3[['Label']]

print(df2.groupby('Label').size())

"""desordena el dataset"""
df2=df2.sort_values('Fwd Pkt Len Min')

df2 = df2[df2['Label'].isin([1,4])]


""" convertir valores número la clase"""
df2['Label'] = df2['Label'].replace(1,"Benigno")
df2['Label'] = df2['Label'].replace(4,"Ransomware")



print(df2.groupby('Label').size())

"""CREAMOS EL MODELO Y LO PROBAMOS CON LOS MISMOS DATOS"""

X = np.array(df2.drop(['Label'],1))
y = np.array(df2['Label'])

from sklearn.tree import DecisionTreeClassifier

X_train, X_test, Y_train, Y_test = train_test_split(
        df2[['Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts',
       'TotLen Bwd Pkts', 'Fwd Pkt Len Min', 'Bwd Pkt Len Min',
       'Fwd Pkt Len Max', 'Bwd Pkt Len Max', 'Fwd Pkt Len Mean',
       'Bwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Std', 'Flow Pkts/s',
       'Flow Byts/s']]
, df2['Label'], test_size = 0.20, random_state=0)

clf2 = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=7,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, 
                       random_state=0, splitter='best')#presort=False,
#entreno modelo

clf2.fit(X_train, Y_train)

acc_decision_tree = round(clf2.score(X_train, Y_train) * 100, 2)
print("Precisión Modelo entrenado (Accuracy) : ", acc_decision_tree, "%")
acc_decision_tree = round(clf2.score(X_test, Y_test) * 100, 2)
print("Precisión Modelo validado (Accuracy) : ", acc_decision_tree, "%")

#Realizo una predicción
y_pred2 = clf2.predict(X_test)

matriz = confusion_matrix(Y_test, y_pred2)
print('Matriz de Confusión AD Validada :')
print(matriz)

print("Métricas AD Validadas")
print(classification_report(Y_test, y_pred2))


"""
dtype: int64
Precisión Modelo entrenado (Accuracy) :  93.48 %
Precisión Modelo validado (Accuracy) :  93.45 %
Matriz de Confusión AD Validada :
[[26036  1556]
 [ 1233 13734]]
Métricas AD Validadas
              precision    recall  f1-score   support

     Benigno       0.95      0.94      0.95     27592
  Ransomware       0.90      0.92      0.91     14967

    accuracy                           0.93     42559
   macro avg       0.93      0.93      0.93     42559
weighted avg       0.93      0.93      0.93     42559

con
dtype: int64
Label
Benigno       137907
Ransomware     74888
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

