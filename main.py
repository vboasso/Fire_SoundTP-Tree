import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
# Necesarias para mostrar el arbol:
from sklearn.tree import export_graphviz
#from sklearn.externals.six import StringIO
from six import StringIO
from IPython.display import Image
import pydotplus

# Cargamos el dataset

DataSet = pd.read_excel('/home/vale/PycharmProjects/Fire_soundTP/Fire_Dataset.xlsx')
print(DataSet.head())  # Reviso que se haya cargado

print("Dataset info")
DataSet.info()  # Veo si hay valores null y el tipo de variables

print("Valores NaN")
print(DataSet.isna().sum())  # Veo si hay valores NaN

print("Estadísticas")
print(DataSet.describe())  # Obtengo estadísticas de los datos

oe = OrdinalEncoder()
DataSet['FUEL'] = oe.fit_transform(DataSet[['FUEL']])  # Ordinal encoden me convierte de categorica a array de enteros

# Con ésto puedo ver la interelación entre las variables:

plt.figure(figsize=(10, 5))
sns.heatmap(DataSet.corr(), annot=True, cmap='viridis', fmt='.2f')
plt.show()

# Defino los vectores

feature_cols = ['SIZE', 'FUEL', 'DISTANCE', 'DESIBEL', 'AIRFLOW', 'FREQUENCY', 'FREQUENCY']
X = DataSet[feature_cols]  # Features
y = DataSet.STATUS  # Target variable

print("Columnas de entrada")
print(X)
print("Etiqueta(estado de la llama):")
print(y)  # Esta sería mi etiqueta

# Divido mi dataset en prueba y entrenamiento

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)  # 80% training and 20% test

#Escalo mis variables

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Creo el arbol

from sklearn.ensemble import RandomForestClassifier
clf = DecisionTreeClassifier(max_depth=5)

# Entreno el arbol

clf = clf.fit(X_train,y_train)

# Cargo los datos de test para hacer una predicción

y_pred = clf.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('FireRF.png')
Image(graph.create_png())

# Genero la matriz de confusión

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='xx-large')

plt.xlabel('Predicción', fontsize=18)
plt.ylabel('Reales', fontsize=18)
plt.title('Matriz de confusión', fontsize=18)
plt.show()

