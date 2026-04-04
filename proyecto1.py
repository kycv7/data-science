#import libraries
import pandas as pd #to manipulate tables
import numpy as np #for mathematical calculations
import matplotlib.pyplot as plt #to create graphics
import seaborn as sns #to prettier graphics
from sklearn.model_selection import train_test_split #to split the data
from sklearn.ensemble import RandomForestClassifier #the brain lof the model
from sklearn.metrics import classification_report #for metrics

#Exploratory data analysis
filepath="D:/proyectos/data science/dataset1.csv"

df=pd.read_csv(filepath)
print(df.head())
print(df.isnull().sum())
print(df.describe())

mediana_gasto=df['Gasto_Total'].median()
df['Gasto_Total'].fillna(mediana_gasto, inplace=True)
print(df['Gasto_Total'].isnull().sum())

plt.hist(df['Dias_desde_ultima_compra'],bins=30,edgecolor='black',alpha=0.7,color='steelblue')
plt.xlabel('Días desde última compra')
plt.ylabel('Frecuencia')
plt.title('Histograma: Días desde última compra')
plt.legend()
plt.show()

corr_matrix=df.corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix,annot=True,cmap='coolwarm',fmt=".2f",linewidths=0.5)
plt.title('Matriz de Correlación de Variables')
plt.show()

#model RANDOMFORESTCLASSIFIER
print(df.columns)
X=df.drop(['ID_Cliente','Se_Fue'],axis=1)
y=df['Se_Fue']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(classification_report(y_test, y_pred))

#Extract the importance of the variables
importancias = rf.feature_importances_
columnas = X.columns

#Create a DataFrame to visualize
feature_imp = pd.DataFrame({'Variable': columnas, 'Importancia': importancias})
feature_imp=feature_imp.sort_values(by='Importancia',ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importancia',y='Variable',data=feature_imp,palette='magma')
plt.title('¿Qué factores causan realmente la fuga del cliente?')
plt.show()
