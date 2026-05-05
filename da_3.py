import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
col_names = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width','Species']
df = pd.read_csv(url, names=col_names)
print(df)

df.info()

print(df.describe())

print(df.isnull().sum())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])
print("\nEncoded Data:")
print(df.head())

X = df[['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']]
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

model = GaussianNB()
model.fit(X_train, y_train)
print("\nModel trained successfully")

y_pred = model.predict(X_test)
print("\nPredicted values:")
print(y_pred)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

accuracy = np.sum(y_pred == y_test) / len(y_test)
print("\nAccuracy:", accuracy)

import seaborn as sns
import matplotlib.pyplot as plt
sns.pairplot(df, hue='Species')
plt.show()

sns.boxplot(x='Species', y='Petal_Length', data=df)
plt.show()

sns.boxplot(x='Species', y='Petal_Width', data=df)
plt.show()
sns.boxplot(x='Species', y='Sepal_Length', data=df)
plt.show()
sns.boxplot(x='Species', y='Sepal_Width', data=df)
plt.show()

df.corr()