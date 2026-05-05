import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)
df

df.shape

df.size

df.columns

df.info()

df.describe()

X = df.drop('medv', axis=1)   # features
y = df['medv']                # target

X

y

X.describe()

y.describe()

X.shape

y.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

result = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
result.head()

result

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()

from sklearn.metrics import r2_score
score = r2_score(y_test, y_pred)
print("\nR2 Score:", score)

model.coef_

