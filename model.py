import pandas as pd
df=pd.read_csv("car data.csv")
print(df.head())
print(df.info())
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
print("After cleaning:", df.shape)
df = pd.get_dummies(df)
X = df[["Year", "Present_Price", "Kms_Driven"]]
y = df["Selling_Price"]
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)
from sklearn.metrics import r2_score
y_pred = model.predict(X_test)
print("Accuracy:", r2_score(y_test, y_pred))
import pickle
pickle.dump(model, open("model.pkl", "wb"))