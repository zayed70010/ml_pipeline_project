import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

df = pd.read_csv('./data/sales.csv')

encoder = LabelEncoder()
df.Street = encoder.fit_transform(df.Street)
df.SaleCondition = encoder.fit_transform(df.SaleCondition)
df['TotalArea'] = df['GrLivArea'] + df['GarageArea']

x = df[['GrLivArea', 'GarageArea', 'OverallQual', 'Street', 'SaleCondition', 'TotalArea']]
y = df['SalePrice']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=123)

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

mlflow.set_experiment("linear model sales")
with mlflow.start_run():
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    mlflow.sklearn.log_model(model, "model")
    best_model_path = mlflow.get_artifact_uri("model")

with open("best_model.txt", "w") as f:
    f.write(best_model_path)

print("✅ Training done and model path saved.")
