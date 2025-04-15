import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import mlflow
import mlflow.sklearn

# تحميل البيانات
df = pd.read_csv("data/vehicles_dataset_prepared.csv")

# تجهيز البيانات
X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تفعيل mlflow لتتبع التجربة
mlflow.set_experiment("linear_regression_car_prices")

with mlflow.start_run():
    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    # تسجيل القيم في mlflow
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("mse", mse)

    # تسجيل النموذج
    mlflow.sklearn.log_model(model, "model")

    print("Model trained and logged with MLflow.")
