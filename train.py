
import joblib
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression

data = load_diabetes()

X = data.data
y = data.target

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "diabetes_model.joblib")

print("Model trained and saved successfully!")
