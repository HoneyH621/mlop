
import joblib
import numpy as np

model = joblib.load("diabetes_model.joblib")

def test_model_prediction():
    input_data = np.array(
        [0.02, 0.07, 0.03, 0.02, 0.01,
         0.03, 0.01, 0.04, 0.03, 0.01]
    ).reshape(1, -1)

    prediction = model.predict(input_data)

    assert isinstance(prediction[0], (float, np.floating))


def test_model_output_shape():
    input_data = np.zeros((5, 10))

    predictions = model.predict(input_data)

    assert predictions.shape == (5,)
