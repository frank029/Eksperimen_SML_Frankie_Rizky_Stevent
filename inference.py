import joblib
import pandas as pd

model = joblib.load(
    "titanic_model.pkl"
)

sample = pd.DataFrame(
    [{
        "Pclass": 3,
        "Sex": 1,
        "Age": 22,
        "Fare": 7.25,
        "Embarked": 2
    }]
)

prediction = model.predict(
    sample
)

print(
    prediction
)