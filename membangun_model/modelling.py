import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

mlflow.sklearn.autolog()

df = pd.read_csv("titanic_preprocessed.csv")

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

with mlflow.start_run():

    model = RandomForestClassifier(
        random_state=42
    )

    model.fit(X_train, y_train)