import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ====================================
# LOAD DATA
# ====================================

df = pd.read_csv("titanic_preprocessed.csv")

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ====================================
# HYPERPARAMETER TUNING
# ====================================

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(
        random_state=42
    ),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(
    X_train,
    y_train
)

best_model = grid_search.best_estimator_

predictions = best_model.predict(
    X_test
)

# ====================================
# METRICS
# ====================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions
)

recall = recall_score(
    y_test,
    predictions
)

f1 = f1_score(
    y_test,
    predictions
)

# ====================================
# MANUAL LOGGING
# ====================================

with mlflow.start_run():

    # parameter terbaik
    mlflow.log_params(
        grid_search.best_params_
    )

    # metrics
    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )

    # model
    mlflow.sklearn.log_model(
        best_model,
        "model"
    )

print("Best Parameters:")
print(grid_search.best_params_)

print("\nMetrics:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")