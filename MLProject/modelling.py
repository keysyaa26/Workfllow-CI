import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import (train_test_split, GridSearchCV)
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)


# Data Prepare
X_train = pd.read_csv('dataset_preprocessed/X_train.csv')
X_test = pd.read_csv('dataset_preprocessed/X_test.csv')
y_train = pd.read_csv('dataset_preprocessed/y_train.csv')
y_test = pd.read_csv('dataset_preprocessed/y_test.csv')


# MLflow
# mlflow.set_tracking_uri("file:./mlruns")
mlflow.sklearn.autolog()

# Training
with mlflow.start_run():
    model = SVC()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"Accuracy: {accuracy:.4f}")