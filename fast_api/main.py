from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from enum import Enum
import pickle
import pandas as pd
from typing import List

app = FastAPI()


@app.post("/train")
async def train_model(
        X: List[List[float]],
        y: List[int],
        clf_type: str,
        model_name: str
):
    """
    Fits the model with the given dataset.
    :param X: Features to be fitted (X_train)
    :param y: Target to be fitted (y_train)
    :param clf_type: Type of the classifier: DecisionTreeClassifier or LogisticRegression
    :param model_name: Name of the model to save
    :return: Fitted classifier
    """
    if clf_type == 'decision_tree_classifer':
        model = DecisionTreeClassifier()
    elif clf_type == 'logistic_regression':
        model = LogisticRegression()
    else:
        return {"message": "Such model is not supported"}

    model.fit(X, y)

    pkl_filename = f"{model_name}_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)

    return {"message": "The model has been trained and saved successfully"}


@app.post("/predict/{model_name}")
async def predict(model_name: str, X_to_pred: List[List[float]]):
    """
    Predict with the given model
    :param model_name: name of the fitted model
    :param X_to_pred: data to predict
    :return:
    """
    df = pd.DataFrame(X_to_pred)

    pkl_filename = f"{model_name}_model.pkl"
    with open(pkl_filename, 'rb') as file:
        model = pickle.load(file)

    prediction = model.predict(df)

    return {"prediction": prediction.tolist()}
