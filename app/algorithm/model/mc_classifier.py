import numpy as np, pandas as pd
import joblib
import sys
import os, warnings

warnings.filterwarnings("ignore")


from xgboost import XGBClassifier


trainX_fname = "trainX_fname.save"
model_fname = "model.save"
MODEL_NAME = "multi_class_base_xgboost"


class Classifier:
    def __init__(self, **kwargs) -> None:
        self.model = self.build_model(**kwargs)
        self.train_X = None

    def build_model(self, **kwargs):
        model = XGBClassifier(**kwargs, verbosity=0)
        return model

    def fit(self, train_X, train_y):
        self.train_X = train_X  # need to save the data for explanations
        self.model.fit(train_X, train_y)

    def predict(self, X):
        preds = self.model.predict(X)
        return preds

    def predict_proba(self, X):
        preds = self.model.predict_proba(X)
        return preds

    def summary(self):
        self.model.get_params()

    def evaluate(self, x_test, y_test):
        """Evaluate the model and return the loss and metrics"""
        if self.model is not None:
            return self.model.score(x_test, y_test)

    def save(self, model_path):
        return joblib.dump(self, os.path.join(model_path, model_fname))

    @classmethod
    def load(cls, model_path):
        classifier = joblib.load(os.path.join(model_path, model_fname))
        return classifier


def save_model(model, model_path):
    model.save(model_path)


def load_model(model_path):
    model = joblib.load(os.path.join(model_path, model_fname))
    return model
