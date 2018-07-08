import xgboost as xgb
import pandas as pd
import numpy as np

fullDF = pd.read_csv("iris.csv")

varsDF = fullDF[["sepal.length", "sepal.width", "petal.length"]]
respDF = fullDF[["petal.width"]]

DM = xgb.DMatrix(varsDF, label=respDF)

parameters={"max_depth":1, "learning_rate":0.1}

b = xgb.train(parameters, DM, num_boost_rounds=10)

preds = b.predict(DM)

c = xgb.train(parameters, DM, num_boost_rounds=10, xgb_model=b)

preds = c.predict(DM)
