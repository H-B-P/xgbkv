import pandas as pd
import xgboost as xgb
import numpy as np
import xgbkv

fullDF = pd.read_csv("iris.csv")

varsDF = fullDF[["sepal.length", "sepal.width", "petal.length"]]
respDF = fullDF[["petal.width"]]

DM = xgb.DMatrix(varsDF, label=respDF)
parameters={"max_depth":1, "learning_rate":0.1}

b = xgbkv.XGBKVRegressor(parameters,  DM,  100)
