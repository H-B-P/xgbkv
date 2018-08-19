import pandas as pd
import xgboost as xgb
import numpy as np
import xgbkv

fullDF = pd.read_csv("iris.csv")

#REG

#varsDF = fullDF[["sepal.length", "sepal.width", "petal.length"]]
#respDF = fullDF[["petal.width"]]
#
#DM = xgb.DMatrix(varsDF, label=respDF)
#parameters={"max_depth":1, "learning_rate":0.1}
#
#a = xgbkv.XGBKVRegressor(parameters,  DM,  100, metrics=["min", "max", "rmse",  "mae"])
#
#a.predict(DM)
#
#b = xgbkv.XGBKVRegressor(parameters,  DM,  100, 3,   metrics=["quintiles",  "mae"])

#CLASS

varsDF = fullDF[["sepal.length", "sepal.width", "petal.length", "petal.width"]]
respDF = fullDF[["variety"]]

respDF["variety"] = respDF["variety"].map({"Setosa":0,"Versicolor":1, "Virginica":2})

DM = xgb.DMatrix(varsDF, label=respDF)

parameters={"max_depth":2, "learning_rate":0.1,  "objective":"multi:softprob",  "num_class":3}

c=xgbkv.XGBKVClassifier(parameters,  DM,  30)
