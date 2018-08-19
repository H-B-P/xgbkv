import xgboost as xgb
import pandas as pd
import numpy as np

fullDF = pd.read_csv("iris.csv")

varsDF = fullDF[["sepal.length", "sepal.width", "petal.length", "petal.width"]]
respDF = fullDF[["variety"]]

respDF["variety"] = respDF["variety"].map({"Setosa":0,"Versicolor":1, "Virginica":2})

print(respDF)

DM = xgb.DMatrix(varsDF, label=respDF)

parameters={"max_depth":1, "learning_rate":0.1,  "objective":"multi:softprob",  "num_class":3}

b = xgb.train(parameters, DM, 10)

preds = b.predict(DM)

bestPreds = np.asarray([np.argmax(line) for line in preds])

print(bestPreds)
print(respDF)
