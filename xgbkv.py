import xgboost as xgb
import numpy as np

def provide_average(pred,  act):
    return(np.mean(pred))

def provide_mae(pred, act):
    return(sum(abs(pred-act))/len(pred))

def provide_min(pred, act):
    return(min(pred))

def provide_max(pred, act):
    return(max(pred))

evalLookup={
"average":provide_average, 
"mae":provide_mae, 
"min":provide_min, 
"max":provide_max}

def XGBKVRegressor(params,  trainData,  nRounds, evalRounds=10,  analytics=["average", "mae", "min", "max"],  testData=None, outputs=["terminal",  "csv",  "graphs"]):
    
    if testData==None:
        testData=trainData
    
    roundsToGo=nRounds
    
    params["silent"]=True
    
    template = " {0:10}"
    for i in range(len(analytics)):
        template+=" | {" + str(i+1)+":10}"
    
    roundPlusAnalytics = ["round"]+analytics
    
    print(template.format(*roundPlusAnalytics))
    
    print ("------------"+"+------------"*len(analytics))
    
    #bst=xgb.train(params,  trainData,  min(evalRounds, nRounds))
    
    bst=None
    
    while roundsToGo>0:
        bst=xgb.train(params, trainData, min(evalRounds, roundsToGo), xgb_model=bst)
        roundsToGo-=evalRounds
        
        predicted=bst.predict(testData)
        actual = testData.get_label()
        
        pline=[str(nRounds-roundsToGo)]
        for metric in analytics:
            eval_func=evalLookup[metric]
            pline.append(str(round(eval_func(predicted, actual), 8)))
        
        print(template.format(*pline))
    
    return bst


