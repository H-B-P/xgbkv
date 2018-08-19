import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def provide_average(pred,  act):
    return(np.mean(pred))

def provide_mae(pred, act):
    return(sum(abs(pred-act))/len(pred))

def provide_min(pred, act):
    return(min(pred))

def provide_max(pred, act):
    return(max(pred))

def provide_rmse(pred, act):
    return(np.sqrt(sum((pred-act)*(pred-act))/len(pred)))

def provide_Xile_pred_provider(n, N):
    def provide_Xile_pred(pred, act):
        inds=pred.argsort()
        sortedPred=pred[inds]
        quintileSubset=sortedPred[len(pred)*(n-1)/N:len(pred)*n/N]
        return sum(quintileSubset)/len(quintileSubset)
    return provide_Xile_pred

def provide_Xile_act_provider(n, N):
    def provide_Xile_act(pred, act):
        inds=pred.argsort()
        sortedAct=act[inds]
        quintileSubset=sortedAct[len(pred)*(n-1)/N:len(pred)*n/N]
        return sum(quintileSubset)/len(quintileSubset)
    return provide_Xile_act

metrics1DLookup={
"quintiles":["qu1 pred", "qu2 pred", "qu3 pred", "qu4 pred",  "qu5 pred",  "qu1 act",  "qu2 act",  "qu3 act",  "qu4 act",  "qu5 act"]}

evalLookup={
"average":provide_average, 
"mae":provide_mae, 
"min":provide_min, 
"max":provide_max, 
"rmse":provide_rmse, 
"qu1 pred":provide_Xile_pred_provider(1, 5), 
"qu2 pred":provide_Xile_pred_provider(2, 5), 
"qu3 pred":provide_Xile_pred_provider(3, 5), 
"qu4 pred":provide_Xile_pred_provider(4, 5),  
"qu5 pred":provide_Xile_pred_provider(5, 5), 
"qu1 act":provide_Xile_act_provider(1, 5), 
"qu2 act":provide_Xile_act_provider(2, 5), 
"qu3 act":provide_Xile_act_provider(3, 5), 
"qu4 act":provide_Xile_act_provider(4, 5), 
"qu5 act":provide_Xile_act_provider(5, 5), 
}

def XGBKVRegressor(params,  trainData,  nRounds, evalRounds=10,  metrics=["average", "mae", "min", "max",  "rmse"],  testData=None, outputs=["terminal","csv",  "graphs"], csvName="output.csv",  graphPrefix="output", ideals=True):
    
    if testData==None:
        testData=trainData
    
    #handle list(s) of metrics
    
    if "str" in str(type(metrics)):
        metrics=[metrics]
    for metric in metrics:
        if (metric not in evalLookup) and (metric not in metrics1DLookup):
            metrics.remove(metric)
    
    def remetricise1D(mets):
        for i in range(len(mets)):
            if mets[i] in metrics1DLookup:
                mets= mets[0:i]+metrics1DLookup[mets[i]]+mets[i+1:len(mets)]
                return remetricise1D(mets)
        return mets
    
    expandedMetrics=remetricise1D(metrics)
    metrics0D=[metric for metric in metrics if metric in expandedMetrics]
    metrics1D=[metric for metric in metrics if metric not in expandedMetrics]
    
    #rounds
    
    roundsToGo=nRounds
    roundPlusMetrics = ["round"]+expandedMetrics
    
    #prep outputs
    
    if "terminal" in outputs:
        params["silent"]=True
        
        template = " {0:10}"
        for i in range(len(expandedMetrics)):
            template+=" | {" + str(i+1)+":10}"
        print(template.format(*roundPlusMetrics))
        print ("------------"+"+------------"*len(expandedMetrics))
    
    if "csv" in outputs:
        csv = open(csvName, "w")
        csvLine=""
        for item in roundPlusMetrics:
            csvLine+=str(item)
            csvLine+=","
        csvLine=csvLine[:-1]
        csvLine+="\n"
        csv.write(csvLine)
    
    if "graphs" in outputs:
        graphfodder=[[] for x in range(len(roundPlusMetrics))]
        idealfodder=[[] for x in range(len(roundPlusMetrics))]
    
    bst=None
    
    while roundsToGo>0:
        bst=xgb.train(params, trainData, min(evalRounds, roundsToGo), xgb_model=bst)
        roundsToGo-=min(evalRounds, roundsToGo)
        
        predicted=bst.predict(testData)
        actual = testData.get_label()
        
        #calculate the metrics
        
        metricLine=[nRounds-roundsToGo]
        idealLine=[None]
        for metric in expandedMetrics:
            if metric in evalLookup:
                eval_func=evalLookup[metric]
                metricLine.append(eval_func(predicted, actual))
                idealLine.append(eval_func(actual, actual))
            else:
                metricLine.append(None)
                idealLine.append(None)
        
        #output to terminal
        
        if "terminal" in outputs:
            printLine=[]
            for item in metricLine:
                if item==None:
                    printLine.append("")
                else:
                    printLine.append(str(round(item, 8)))
            print(template.format(*printLine))
        
        #output to csv
        
        if "csv" in outputs:
            csvLine=""
            for item in metricLine:
                csvLine+=str(item)
                csvLine+=","
            csvLine=csvLine[:-1]
            csvLine+="\n"
            csv.write(csvLine)
        
        #add to graph output(s)
        
        if "graphs" in outputs:
            for i in range(len(graphfodder)):
                graphfodder[i].append(metricLine[i])
                idealfodder[i].append(idealLine[i])
    
    #output graphs
    
    if "graphs" in outputs:
        
        #graphs for 0D metrics
        
        for i in range(1, len(graphfodder)):
            if roundPlusMetrics[i] in metrics0D:
                if ideals:
                    plt.plot(graphfodder[0], idealfodder[i], "y")
                    yellow_patch = mpatches.Patch(color='yellow', label='ideal')
                    blue_patch = mpatches.Patch(color='blue', label='predicted')
                    plt.legend(handles=[blue_patch, yellow_patch])
                plt.plot(graphfodder[0], graphfodder[i],  "b.-")
                
                plt.xlim((0, max(graphfodder[0])*1.1))
                plt.ylim((min(min(graphfodder[i])*1.1, 0), max(max(idealfodder[i]),  max(graphfodder[i]))*1.1))
                plt.xlabel("rounds of training")
                plt.ylabel(roundPlusMetrics[i])
                plt.savefig(graphPrefix+"_"+roundPlusMetrics[i]+".png")
                plt.clf()
        
        #graphs for 1D metrics (bespoke enough that they need individual coding)
        
        if "quintiles" in metrics1D:
            subMetrics=metrics1DLookup["quintiles"]
            relevantData=np.array([graphfodder[expandedMetrics.index(subMetric)+1] for subMetric in subMetrics])
            idealData=np.array([idealfodder[expandedMetrics.index(subMetric)+1] for subMetric in subMetrics])
            for i in range(len(relevantData[0])):
                if ideals:
                    plt.plot([1, 2, 3, 4, 5], list(idealData[0:5, i]), "y.-")
                    yellow_patch = mpatches.Patch(color='yellow', label='ideal')
                plt.plot([1, 2, 3, 4, 5], list(relevantData[0:5, i]),  "b.-")
                blue_patch = mpatches.Patch(color='blue', label='predicted')
                plt.plot([1, 2, 3, 4, 5], list(relevantData[5:10, i]), "r.-")
                red_patch = mpatches.Patch(color='red', label='actual')
                if ideals:
                    plt.legend(handles=[blue_patch, red_patch, yellow_patch])
                else:
                    plt.legend(handles=[blue_patch, red_patch])
                plt.xlim(0.5, 5.5)
                plt.ylim(relevantData.min()-(relevantData.max()-relevantData.min())*0.1,  relevantData.max()+(relevantData.max()-relevantData.min())*0.1)
                plt.xlabel("quintile (predicted)")
                plt.ylabel("average response")
                plt.title('quintiles, round '+str(graphfodder[0][i]))
                plt.savefig(graphPrefix+"_deciles_round"+str(graphfodder[0][i])+".png")
                plt.clf()
    
    return bst


