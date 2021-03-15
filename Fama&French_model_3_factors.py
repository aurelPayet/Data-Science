# -*- coding: utf-8 -*-
# Imports
import pandas as pd
import numpy as np 
import datetime as dt
import matplotlib.pyplot as plt
import statsmodels.api as sm
import openpyxl
from sklearn import metrics
from os import listdir
from datetime import datetime, timedelta

# Functions
def enumerate2(xs, start=0, step=1):
    for x in xs:
        yield (start, x)
        start += step

def preprocess_financial_data(file_path,vix_path):
    
    df_stock=pd.read_csv(file_path,delimiter=";", decimal=",")

    columns_df=df_stock.columns

    idx_turnover1 = list(columns_df).index("TURNOVER.1")
    df_stock = df_stock.drop(labels=[columns_df[idx_turnover1], columns_df[idx_turnover1+1]], axis=1)
    columns_df=df_stock.columns

    columns_data=[]
    for i in range(0,len(columns_df)-1,2):
        columns_data.append(columns_df[i])

    dfs={}
    for i in range(0,len(columns_df)-1,2):
        df = pd.DataFrame(df_stock.iloc[:,i:i+2])
        idx = (int)(i/2)
        df[columns_data[idx]] = pd.to_datetime(df[columns_data[idx]]) #revoir, potentiellement là le problème (à printer pour checker)
        dfs[str(columns_df[i])] = df
        
    for i,value in enumerate2(columns_data,start=1,step=2):
        dfs[str(value)]=dfs[str(value)].rename(columns={dfs[str(value)].columns[0]:"date",dfs[str(value)].columns[1]:str(value)})

    for i in columns_data:
        dfs[str(i)]=dfs[str(i)].set_index("date").dropna()

    df_final=pd.DataFrame(dfs[str(columns_data[0])])
    
    for i in range(1,len(columns_data),1):
        df_final=df_final.merge(pd.DataFrame(dfs[str(columns_data[i])]), on="date",how='outer')

    df_final = df_final.dropna(how="all")

    #print(list(df_final.sort_index().index))

    df_vix=pd.read_csv(vix_path,delimiter=";", decimal=",")
    df_vix["date"] = pd.to_datetime(df_vix["date"])
    df_vix=df_vix.set_index("date")
    df_final=df_final.merge(pd.DataFrame(df_vix),how='left', left_index=True, right_index=True)
    
    df_final=df_final.sort_index()

    df_final=df_final.drop(df_final[df_final.index < dt.datetime(2005,1,1)].index)

    return df_final

def extract_price(df):
    df_price = df["PX_last"].reset_index()
    df_price.rename(columns = {"PX_last":"Price","date":"Date"}, inplace = True)
    return df_price

def open_ff(file_path):
    df_ff = pd.DataFrame(pd.read_csv(file_path,delimiter=","))
    df_ff['Unnamed: 0'] = pd.to_datetime(df_ff['Unnamed: 0'], format='%Y%m%d')
    df_ff.rename(columns ={"Unnamed: 0":"Date"}, inplace = True)
    return df_ff

def merging(df1,df2):
    df_final = df1.merge(df2,how = 'left',right_on = 'Date',left_on = 'Date')
    return df_final

def removal_2021(df):
    df = df[df["Date"]<= "2020-11-30"]
    return df

def fill_na(df):
    df.fillna(method='ffill', inplace=True)
    df = df.dropna(inplace = True)
    return df

def reset_index(df):
    df = df.reset_index()
    df = df.drop(["index"],axis = 1)
    return df

def concat(df1,df2):
    df_concat = pd.concat([df1,df2],axis=1)
    return df_concat

def moving_average_rm_rf(df):
    list_Mkt_RF = []
    list_MA_Mkt_RF = []
    i = 0
    while i <= (len(df["Date"])-1):
        list_Mkt_RF.append(df["Mkt-RF"][i])
        ma = np.mean(list_Mkt_RF)
        list_MA_Mkt_RF.append(ma)
        i +=1
    return list_MA_Mkt_RF

def moving_average_SMB(df):
    list_SMB = []
    list_MA_SMB = []
    i = 0
    while i <= (len(df["Date"])-1):
        list_SMB.append(df["SMB"][i])
        ma = np.mean(list_SMB)
        list_MA_SMB.append(ma)
        i +=1
    return list_MA_SMB

def moving_average_HML(df):
    list_HML = []
    list_MA_HML = []
    i = 0
    while i <= (len(df["Date"])-1):
        list_HML.append(df["HML"][i])
        ma = np.mean(list_HML)
        list_MA_HML.append(ma)
        i +=1
    return list_MA_HML

def pandator_ma(df):
    list_1 = moving_average_rm_rf(df)
    list_2 = moving_average_SMB(df)
    list_3 = moving_average_HML(df)
    ma_rm_rf = pd.DataFrame(list_1,columns = ['MA_Rm-Rf'])
    ma_smb = pd.DataFrame(list_2,columns = ['MA_SMB'])
    ma_hml = pd.DataFrame(list_3,columns = ['MA_HML'])
    return ma_rm_rf,ma_smb,ma_hml

def super_concatener(df1,df2,df3,df4):
    df_intermed_1 = concat(df1,df2)
    df_intermed_2 = concat(df_intermed_1,df3)
    df_intermed_3 = concat(df_intermed_2,df4)
    return df_intermed_3

def separating(df_1,df_2):
    df_train = df_1[df_1["Date"]<= (max(df_2["Date"])- timedelta(days = 60))]
    df_test = df_1[df_1["Date"]> (max(df_2["Date"])- timedelta(days = 60))]
    return df_train, df_test

def return_calculation(df):
    result = df.Yields-df.RF
    return result

def fama_french(df):
    #x = np.column_stack((df["Mkt-RF"], df['SMB'], df['HML']))
    x = np.column_stack((df["MA_Rm-Rf"], df['MA_SMB'], df['MA_HML']))
    y = df['Returns']
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    return model, model.params['const'],model.params['x1'],model.params['x2'],model.params['x3']

def separation_X_y(df):
    y = df["Returns"]
    #X = df.drop(["Returns","Yields","Price","Mkt-RF","SMB","HML"],axis = 1)
    X = df.drop(["Returns","Yields","Price","MA_Rm-Rf","MA_SMB","MA_HML"],axis = 1)
    return y, X

def prediction_ff(df,c,b1,b2,b3):
    preds = b1*df["Mkt-RF"]+b2*df["SMB"]+b3*df["HML"]+c
    #preds = b1*df["MA_Rm-Rf"]+b2*df["MA_SMB"]+b3*df["MA_HML"]+c
    preds_df = pd.DataFrame(preds,columns = ["Predicted_Returns"])
    df = concat(df,preds_df)
    df["Predicted_Yields"]= df["Predicted_Returns"]+df["RF"]
    return preds,df

def MAE(y1,y2):
    mae = metrics.mean_absolute_error(y1,y2)
    return mae

def comparative_graph(y1,df):
    name = str(input("Stock Name: "))
    graph_name = "Chart_Comparative_"+name+".png"
    fig = plt.figure()
    plt.title("Comparison Reality v Prediction : Fama and French Model")
    plt.plot(df["Date"], (y1+df["RF"])*100, color = 'blue', label = 'Reality')
    plt.plot(df["Date"], df["Predicted_Yields"]*100, color = 'orange', label = 'Prediction')
    plt.legend(loc = "upper left")
    plt.xlabel("Date")
    plt.ylabel("Yields in %")
    #plt.rc('figure', figsize=[20, 20])
    fig.savefig(graph_name)

def export_results(df,list_y,c,b1,b2,b3):
    df_real = pd.DataFrame(list(list_y),columns = ["Real_Returns"])
    df = concat(df,df_real)
    df["Real_Yields"]=df["Real_Returns"]+df["RF"]
    name = str(input("Stock Name: "))
    name_file = "Results_"+name+".xlsx"
    df.to_excel(name_file)
    dict_temp = dict({"β1":x1,"β2":x2,"β3":x3,"ε":c})
    df_ff_factors = pd.DataFrame(list(dict_temp.items()),columns = ["Name","Value"])
    factors_name = "Fama_&_French_Factors_"+name+".xlsx"
    df_ff_factors.to_excel(factors_name)

# Main 
df_specific = preprocess_financial_data("apple.csv","VIX.csv")
df_price = extract_price(df_specific)
df_ff = open_ff("F&F.csv")
df_final = merging(df_price,df_ff)
df_final = removal_2021(df_final)
fill_na(df_final)
df_final = reset_index(df_final)
yields_list= [0.0,]
i = 0
while i < len(df_final["Price"])-1:
    yields = (df_final["Price"][i+1]-df_final["Price"][i])/df_final["Price"][i]
    yields_list.append(yields)
    i =i+1

yields = pd.DataFrame(yields_list, columns = ['Yields'])
df_final = concat(df_final, yields)
MA_RM_RF, MA_SMB, MA_HML = pandator_ma(df_final)
df_final = super_concatener(df_final,MA_RM_RF,MA_SMB,MA_HML)
df_train, df_test = separating(df_final,df_ff)
df_test = reset_index(df_test)
df_train["Returns"] = df_train.apply(lambda row: return_calculation(row), axis = 1)
df_test["Returns"] = df_test.apply(lambda row: return_calculation(row), axis = 1)
model_ff, const_ff, x1, x2, x3 = fama_french(df_train)
y_test,X_test = separation_X_y(df_test)
y_pred,df_pred = prediction_ff(X_test,const_ff,x1,x2,x3)
print("The MAE of the model is ",MAE(y_test,y_pred),". The average yield is ",np.median(y_test),".")
plt.figure()
plt.rc('figure', figsize=[20, 20])
final_graph = comparative_graph(y_test,df_pred)
export_results(df_pred,y_test,const_ff,x1,x2,x3)
