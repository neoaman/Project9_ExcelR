import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.utils import resample
import numpy as np
from sklearn.ensemble import RandomForestClassifier
try:
    from catboost import CatBoostClassifier
except:
    ""

from xgboost import XGBClassifier

# test = pd.read_csv("D:\\STUDY PROCESS\\ExcelR Project\\test.csv")
# df = pd.read_csv('D:\\STUDY PROCESS\\ExcelR Project\\Project9_ExcelR\\static\\data\\df_Clean.csv')

def data_clean(data):
    columns = ['Product_Age', 'Call_details', 'Claim_Value']
    for i in data.columns:
        if i not in columns:
            data[i] = data[i].astype('category')
    for i in data.columns:
        if data[i].dtype.name == 'category':
            enco = preprocessing.LabelEncoder()
            enco.fit(list(set(data[i])))
            data[i] = enco.transform(data[i])
    return data

def data_cat(data):
    columns = ['Product_Age', 'Call_details', 'Claim_Value']
    for i in data.columns:
        if i not in columns:
            data[i] = data[i].astype('category')
    return data


def DT(data0,testdata):
    data = data_clean(data0)
    test = pd.DataFrame.copy(testdata)
    test = data_clean(test)
    df_majority = data[data.Fraud == 0]
    df_minority = data[data.Fraud == 1]
    df_minority_upsampled = resample(df_minority,
                                     replace=True,  # sample with replacement
                                     n_samples=300,  # to match majority class
                                     random_state=123)  # reproducible results
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    X = pd.DataFrame.copy(df_upsampled)
    X = X[['Region', 'State', 'Area', 'City', 'Consumer_profile','Product_category', 'Product_type', 'AC_1001_Issue', 'AC_1002_Issue','AC_1003_Issue', 'TV_2001_Issue', 'TV_2002_Issue', 'TV_2003_Issue','Claim_Value', 'Service_Centre', 'Product_Age', 'Purchased_from','Call_details', 'Purpose', 'Fraud']]
    Y = X.pop('Fraud')
    test = test[['Region', 'State', 'Area', 'City', 'Consumer_profile','Product_category', 'Product_type', 'AC_1001_Issue', 'AC_1002_Issue','AC_1003_Issue', 'TV_2001_Issue', 'TV_2002_Issue', 'TV_2003_Issue','Claim_Value', 'Service_Centre', 'Product_Age', 'Purchased_from','Call_details', 'Purpose']]
    Warrenty_Tree = DecisionTreeClassifier(criterion='entropy', max_depth=9, min_samples_leaf=2, random_state=21,splitter='random')
    Warrenty_Tree.fit(X, Y)
    pred = pd.DataFrame(Warrenty_Tree.predict(test),columns=['Fraud'])
    pred2 = pd.DataFrame(Warrenty_Tree.predict_proba(test), columns=['P_Fraud_No', 'P_Fraud_Yes'])
    pred2['P_Fraud_No'] = round(pred2.P_Fraud_No, 2)
    pred2['P_Fraud_Yes'] = round(pred2.P_Fraud_Yes, 2)
    output = pd.concat([testdata, pred, pred2], axis=1)
    msg = "As per our Decission True Model There are %d Fraud cases in the data"%(len(output.loc[output.Fraud==1]))
    lol = output.loc[output.Fraud == 1,[output.columns[0],'State', 'Area', 'City', 'Consumer_profile','Product_type','Fraud','P_Fraud_Yes']].reset_index()
    lol.pop('index')
    return msg,lol,output


def RF(data0,testdata):
    data = data_clean(data0)
    test = pd.DataFrame.copy(testdata)
    test = data_clean(test)
    df_majority = data[data.Fraud == 0]
    df_minority = data[data.Fraud == 1]
    df_minority_upsampled = resample(df_minority,
                                     replace=True,  # sample with replacement
                                     n_samples=300,  # to match majority class
                                     random_state=123)  # reproducible results
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    X = pd.DataFrame.copy(df_upsampled)
    X = X[['Region', 'State', 'Area', 'City', 'Consumer_profile', 'Product_category', 'Product_type', 'AC_1001_Issue',
           'AC_1002_Issue', 'AC_1003_Issue', 'TV_2001_Issue', 'TV_2002_Issue', 'TV_2003_Issue', 'Claim_Value',
           'Service_Centre', 'Product_Age', 'Purchased_from', 'Call_details', 'Purpose', 'Fraud']]
    Y = X.pop('Fraud')
    test = test[
        ['Region', 'State', 'Area', 'City', 'Consumer_profile', 'Product_category', 'Product_type', 'AC_1001_Issue',
         'AC_1002_Issue', 'AC_1003_Issue', 'TV_2001_Issue', 'TV_2002_Issue', 'TV_2003_Issue', 'Claim_Value',
         'Service_Centre', 'Product_Age', 'Purchased_from', 'Call_details', 'Purpose']]
    Warrenty_forest = RandomForestClassifier(n_estimators=12,criterion='gini',max_depth=9,min_samples_leaf=2,max_features='auto',random_state=18)
    Warrenty_forest.fit(X, Y)
    pred = pd.DataFrame(Warrenty_forest.predict(test), columns=['Fraud'])
    pred2 = pd.DataFrame(Warrenty_forest.predict_proba(test), columns=['P_Fraud_No', 'P_Fraud_Yes'])
    pred2['P_Fraud_No'] = round(pred2.P_Fraud_No, 2)
    pred2['P_Fraud_Yes'] = round(pred2.P_Fraud_Yes, 2)
    output = pd.concat([testdata, pred, pred2], axis=1)
    msg = "As per our Random Forest Model , may be there are %d Fraud cases in the data set" %(len(output.loc[output.Fraud == 1]))
    lol = output.loc[output.Fraud == 1,[output.columns[0],'State', 'Area', 'City', 'Consumer_profile','Product_type','Fraud','P_Fraud_Yes']].reset_index()
    lol.pop('index')
    return msg, lol,output
# dtree rf
try:
    def CatBoost(data0,testdata):
        data = data_cat(data0)
        test = pd.DataFrame.copy(testdata)
        test = data_cat(test)
        df_majority = data[data.Fraud == 0]
        df_minority = data[data.Fraud == 1]
        df_minority_upsampled = resample(df_minority,
                                         replace=True,  # sample with replacement
                                         n_samples=300,  # to match majority class
                                         random_state=123)  # reproducible results
        df_upsampled = pd.concat([df_majority, df_minority_upsampled])
        X = pd.DataFrame.copy(df_upsampled)
        X = X[['Region', 'State', 'Area', 'City', 'Consumer_profile', 'Product_category', 'Product_type', 'AC_1001_Issue',
               'AC_1002_Issue', 'AC_1003_Issue', 'TV_2001_Issue', 'TV_2002_Issue', 'TV_2003_Issue', 'Claim_Value',
               'Service_Centre', 'Product_Age', 'Purchased_from', 'Call_details', 'Purpose', 'Fraud']]
        Y = X.pop('Fraud')
        test = test[
            ['Region', 'State', 'Area', 'City', 'Consumer_profile', 'Product_category', 'Product_type', 'AC_1001_Issue',
             'AC_1002_Issue', 'AC_1003_Issue', 'TV_2001_Issue', 'TV_2002_Issue', 'TV_2003_Issue', 'Claim_Value',
             'Service_Centre', 'Product_Age', 'Purchased_from', 'Call_details', 'Purpose']]
        modelcat = CatBoostClassifier(learning_rate=0.1, depth=3, n_estimators=400,
                                      cat_features=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18])
        modelcat.fit(X, Y)
        pred = pd.DataFrame(modelcat.predict(test), columns=['Fraud'])
        pred2 = pd.DataFrame(modelcat.predict_proba(test),columns=['P_Fraud_No','P_Fraud_Yes'])
        pred2['P_Fraud_No'] = round(pred2.P_Fraud_No,2)
        pred2['P_Fraud_Yes'] = round(pred2.P_Fraud_Yes,2)
        output = pd.concat([testdata, pred,pred2], axis=1)
        msg = "As per our Cat Boost model, may be there are %d Fraud cases in the data set" % (len(output.loc[output.Fraud == 1]))
        lol = output.loc[output.Fraud == 1,[output.columns[0],'State', 'Area', 'City', 'Consumer_profile','Product_type','Fraud','P_Fraud_Yes']].reset_index()
        lol.pop('index')
        return msg, lol, output
except:
    ""
def XGBoost(data0,testdata):
    data = data_clean(data0)
    test = pd.DataFrame.copy(testdata)
    test = data_clean(test)
    df_majority = data[data.Fraud == 0]
    df_minority = data[data.Fraud == 1]
    df_minority_upsampled = resample(df_minority,
                                     replace=True,  # sample with replacement
                                     n_samples=300,  # to match majority class
                                     random_state=123)  # reproducible results
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    X = pd.DataFrame.copy(df_upsampled)
    X = X[['Region', 'State', 'Area', 'City', 'Consumer_profile', 'Product_category', 'Product_type', 'AC_1001_Issue',
           'AC_1002_Issue', 'AC_1003_Issue', 'TV_2001_Issue', 'TV_2002_Issue', 'TV_2003_Issue', 'Claim_Value',
           'Service_Centre', 'Product_Age', 'Purchased_from', 'Call_details', 'Purpose', 'Fraud']]
    Y = X.pop('Fraud')
    test = test[
        ['Region', 'State', 'Area', 'City', 'Consumer_profile', 'Product_category', 'Product_type', 'AC_1001_Issue',
         'AC_1002_Issue', 'AC_1003_Issue', 'TV_2001_Issue', 'TV_2002_Issue', 'TV_2003_Issue', 'Claim_Value',
         'Service_Centre', 'Product_Age', 'Purchased_from', 'Call_details', 'Purpose']]
    xgm = XGBClassifier(max_depth=5,learning_rate=0.2,n_estimators=200)
    xgm.fit(X, Y)
    pred = pd.DataFrame(xgm.predict(test), columns=['Fraud'])
    pred2 = pd.DataFrame(xgm.predict_proba(test), columns=['P_Fraud_No', 'P_Fraud_Yes'])
    pred2['P_Fraud_No'] = round(pred2.P_Fraud_No, 2)
    pred2['P_Fraud_Yes'] = round(pred2.P_Fraud_Yes, 2)
    output = pd.concat([testdata, pred, pred2], axis=1)
    msg = "As per our XG Boost model, may be there are %d Fraud cases in the data set" % (len(output.loc[output.Fraud == 1]))
    lol = output.loc[output.Fraud == 1,[output.columns[0],'State', 'Area', 'City', 'Consumer_profile','Product_type','Fraud','P_Fraud_Yes']].reset_index()
    lol.pop('index')
    return msg, lol, output


def ModelSelection(data,model,testdata):
    if model == 'DecissionTree':
        msg,out,_ = DT(data,testdata)
    if model == 'RandomForest':
        msg,out,_ = RF(data,testdata)
    try:
        if model == 'CatBoost':
            msg, out, _ = CatBoost(data, testdata)
    except:
        msg, out, _ = XGBoost(data, testdata)
    if model == 'XGBoost':
        msg, out, _ = XGBoost(data, testdata)


    return msg,out,_