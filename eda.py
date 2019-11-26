# Import Required Packages
import pandas as pd
import os


#df = pd.read_csv('D:\\STUDY PROCESS\\ExcelR Project\\Project9_ExcelR\\static\\data\\df_Clean.csv')
#print(df.head())

def Data_Description(df):
    return df.head()
def Univ_analysis(df,col):
    coll = ['Claim_Value','Product_Age','Call_details']
    if col not in coll:
        uni = pd.DataFrame(pd.crosstab(df[col],df['Fraud'])).reset_index()
        uni['No of Customers'] = uni.sum(1)
        uni['Percentage of Fraud'] = round(uni[1]/uni['No of Customers'] * 100,2)
        uni['Percentage of Fraud'] = ['%d %%'%(i) for i in uni['Percentage of Fraud']]
        msg = '%s with percentage of Fraud'%(col)
    else:
        uni = pd.DataFrame(df[col].describe()).reset_index()
        msg = '%s Description'%(col)
    return uni,msg

def biv_analysis(df,col1,col2):
    coll = ['Claim_Value', 'Product_Age', 'Call_details']
    if col1 not in coll and col2 not in coll:
        ans = pd.DataFrame(pd.crosstab(df[col1],df[col2])).reset_index()
        msg = "Cross Table between %s and %s"%(col1,col2)
    elif col1 not in coll and col2 in coll:
        bi = pd.DataFrame(df[col2].groupby(df[col1]).mean()).reset_index()
        bi[col2] = [round(i) for i in bi[col2]]
        ans =  bi.sort_values(col2,ascending=False)
        msg = "Cross Table between %s and mean of %s"%(col1,col2)
    elif col1 in coll and col2 not in coll:
        bi = pd.DataFrame(df[col1].groupby(df[col2]).mean()).reset_index()
        bi[col1] = [ round(i)  for i in bi[col1]]
        ans = bi.sort_values(col1,ascending=False)
        msg = "Cross Table between %s and mean of %s"%(col2,col1)
    elif col1 in coll and col2 in coll:
        ans = pd.DataFrame(df[[col1, col2]].corr()).reset_index()
        msg = "Correlation between %s and %s"%(col1,col2)
    return ans,msg
