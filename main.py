# Importing the required Packages
from flask import Flask, render_template, request, session, redirect,send_file
from flask_sqlalchemy import SQLAlchemy
import os
import json
from flask_mail import Mail
import pandas as pd
from werkzeug.utils import secure_filename
import time
import eda



app = Flask(__name__)

with open("config.json") as c:
    params = json.load(c)["params"]

df = pd.read_csv(params['upload_location']+'df_Clean.csv')
app.config['UPLOAD_FOLDER'] = params['upload_location']
@app.route("/")
def home():
    return render_template('index.html')

@app.route("/eda",methods = ["GET","POST"])
def eda():
    if request.method == "POST":
        product = request.form.get('column_name1')
        column1 = request.form.get('column_name2')
        column2 = request.form.get('column_name3')
        if(product != "TV_&_AC"):
            if(product == "TV"):
                out = df.loc[df['Product_type']=='TV']
            if(product == "AC"):
                out = df.loc[df['Product_type'] == 'AC']
        else:
            out = df
        import eda
        if (column2 == 'Variable2' and column1 == 'Variable1') or (column2 != 'Variable2' and column1 == 'Variable1'):
            A = pd.DataFrame()
            msg = "Please Enter value for Column 1"
        elif (column2 == 'Variable2' and column1 != 'Variable1') or column2==column1 :
            A,msg = eda.Univ_analysis(out, column1)
        else:
            A,msg = eda.biv_analysis(out, column1,column2)

    else:
        A = df.head()
        msg = f'Showing the top 5 records of the data set'
    return render_template('eda.html',col = df.columns[1:],data = A,len=len(A),msg=msg)

@app.route("/model")
def model():
    available_data = os.listdir(params["upload_location"])
    answer = pd.DataFrame()
    return render_template('model.html', out=answer, avil=available_data, len=len(answer))


@app.route("/uploader", methods = ["GET","POST"])
def uploader():
    available_data = os.listdir(params["upload_location"])
    if (request.method == "POST"):
        f = request.files['file1']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename("Test.csv")))
        m = "Your data is uploaded as name Test.csv"
    else:
        m = "Kindly Upload Your Evaluation data"
    answer = pd.DataFrame()
    return render_template('model.html',out = answer,avil = available_data,len=len(answer),msg = m,messg = "")


@app.route("/model_Eval",methods = ['GET','POST'])
def answer():
    if (request.method == 'POST'):
        available_data = os.listdir(params["upload_location"])
        train = pd.read_csv(params["upload_location"]+request.form.get('train_data'))
        test = pd.read_csv(params["upload_location"]+request.form.get('test_data'))
        if ('Fraud' not in test.columns):
            models = request.form.get('modelselection')
            import model
            global answer
            m,answer,_ = model.ModelSelection(train,models,test)
            _.to_excel(params["upload_location"]+"Prediction.xlsx")
        elif 'Fraud' not in train.columns:
            answer = pd.DataFrame()
            m = "Train Data must contain Fraud column/Variable"
        elif 'Fraud' in test.columns:
            answer = pd.DataFrame()
            m = "Evaluation Set Should not contain Fraud column/Variable"
        else:
            m= "check whether your test data contains the following columns or not  :/n'Region', 'State', 'Area', 'City', 'Consumer_profile', 'Product_category', 'Product_type', 'AC_1001_Issue','AC_1002_Issue', 'AC_1003_Issue', 'TV_2001_Issue', 'TV_2002_Issue', 'TV_2003_Issue', 'Claim_Value','Service_Centre', 'Product_Age', 'Purchased_from', 'Call_details', 'Purpose'"
    else:
        answer = pd.DataFrame()
        m=""
    return render_template('model.html',out = answer,avil = available_data,len=len(answer),msg="",messg = m)

@app.route("/download",methods = ['GET','POST'])
def download():
    return send_file(params["upload_location"]+"Prediction.xlsx")

app.run(debug = True)
