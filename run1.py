# Importing the required Packages
from flask import Flask, render_template, request, session, redirect

import pandas as pd
import eda
import os
from werkzeug.utils import secure_filename
import json

df = pd.read_csv('/home/neoml/Project9_ExcelR/static/data/df_Clean.csv')

app = Flask(__name__)

with open("/home/neoml/Project9_ExcelR/config.json") as c:
    params = json.load(c)["params"]
app.config['UPLOAD_FOLDER'] = params['upload_location2']


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
        msg = 'Showing the top 5 records of the data set'
    return render_template('eda.html',col = df.columns[1:],data = A,len=len(A),msg=msg)

@app.route("/model")
def model():
    return render_template('model.html')

@app.route("/uploader", methods = ["GET","POST"])
def uploader():
    if (request.method == "POST"):
        f = request.files['file1']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename("train.csv")))
        A = 'upload Successfull'
        return A

