import numpy as np
import pandas as pd
from flask import Flask,render_template,request
import pickle
import warnings
warnings.filterwarnings("ignore")

with open("model.pkl",'rb') as f:
    m=pickle.load(f)
app=Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["GET", "POST"])
def prediction():

    a = [float(i) for i in request.form.values()]

    sol = m.predict([a])
    species={
        0:"iris-setosa",
        1:"iris-versicolor",
        2:"iris-verginica"
    }
    result=species[int(sol[0])]
    return render_template(
        "index.html",
        result=result
    )

if __name__=="__main__":
    app.run(debug=True)

