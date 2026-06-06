from flask import Flask,render_template,request
import numpy as np
import joblib
import os

my_dir=os.path.dirname(__file__)

pickle_file_path=os.path.join(my_dir,'knn_crop_model.pkl')

model=joblib.load(pickle_file_path)

#print(__name__)

app=Flask(__name__)#indicates where the execution has to strt from(app is object)

@app.route('/')

def index():

    return render_template('index.html')
@app.route('/predict',methods=['POST','GET'])
def predict():
    n=float(request.form['n'])
    p=float(request.form['p'])
    k=float(request.form['k'])
    temp=float(request.form['temp'])
    hum=float(request.form['hum'])
    ph=float(request.form['ph'])
    rain=float(request.form['rain'])

    test_data=np.array([[n,p,k,temp,hum,ph,rain]])

    pred=model.predict(test_data)

    result=pred[0]

    print(result)

    return render_template('home.html',res=result)

@app.route('/login',methods=['POST','GET'])
def login():

    uname=request.form['username']
    pwd=request.form['password']

    if uname=="admin" and pwd=="admin123":
        return render_template('home.html')
    else:
        return render_template('index.html',error="login failed")