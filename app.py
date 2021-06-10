# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 21:00:41 2021

@author: admin
"""

from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__,template_folder='template')
model = pickle.load(open('random_forest_regression.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    #Fuel_Type_Diesel=0
    if request.method == 'POST':
        no_years = int(request.form['no_years'])
        km_driven=int(request.form['km_driven'])
        mileage=int(request.form['mileage'])
        Owner=int(request.form['Owner'])
        if(Owner==1):
            owner_Fourth=0
            owner_Second=0
            owner_Third=0
        elif(Owner==2):
            owner_Fourth=0
            owner_Second=1
            owner_Third=0
        elif(Owner==3):
            owner_Fourth=0
            owner_Second=0
            owner_Third=1
        else:
            owner_Fourth=1
            owner_Second=0
            owner_Third=0
        engine = int(request.form['engine'])
        max_power = int(request.form['max_power'])
        seats = int(request.form['seats'])
        fuel_Petrol=request.form['fuel_Petrol']
        if(fuel_Petrol=='Petrol'):
                fuel_Petrol=1
                fuel_Diesel=0
                fuel_LPG=0
        elif(fuel_Petrol=='Diesel'):
            fuel_Petrol=0
            fuel_Diesel=1
            fuel_LPG=0
        elif(fuel_Petrol=='LPG'):
            fuel_Petrol=0
            fuel_Diesel=0
            fuel_LPG=1
        else:
            fuel_Petrol=0
            fuel_Diesel=0
            fuel_LPG=0
            
            
        no_years=2021-no_years
        seller_type_Individual=request.form['seller_type_Individual']
        if(seller_type_Individual=='Individual'):
            seller_type_Individual=1
            seller_type_Trustmark_Dealer=0
        elif(seller_type_Individual=='Trustmark Dealer'):
            seller_type_Individual=0	
            seller_type_Trustmark_Dealer=1
        else:
            seller_type_Individual=0	
            seller_type_Trustmark_Dealer=0
            
        transmission_Manual=request.form['transmission_Manual']
        if(transmission_Manual=='Mannual'):
            transmission_Manual=1
        else:
            transmission_Manual=0
        prediction=model.predict([[km_driven, mileage, engine, max_power, seats, no_years,
       fuel_Diesel, fuel_LPG, fuel_Petrol, seller_type_Individual,
       seller_type_Trustmark_Dealer, transmission_Manual,
       owner_Fourth, owner_Second,
       owner_Third]])
        output=round(prediction[0],2)
        if output<0:
            return render_template('index.html',prediction_texts="Sorry you cannot sell this car")
        else:
            return render_template('index.html',prediction_text="You Can Sell The Car at {}".format(output))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)