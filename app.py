#import xgboost

import numpy as np
from flask import Flask, request,render_template
import pickle


app = Flask(__name__, template_folder='template')
model_bfp1 = pickle.load(open("BFP2_XGBOOST_MULTI_OUTPUT.pkl", "rb"))
scaler_bfp1 = pickle.load(open("BFP_2_XGBOOST_MULTIOUTPUT_STD_SCALER.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("Template_2.html")

@app.route("/predict", methods =["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    features = scaler_bfp1.transform(features)
    prediction = model_bfp1.predict(features)
    prediction = scaler_bfp1.inverse_transform(prediction)[0]
    p0 = prediction[0]
    p1 = prediction[1]
    p2 = prediction[2]
    p3 = prediction[3]
    p4 = prediction[4]
    p5 = prediction[5]
    p6 = prediction[6]
    p7 = prediction[7]
    p8 = prediction[8]
    p9 = prediction[9]
    p10 = prediction[10]
    p11 = prediction[11]
    p12 = prediction[12]
    p13 = prediction[13]


    return render_template("Template_2.html",p0=p0,p1=p1,p2=p2,p3=p3,p4=p4,p5=p5,p6=p6,p7=p7,p8=p8,p9=p9,p10=p10,p11=p11,p12=p12,p13=p13)

if __name__ == "__main__":
    app.run(debug=True)