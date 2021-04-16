
from flask import Flask, request,render_template
import pickle
import numpy as np


model = pickle.load(open('heart_attack.pickle', 'rb'))


app = Flask(__name__)

@app.route('/')
def man():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def home():
    data1 = request.form["Age"]
    data2 = request.form['Mygender']
    data3 = request.form['Chestpain']
    data4 = request.form['bloodpressure']
    data5 = request.form['cholestrol']
    data6 = request.form['fbloodsugar']
    data7 = request.form['RestingECG']
    data8 = request.form['Heartrate']
    data9 = request.form['ExerciseAngina']
    data10 = request.form['Stdepression']
    data11 = request.form['Slope']
    data12 = request.form['Caa']
    data13 = request.form['thal']
    arr = np.array([[data1, data2, data3, data4,data5,data6,data7,data8,data9,data10,data11,data12,data13]])
    pred = model.predict(arr)
    return render_template('after.html', data=pred)



if __name__=="__main__":
    app.run(debug=True)
