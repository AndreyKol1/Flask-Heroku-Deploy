from flask import Flask, render_template, request, make_response, jsonify
import pickle
import numpy as np

app = Flask(__name__, template_folder='templates', static_folder='static')

model = pickle.load(open("loan_prediction.pkl", "rb"))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods =["POST"])
def predict():

    values = [x for x in request.form.values()]

  
    gender = values[0]  
    married = values[1]  
    dependents = int(values[2])  
    education = values[3]  
    self_employed = values[4]  
    applicant_income = float(values[5])
    coapplicant_income = float(values[6])
    loan_amount = float(values[7])
    loan_amount_term = float(values[8])
    credit_history = float(values[9])
    property_area = values[10]  


    gender_male = 1 if gender == "Male" else 0
    married_yes = 1 if married == "Yes" else 0
    education_not_graduate = 1 if education == "Not Graduate" else 0
    self_employed_yes = 1 if self_employed == "Yes" else 0
    property_area_semiurban = 1 if property_area == "Semiurban" else 0
    property_area_urban = 1 if property_area == "Urban" else 0


    features = np.array([
        dependents,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_amount_term,
        credit_history,
        gender_male,
        married_yes,
        education_not_graduate,
        self_employed_yes,
        property_area_semiurban,
        property_area_urban
    ]).reshape(1, -1)


    prediction = model.predict(features)[0]


    return render_template("index.html", prediction_text=f"The result of loan commission is {'Approved' if prediction == 1 else 'Rejected'}")


        
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)