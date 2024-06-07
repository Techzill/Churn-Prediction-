from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the XGBoost model and preprocessing encoders
xgb_model = joblib.load('xgboost_model.joblib')
scaler = joblib.load('scaler.joblib')
label_encoder_geography = joblib.load('label_encoder_geography.joblib')
label_encoder_gender = joblib.load('label_encoder_gender.joblib')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract form data
    customer_details = {
        'CreditScore': float(request.form['credit_score']),
        'Geography': request.form['geography'],
        'Gender': request.form['gender'],
        'Age': float(request.form['age']),
        'Tenure': float(request.form['tenure']),
        'Balance': float(request.form['balance']),
        'NumOfProducts': float(request.form['num_of_products']),
        'HasCrCard': float(request.form['has_cr_card']),
        'IsActiveMember': float(request.form['is_active_member']),
        'EstimatedSalary': float(request.form['estimated_salary'])
    }

    # Encode categorical features
    customer_details['Geography'] = label_encoder_geography.transform([customer_details['Geography']])[0]
    customer_details['Gender'] = label_encoder_gender.transform([customer_details['Gender']])[0]

    # Scale numerical features
    scaled_customer_details = scaler.transform([list(customer_details.values())])

    # Predict churn
    churn_prediction = xgb_model.predict(scaled_customer_details)
    churn_probability = xgb_model.predict_proba(scaled_customer_details)[:, 1]

    # Render prediction result
    return render_template('result.html', churn_prediction=int(churn_prediction[0]), churn_probability=churn_probability[0])

if __name__ == '__main__':
    app.run(debug=True)
