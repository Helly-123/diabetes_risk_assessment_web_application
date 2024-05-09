import requests
import configparser
from flask import Flask, render_template, request
import pickle
from model.model import predict_and_calculate_probability

app = Flask(__name__)

# Loading the trained model
with open("C:\\Users\\Public\\Downloads\\organized_deployment\\model\\model.pkl", 'rb') as f:
    model = pickle.load(f)


# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')


# Route for handling form submission and making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Retrieval of form data
    pregnancies = float(request.form['Pregnancies'])
    bmi = float(request.form['BMI'])
    age = float(request.form['Age'])

    prediction, prob_of_diabetes_given_input = predict_and_calculate_probability([[pregnancies, bmi, age]])

    if prediction[0] == 1:
        result = f"You may be at risk of developing the disease. We recommend consulting a doctor for further evaluation. Probability of having diabetes is {prob_of_diabetes_given_input:.2%}."
    else:
        result = f"No need to worry! You currently show no signs of the disease. However, it's important to maintain a healthy lifestyle. "

    return render_template('result.html', prediction=result, prob_of_diabetes_given_input=prob_of_diabetes_given_input)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)


