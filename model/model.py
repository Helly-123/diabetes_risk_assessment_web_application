#Importing relevant libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Loading the dataset
diabetes_data = pd.read_csv("C:\\Users\\Public\\Downloads\\organized_deployment\\model\\diabetes.csv")

# Selection of features-Pregnancies, BMI, and Age for diabetes risk prediction
X = diabetes_data[['Pregnancies', 'BMI', 'Age']]

#Selecting outcome as the target variable
y = diabetes_data['Outcome']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creation and training of random forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# Navigating to the project root directory
os.chdir("C:\\Users\\Public\\Downloads\\organized_deployment")

# Saving the trained model in the model subdirectory
with open('model/model.pkl', 'wb') as f:
    pickle.dump(model, f)


# Function to predict and calculate probability using Bayes Theorem
def predict_and_calculate_probability(input_data):
    # Make prediction
    prediction = model.predict(input_data)
    # Prior probability of having diabetes
    prior_diabetes = diabetes_data['Outcome'].mean()

    # Prediction of probability
    probability = model.predict_proba(input_data)
    prob_diabetes = probability[0][1]  # Probability of having diabetes

    # Application of Bayes' theorem
    prob_of_diabetes_given_input = (prob_diabetes * prior_diabetes) / (
    prob_diabetes * prior_diabetes + (1 - prob_diabetes) * (1 - prior_diabetes))

    return prediction, prob_of_diabetes_given_input