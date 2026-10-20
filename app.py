from flask import Flask, request, render_template
import joblib
from preprocess import preprocess_input

# Start Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('model/model.joblib')

# Home page that shows the form
@app.route('/')
def home():
    return render_template('index.html')  # HTML form is here

# Route that handles prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    pclass = int(request.form['pclass'])
    sex = request.form['sex']
    age = float(request.form['age'])

    # Preprocess the input
    input_values = preprocess_input(pclass, sex, age)

    # Make prediction
    prediction = model.predict(input_values)[0]

    # Return result to the user
    return render_template('index.html',
        prediction_text=f'Survival Prediction: {"Survived" if prediction == 1 else "Did Not Survive"}')

# Run the app locally
if __name__ == '__main__':
    app.run(debug=False)
