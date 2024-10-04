import numpy as np
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

model1 = pickle.load(open('Models/heart_LR.pkl', 'rb'))


def bmi(height, weight):
    return round(weight / ((height / 100) ** 2), 2)


@app.route('/quick', methods=['GET', 'POST'])
def quick():
    try:
        data = request.json
        app.logger.info(f"Received data: {data}")  # Log received data

        sex = data.get('SEX')
        age = data.get('age')
        smoker = data.get('Current Smoker')
        cp = data.get('Chest Pain Type')
        stroke = data.get('Stroke')
        bp_meds = data.get('BP medication')
        diabetes = data.get('Diabetes')
        height = data.get('Height')
        weight = data.get('Weight')
        hrv = data.get('Heart Rate')
        exang = data.get('Exang')
        oldpeak = data.get('Oldpeak')
        ca = data.get('ca')
        thal = data.get('thalassemia')

        # Input validation
        if height <= 0 or weight <= 0:
            return jsonify({'error': 'Height and weight must be greater than zero'}), 400

        # Calculate BMI
        bmi_value = bmi(height, weight)

        # Prepare the feature list
        features = [cp, hrv, exang, oldpeak, ca, thal]

        # Check the shape of final_features
        final_features = [np.array(features)]
        prediction = model1.predict(final_features)

        result = "No need to worry" if prediction[
                                           0] == 0 else "You are detected with heart problems. You need to consult a doctor immediately"

        return jsonify({
            'prediction': result,
            'gender': 'Male' if sex == 1 else 'Female',
            'age': age,
            'smoking': 'Yes' if smoker == 1 else 'No',
            'stroke': 'Yes' if stroke == 1 else 'No',
            'exang': 'Yes' if exang == 1 else 'No',
            'bp_meds': 'Yes' if bp_meds == 1 else 'No',
            'diabetes': 'Yes' if diabetes == 1 else 'No',
            'bmi': bmi_value,
            'hrv': hrv
        })

    except Exception as e:
        app.logger.error(f"Error occurred: {str(e)}")  # Log error
        return jsonify({'error': str(e)}), 500


@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Heart Disease Prediction API!"})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
