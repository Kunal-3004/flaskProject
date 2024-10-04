import numpy as np
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
app.config["DEBUG"] = True

model1 = joblib.load(open('Models/heart_LR.pkl','rb'))

@app.route('/quick', methods=['GET', 'POST'])
def quick():
    try:
        data = request.json
        app.logger.info(f"Received data: {data}")

        chest_pain_type = data.get('Chest Pain Type')
        heart_rate = data.get('Heart Rate')
        exang = data.get('Exang')
        oldpeak = data.get('Oldpeak')
        ca = data.get('ca')
        thalassemia = data.get('thalassemia')

        if chest_pain_type is None or heart_rate is None or exang is None or oldpeak is None or ca is None or thalassemia is None:
            return jsonify({'error': 'One or more input features are missing'}), 400

        try:
            chest_pain_type = int(chest_pain_type)
            heart_rate = int(heart_rate)
            exang = int(exang)
            oldpeak = float(oldpeak)
            ca = int(ca)
            thalassemia = int(thalassemia)
        except ValueError as e:
            return jsonify({'error': 'Invalid input types: ' + str(e)}), 400

        features = [chest_pain_type, heart_rate, exang, oldpeak, ca, thalassemia]

        final_features = np.array(features).reshape(1, -1)
        app.logger.info(f"Features before prediction: {features}")
        prediction = model1.predict(final_features)

        result = "No need to worry" if prediction[0] == 0 else "You are detected with heart problems. You need to consult a doctor immediately"

        return jsonify({
            'prediction': result,
            'features': features
        })

    except Exception as e:
        app.logger.error(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500
@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Heart Disease Prediction API!"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
