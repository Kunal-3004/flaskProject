import numpy as np
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
app.config["DEBUG"] = True

# Load the pre-trained model
model1 = joblib.load(open('Models/heart_LR.pkl', 'rb'))

@app.route('/quick', methods=['GET', 'POST'])
def quick():
    try:
        data = request.json
        app.logger.info(f"Received data: {data}")

        chest_pain_type = data.get('chestPainType')
        heart_rate = data.get('heartRate')
        exang = data.get('exang')
        oldpeak = data.get('oldPeak')
        ca = data.get('ca')
        thalassemia = data.get('thalassemia')

        if any(param is None for param in [chest_pain_type, heart_rate, exang, oldpeak, ca, thalassemia]):
            return jsonify({'error': 'One or more input features are missing'}), 400

        try:
            features = [
                int(chest_pain_type),
                int(heart_rate),
                int(exang),
                float(oldpeak),
                int(ca),
                int(thalassemia)
            ]
        except ValueError as e:
            return jsonify({'error': f'Invalid input types: {str(e)}'}), 400

        app.logger.info(f"Features before prediction: {features}")
        final_features = np.array(features).reshape(1, -1)
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
