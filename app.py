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

        features = [chest_pain_type, heart_rate, exang, oldpeak, ca, thalassemia]


        final_features = np.array(features).reshape(1, -1)
        prediction = model1.predict(final_features)

        result = "No need to worry" if prediction[0] == 0 else "You are detected with heart problems. You need to consult a doctor immediately"

        return jsonify({
            'prediction': result,
            'features': features
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Heart Disease Prediction API!"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
