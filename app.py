import numpy as np
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model1 = joblib.load(open('Models/heart_LR.pkl','rb'))

@app.route('/quick', methods=['GET', 'POST'])
def quick():
    try:
        data = request.json
        app.logger.info(f"Received data: {data}")

        features = [
            data.get('Chest Pain Type'),
            data.get('Heart Rate'),
            data.get('Exang'),
            data.get('Oldpeak'),
            data.get('ca'),
            data.get('thalassemia')
        ]


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
