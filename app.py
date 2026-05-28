from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

print("Loading model...")
model = joblib.load('model/gesture_model.pkl')
print("Model loaded!")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'running'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        landmarks = data['landmarks']
        features = np.array(landmarks).reshape(1, -1)
        prediction = model.predict(features)[0]
        confidence = float(max(model.predict_proba(features)[0]))
        return jsonify({
            'gesture': prediction,
            'confidence': round(confidence * 100, 1)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')