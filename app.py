import os
import json
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load saved models and artifacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, 'label_encoder.pkl')
ALL_MODELS_PATH = os.path.join(BASE_DIR, 'all_models.pkl')
METADATA_PATH = os.path.join(BASE_DIR, 'model_metadata.json')

scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)
models_dict = joblib.load(ALL_MODELS_PATH)

with open(METADATA_PATH, 'r') as f:
    metadata = json.load(f)

SPECIES_INFO = {
    'Setosa': {
        'name': 'Iris Setosa',
        'badge_color': 'bg-emerald-500',
        'border_color': 'border-emerald-500',
        'text_color': 'text-emerald-700',
        'bg_light': 'bg-emerald-50',
        'icon': '🌸',
        'description': 'Known for smaller petals and wider sepals. This species is easily linearly separable from the other two species.',
        'traits': [
            'Shortest petal length (average ~1.46 cm)',
            'Smallest petal width (average ~0.24 cm)',
            'Relatively wide sepals (average ~3.42 cm)',
            'Native to Alaska, northeastern North America, and Eastern Asia'
        ]
    },
    'Versicolor': {
        'name': 'Iris Versicolor',
        'badge_color': 'bg-blue-500',
        'border_color': 'border-blue-500',
        'text_color': 'text-blue-700',
        'bg_light': 'bg-blue-50',
        'icon': '🌺',
        'description': 'An intermediate species with moderate petal lengths and widths. Often called the Blue Flag iris.',
        'traits': [
            'Medium petal length (average ~4.26 cm)',
            'Medium petal width (average ~1.33 cm)',
            'Sepals are moderately sized (average ~5.94 cm length)',
            'Commonly found in wetland marshes across North America'
        ]
    },
    'Virginica': {
        'name': 'Iris Virginica',
        'badge_color': 'bg-purple-500',
        'border_color': 'border-purple-500',
        'text_color': 'text-purple-700',
        'bg_light': 'bg-purple-50',
        'icon': '🪻',
        'description': 'Characterized by the largest petals among the three species. Often overlaps slightly with Versicolor at boundaries.',
        'traits': [
            'Longest petal length (average ~5.55 cm)',
            'Widest petal width (average ~2.03 cm)',
            'Longest sepal length (average ~6.59 cm)',
            'Also known as the Virginia iris, native to eastern North America'
        ]
    }
}

@app.route('/')
def home():
    return render_template(
        'index.html',
        metadata=metadata,
        species_info=SPECIES_INFO,
        models=list(models_dict.keys())
    )

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        sepal_length = float(data.get('sepal_length', 5.1))
        sepal_width = float(data.get('sepal_width', 3.5))
        petal_length = float(data.get('petal_length', 1.4))
        petal_width = float(data.get('petal_width', 0.2))
        model_name = data.get('model_name', metadata.get('best_model', 'Support Vector Machine (SVM)'))

        if model_name not in models_dict:
            model_name = metadata.get('best_model')

        model = models_dict[model_name]

        # Prepare feature vector
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        features_scaled = scaler.transform(features)

        # Prediction
        pred_idx = model.predict(features_scaled)[0]
        predicted_class = label_encoder.inverse_transform([pred_idx])[0]

        # Class probabilities
        probabilities = {}
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(features_scaled)[0]
            for idx, cls in enumerate(label_encoder.classes_):
                probabilities[cls] = round(float(probs[idx]) * 100, 2)
        else:
            # If model doesn't support predict_proba
            for cls in label_encoder.classes_:
                probabilities[cls] = 100.0 if cls == predicted_class else 0.0

        # Confidence is probability of predicted class
        confidence = probabilities.get(predicted_class, 100.0)

        info = SPECIES_INFO.get(predicted_class, {})

        return jsonify({
            'status': 'success',
            'model_used': model_name,
            'input': {
                'sepal_length': sepal_length,
                'sepal_width': sepal_width,
                'petal_length': petal_length,
                'petal_width': petal_width
            },
            'predicted_species': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities,
            'species_info': info
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    return jsonify(metadata)

if __name__ == '__main__':
    print("Starting Iris Flower Classification Web Server on http://127.0.0.1:5000 ...")
    app.run(host='127.0.0.1', port=5000, debug=False)
