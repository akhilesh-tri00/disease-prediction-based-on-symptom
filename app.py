import os
# Configure OpenBLAS / OpenMP threads to prevent Windows memory allocation errors
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__, static_folder='static', template_folder='templates')

# Load trained model and metadata
MODEL_PATH = 'disease_model.pkl'
METADATA_PATH = 'model_metadata.json'
DISEASE_INFO_PATH = 'disease_info.json'

model = None
metadata = {}
disease_info = {}

def load_resources():
    global model, metadata, disease_info
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
    if os.path.exists(DISEASE_INFO_PATH):
        with open(DISEASE_INFO_PATH, 'r') as f:
            disease_info = json.load(f)

load_resources()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/graphs/<path:filename>')
def serve_graph(filename):
    return send_from_directory('static/graphs', filename)

@app.route('/api/symptoms', methods=['GET'])
def get_symptoms():
    return jsonify({
        'total': metadata.get('total_symptoms', 0),
        'symptoms': metadata.get('symptoms', []),
        'categories': metadata.get('symptom_categories', {})
    })

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    return jsonify({
        'accuracy_benchmark': metadata.get('accuracy_benchmark', 0.976),
        'accuracy_validation': metadata.get('accuracy_validation', 1.0),
        'comparison': metadata.get('comparison', []),
        'total_diseases': metadata.get('total_diseases', 41),
        'total_symptoms': metadata.get('total_symptoms', 132),
        'graphs': [
            {
                'id': 'feature_importance',
                'title': 'Top 20 Most Predictive Symptoms',
                'description': 'Measures the Gini Importance of individual symptoms calculated by the Random Forest ensemble.',
                'image': '/static/graphs/feature_importance.png'
            },
            {
                'id': 'confusion_matrix',
                'title': 'Testing Dataset Confusion Matrix',
                'description': 'Evaluates classification correctness across all 41 disease classes on the unseen Testing benchmark.',
                'image': '/static/graphs/confusion_matrix.png'
            },
            {
                'id': 'model_comparison',
                'title': 'Multi-Model Benchmark Comparison',
                'description': 'Compares Accuracy, Precision, Recall, and F1-Score across 5 distinct Machine Learning algorithms.',
                'image': '/static/graphs/model_comparison.png'
            },
            {
                'id': 'symptom_distribution',
                'title': 'Symptoms per Patient Distribution',
                'description': 'Shows the statistical frequency of active symptoms reported per medical case.',
                'image': '/static/graphs/symptom_distribution.png'
            },
            {
                'id': 'disease_distribution',
                'title': 'Disease Class Balance',
                'description': 'Validates uniform class representation across all 41 diagnostic categories.',
                'image': '/static/graphs/disease_distribution.png'
            }
        ]
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Trained model is not loaded. Please train first.'}), 500

    data = request.get_json() or {}
    selected_symptoms = data.get('symptoms', [])

    if not selected_symptoms:
        return jsonify({'error': 'Please select at least one symptom to make a diagnosis.'}), 400

    all_symptoms = metadata.get('symptoms', [])
    if not all_symptoms and hasattr(model, 'feature_names_in_'):
        all_symptoms = list(model.feature_names_in_)

    # Vectorize input
    feature_vector = {symptom: (1 if symptom in selected_symptoms else 0) for symptom in all_symptoms}
    input_df = pd.DataFrame([feature_vector])

    # Run inference
    try:
        probabilities = model.predict_proba(input_df)[0]
        classes = model.classes_
        
        # Sort predictions descending
        ranked_indices = np.argsort(probabilities)[::-1]
        
        top_candidates = []
        for idx in ranked_indices[:6]:
            prob = float(probabilities[idx])
            disease_name = str(classes[idx])
            
            # Retrieve clinical knowledge
            info = disease_info.get(disease_name.strip(), disease_info.get(disease_name, {
                'description': 'No clinical description available.',
                'specialist': 'General Physician',
                'precautions': ['Consult a licensed medical doctor for tailored diagnosis.'],
                'severity': 'Moderate'
            }))
            
            top_candidates.append({
                'disease': disease_name,
                'probability': round(prob * 100, 2),
                'specialist': info.get('specialist', 'General Physician'),
                'severity': info.get('severity', 'Moderate'),
                'description': info.get('description', ''),
                'precautions': info.get('precautions', [])
            })

        primary = top_candidates[0]

        return jsonify({
            'success': True,
            'primary_prediction': primary['disease'],
            'confidence': primary['probability'],
            'severity': primary['severity'],
            'specialist': primary['specialist'],
            'description': primary['description'],
            'precautions': primary['precautions'],
            'top_candidates': top_candidates,
            'input_symptoms_count': len(selected_symptoms)
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/diseases', methods=['GET'])
def get_all_diseases():
    return jsonify(disease_info)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n==================================================")
    print(f" MediPredict AI Web Application Server")
    print(f" Serving locally at: http://127.0.0.1:{port}")
    print(f"==================================================\n")
    app.run(host='0.0.0.0', port=port, debug=False)
