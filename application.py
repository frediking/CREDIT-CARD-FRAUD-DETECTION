# Standard library imports
import warnings
warnings.filterwarnings('ignore')

# Third-party imports
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)

# Load model and original scaler
try:
    base_dir = "/Users/macm1pro/Desktop/CC FRAUD DETECTION"
    model = joblib.load(f'{base_dir}/models/saved models/rfcc_model2.pkl')
    original_scaler = joblib.load(f'{base_dir}/models/scalerrf.pkl')
except FileNotFoundError as e:
    print(f"Error: Could not find {str(e).split()[-1]}")
    print(f"Make sure the .pkl files are in: {base_dir}/models/")
    exit(1)

# Create a new scaler for the 10 features expected by the model
if original_scaler.n_features_in_ > model.n_features_in_:
    n = model.n_features_in_  # Number of features the model expects (10)
    mean_ = original_scaler.mean_[:n]  # Means for the first 10 features
    scale_ = original_scaler.scale_[:n]  # Scales for the first 10 features
    var_ = original_scaler.var_[:n] if hasattr(original_scaler, 'var_') else None  # Variance, if available

    # Create and configure the new scaler
    feature_scaler = StandardScaler()
    feature_scaler.mean_ = mean_
    feature_scaler.scale_ = scale_
    if var_ is not None:
        feature_scaler.var_ = var_
    feature_scaler.n_features_in_ = n
else:
    feature_scaler = original_scaler  # Use the original scaler if no feature mismatch

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({'error': 'No features provided'}), 400

        # Convert features to numpy array
        features_full = np.array(data['features'], dtype=float)

        # Validate that exactly 30 features are provided
        if len(features_full) != 30:
            return jsonify({
                'error': f'Expected 30 features, got {len(features_full)}'
            }), 400

        # Select the first 10 features (assuming model was trained on these)
        n = model.n_features_in_  # Should be 10
        features = features_full[:n].reshape(1, -1)

        # Check for non-finite values
        if not np.all(np.isfinite(features)):
            invalid_indices = np.where(~np.isfinite(features))[1].tolist()
            return jsonify({
                'error': f'Invalid values at positions: {invalid_indices}. All values must be finite numbers.'
            }), 400

        # Scale the selected features using the new scaler
        try:
            features_scaled = feature_scaler.transform(features)
        except ValueError as ve:
            return jsonify({
                'error': f'Scaling failed: {str(ve)}. Ensure all values are valid numbers.'
            }), 400

        # Make prediction with probability
        pred_proba = model.predict_proba(features_scaled)[:, 1]
        prediction = (pred_proba >= 0.53).astype(int)[0]

        return jsonify({
            'prediction': int(prediction),
            'probability': float(pred_proba[0])
        })

    except ValueError as ve:
        return jsonify({
            'error': f'Input processing failed: {str(ve)}. Provide 30 comma-separated numeric values.'
        }), 400
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000)