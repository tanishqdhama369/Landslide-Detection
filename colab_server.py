import numpy as np
import h5py
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from werkzeug.utils import secure_filename
import os
import tempfile

app = Flask(__name__)
CORS(app)

# Load the model
model = tf.keras.models.load_model('best_model.h5', compile=False)

def process_h5_file(file_path):
    with h5py.File(file_path) as hdf:
        data = np.array(hdf.get('img'))
        
        # Normalize data
        data[np.isnan(data)] = 0.000001
        
        # Calculate required values
        mid_rgb = data[:, :, 1:4].max() / 2.0
        mid_slope = data[:, :, 12].max() / 2.0
        mid_elevation = data[:, :, 13].max() / 2.0
        
        # NDVI calculation
        data_red = data[:, :, 3]
        data_nir = data[:, :, 7]
        data_ndvi = np.divide(data_nir - data_red, np.add(data_nir, data_red))
        
        # Prepare input array
        processed_data = np.zeros((1, 128, 128, 6))
        processed_data[0, :, :, 0] = 1 - data[:, :, 3] / mid_rgb  # RED
        processed_data[0, :, :, 1] = 1 - data[:, :, 2] / mid_rgb  # GREEN
        processed_data[0, :, :, 2] = 1 - data[:, :, 1] / mid_rgb  # BLUE
        processed_data[0, :, :, 3] = data_ndvi  # NDVI
        processed_data[0, :, :, 4] = 1 - data[:, :, 12] / mid_slope  # SLOPE
        processed_data[0, :, :, 5] = 1 - data[:, :, 13] / mid_elevation  # ELEVATION
        
        return processed_data

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
            
        files = request.files.getlist('files')
        location = request.form.get('location')
        
        # Process each file
        all_predictions = []
        for file in files:
            if file.filename == '':
                continue
                
            # Save file temporarily
            temp_dir = tempfile.mkdtemp()
            file_path = os.path.join(temp_dir, secure_filename(file.filename))
            file.save(file_path)
            
            # Process the file
            try:
                processed_data = process_h5_file(file_path)
                
                # Make prediction
                prediction = model.predict(processed_data)
                threshold = 0.5
                binary_prediction = (prediction > threshold).astype(np.uint8)
                
                # Calculate risk level based on prediction
                risk_percentage = np.mean(binary_prediction) * 100
                risk_level = 'low'
                if risk_percentage > 75:
                    risk_level = 'critical'
                elif risk_percentage > 50:
                    risk_level = 'high'
                elif risk_percentage > 25:
                    risk_level = 'medium'
                
                all_predictions.append({
                    'filename': file.filename,
                    'risk_level': risk_level,
                    'probability': float(risk_percentage),
                    'prediction_mask': binary_prediction[0, :, :, 0].tolist()
                })
                
            finally:
                # Clean up
                os.remove(file_path)
                os.rmdir(temp_dir)
        
        return jsonify({
            'predictions': all_predictions,
            'location': location
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)