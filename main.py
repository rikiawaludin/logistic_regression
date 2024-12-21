from flask import Flask, request, jsonify
from joblib import load
import numpy as np
import os  # Tambahkan impor os di sini

# Inisialisasi Flask
app = Flask(__name__)

# Load model yang sudah disimpan
model = load('logistic_reg_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data dari request
        data = request.get_json()
        # Pastikan semua input dikonversi ke tipe numerik
        features = np.array([
            float(data['Sex']),
            float(data['Age']),
            float(data['No_of_Parents_or_Children_on_Board']),
            float(data['No_of_Siblings_or_Spouses_on_Board']),
            float(data['Passenger_Fare'])
        ]).reshape(1, -1)
        
        prediction = model.predict(features)
        return jsonify({'result': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

# Menjalankan Flask
if __name__ == '__main__':
    # Gunakan konfigurasi port dari environment
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
