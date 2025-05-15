from flask import Flask, request, jsonify
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load and prepare data
features = ['acousticness', 'danceability', 'energy', 'instrumentalness',
            'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'popularity', 'duration_ms']

# Load model and scaler using pickle
with open('model.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('pca.pkl', 'rb') as f:
    pca = pickle.load(f)

#with open('kmeans_model.pkl', 'rb') as f:
    #model = pickle.load(f)

@app.route('/')
def index():
    return "KMeans Clustering API for Rolling Stones Tracks"

@app.route('/predict-cluster', methods=['POST'])
def predict_cluster():
    try:
        data = request.json
        input_data = np.array([[data[feature] for feature in features]])
        input_scaled = scaler.transform(input_data)
        input_pca = pca.transform(input_scaled)
        cluster = model.predict(input_pca)[0]
        return jsonify({'cluster': int(cluster)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
