import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import plotly.express as px
from flask import Flask, request, jsonify

# More realistic drone data
def generate_data(num_samples=1000):
    np.random.seed(42)
    
    motor_speed = np.random.normal(loc=5000, scale=100, size=num_samples)  # Tighter variance
    battery_level = np.clip(np.random.normal(loc=90, scale=5, size=num_samples), 0, 100)  # Battery should be between 0-100%
    altitude = np.random.normal(loc=100, scale=15, size=num_samples)  # Normal altitude with moderate variance
    gps_coords = np.cumsum(np.random.normal(loc=0, scale=0.1, size=(num_samples, 2)), axis=0)  # Simulated GPS drift
    
    motor_anomaly_indices = np.arange(0, num_samples, 75)
    battery_anomaly_indices = np.arange(0, num_samples, 120)
    altitude_anomaly_indices = np.arange(0, num_samples, 150)
    gps_anomaly_indices = np.arange(0, num_samples, 200)

    motor_speed[motor_anomaly_indices] = np.random.uniform(7000, 12000, size=len(motor_anomaly_indices))  # Motor speed spikes
    battery_level[battery_anomaly_indices] = np.random.uniform(0, 10, size=len(battery_anomaly_indices))  # Battery sudden drops
    altitude[altitude_anomaly_indices] = np.random.uniform(10, 30, size=len(altitude_anomaly_indices))  # Sudden altitude changes
    
    gps_coords[gps_anomaly_indices] += np.random.normal(loc=5, scale=2, size=(len(gps_anomaly_indices), 2))  # GPS jumps
    
    # Create a DataFrame
    data = pd.DataFrame({
        'Motor_Speed': motor_speed,
        'Battery_Level': battery_level,
        'Altitude': altitude,
        'GPS_X': gps_coords[:, 0],
        'GPS_Y': gps_coords[:, 1]
    })
    return data

# Train Isolation Forest model with hyperparameter tuning for better anomaly detection
def train_model(data):
    features = data[['Motor_Speed', 'Battery_Level', 'Altitude', 'GPS_X', 'GPS_Y']]  # Include GPS
    model = IsolationForest(n_estimators=100, max_samples=256, contamination=0.05, random_state=42)
    model.fit(features)
    return model

# Identify anomalies
def identify_anomalies(model, data):
    features = data[['Motor_Speed', 'Battery_Level', 'Altitude', 'GPS_X', 'GPS_Y']]
    data['Anomaly'] = model.predict(features)  # -1 for anomaly, 1 for normal
    return data

# Visualize anomalies across multiple parameters (Matplotlib version)
def plot_anomalies(data):
    plt.figure(figsize=(12, 8))
    
    #Motor Speed anomalies
    plt.subplot(4, 1, 1)
    plt.scatter(data.index, data['Motor_Speed'], color='blue', label='Normal', s=10)
    plt.scatter(data[data['Anomaly'] == -1].index, data[data['Anomaly'] == -1]['Motor_Speed'], color='red', label='Anomaly', s=20)
    plt.title('Drone Motor Speed Anomalies')
    plt.xlabel('Sample Index')
    plt.ylabel('Motor Speed')
    
    #Battery Level anomalies
    plt.subplot(4, 1, 2)
    plt.scatter(data.index, data['Battery_Level'], color='green', label='Normal', s=10)
    plt.scatter(data[data['Anomaly'] == -1].index, data[data['Anomaly'] == -1]['Battery_Level'], color='red', label='Anomaly', s=20)
    plt.title('Drone Battery Level Anomalies')
    plt.xlabel('Sample Index')
    plt.ylabel('Battery Level')
    
    #Altitude anomalies
    plt.subplot(4, 1, 3)
    plt.scatter(data.index, data['Altitude'], color='purple', label='Normal', s=10)
    plt.scatter(data[data['Anomaly'] == -1].index, data[data['Anomaly'] == -1]['Altitude'], color='red', label='Anomaly', s=20)
    plt.title('Drone Altitude Anomalies')
    plt.xlabel('Sample Index')
    plt.ylabel('Altitude')
    
    #GPS anomalies (X-axis)
    plt.subplot(4, 1, 4)
    plt.scatter(data.index, data['GPS_X'], color='orange', label='Normal', s=10)
    plt.scatter(data[data['Anomaly'] == -1].index, data[data['Anomaly'] == -1]['GPS_X'], color='red', label='Anomaly', s=20)
    plt.title('Drone GPS X-coordinate Anomalies')
    plt.xlabel('Sample Index')
    plt.ylabel('GPS X-coordinate')
    
    plt.tight_layout()
    plt.show()

#Interactive Plot using Plotly
def interactive_plot(data):
    fig = px.scatter_3d(data, x='Motor_Speed', y='Battery_Level', z='Altitude', 
                        color=data['Anomaly'].map({1: 'Normal', -1: 'Anomaly'}), 
                        title='3D Anomaly Detection Visualization')
    fig.show()


if __name__ == "__main__":
    simulated_data = generate_data()
    isolation_forest_model = train_model(simulated_data)
    results = identify_anomalies(isolation_forest_model, simulated_data)
    
    # Plot using Matplotlib
    plot_anomalies(results)
    
    # Interactive Plot using Plotly
    interactive_plot(results)

print("Anomaly detection complete!")

#API Deployment using Flask
app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detect_anomalies():
    # Receive CSV file
    file = request.files['file']
    data = pd.read_csv(file)
    model = train_model(data)  # Train the model on the incoming data
    results = identify_anomalies(model, data)
    anomalies = results[results['Anomaly'] == -1]
    return anomalies.to_json()  # Return anomalies as JSON

if __name__ == "__main__":
    app.run(debug=True)
