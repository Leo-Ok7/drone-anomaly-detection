# Drone-Anomaly-Detection
A project that detects anomalies in simulated drone data using the Isolation Forest algorithm.

**Features**
* Generates fake drone data with some built-in anomalies
* Uses Isolation Forest to find unusual values
* Plots results using Matplotlib and Plotly (2D & 3D)
* Includes a basic Flask API for detecting anomalies from CSV uploads

**Data Columns
* motor: Motor speed (RPM)
* battery: Battery percentage
* alt: Altitude (meters)
* gps_x, gps_y: Simulated GPS positions
* anom: Anomaly label (-1 = anomaly, 1 = normal)

**How to Use
Run Python your_script_name.py

This runs the data simulation, trains the model, shows plots, and prints out anomalies.
