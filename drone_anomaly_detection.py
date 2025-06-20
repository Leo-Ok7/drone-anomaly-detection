import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import IsolationForest
from flask import Flask, request, jsonify

# just making some mock drone data
def drone_data(n=1000):
    np.random.seed(42)
    m = np.random.normal(5000, 120, n)
    b = np.clip(np.random.normal(88, 7, n), 0, 100)
    h = np.random.normal(95, 10, n)
    g = np.cumsum(np.random.normal(0, 0.06, size=(n, 2)), axis=0)

    # throw in anomalies here and there
    for i in range(50, n, 85):
        m[i] = np.random.uniform(7000, 11500)
    for i in range(30, n, 115):
        b[i] = np.random.uniform(2, 9)
    for i in range(20, n, 150):
        h[i] = np.random.uniform(20, 55)
    for i in range(70, n, 200):
        g[i] += np.random.normal(4.5, 1.2, 2)

    return pd.DataFrame({
        'motor': m,
        'battery': b,
        'alt': h,
        'gps_x': g[:, 0],
        'gps_y': g[:, 1]
    })

# train anomaly model (kinda works)
def isolate(df):
    forest = IsolationForest(n_estimators=97, contamination=0.045, random_state=101)
    forest.fit(df[['motor', 'battery', 'alt', 'gps_x', 'gps_y']])
    return forest

# add anomaly flag
def add_anom(df, model):
    df['anom'] = model.predict(df[['motor', 'battery', 'alt', 'gps_x', 'gps_y']])
    return df

# visual stuff
def quickplot(df):
    fig, axs = plt.subplots(4, 1, figsize=(11, 9))
    anom = df['anom'] == -1

    axs[0].plot(df.index, df.motor, lw=1)
    axs[0].scatter(df.index[anom], df.motor[anom], color='red', s=15)
    axs[0].set_title("motor rpm")

    axs[1].plot(df.battery, color='green', lw=1)
    axs[1].scatter(df.index[anom], df.battery[anom], color='red', s=15)
    axs[1].set_title("battery %")

    axs[2].plot(df.alt, color='purple', lw=1)
    axs[2].scatter(df.index[anom], df.alt[anom], color='red', s=15)
    axs[2].set_title("altitude")

    axs[3].plot(df.gps_x, color='orange', lw=1)
    axs[3].scatter(df.index[anom], df.gps_x[anom], color='red', s=15)
    axs[3].set_title("gps x drift")

    plt.tight_layout()
    plt.show()

# 3D plot
def three_d(df):
    df['lab'] = df['anom'].map({-1: 'anomaly', 1: 'ok'})
    fig = px.scatter_3d(df, x='motor', y='battery', z='alt', color='lab')
    fig.show()


# main
if __name__ == '__main__':
    d = drone_data()
    model = isolate(d)
    d = add_anom(d, model)
    quickplot(d)
    three_d(d)
    print("done.")

# flask bit for later
app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detect():
    f = request.files.get('file')
    if f is None:
        return jsonify({'error': 'no file'}), 400
    try:
        df = pd.read_csv(f)
        m = isolate(df)
        df = add_anom(df, m)
        return df[df['anom'] == -1].to_json()
    except Exception as e:
        return jsonify({'error': str(e)}), 500
