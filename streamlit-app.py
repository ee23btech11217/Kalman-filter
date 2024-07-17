import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from kalman_filter import KalmanFilter

# Initialize session state to store our data
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=['x', 'y', 'z'])
    st.session_state.filtered_data = pd.DataFrame(columns=['x', 'y', 'z'])

# Initialize Kalman filter
kf = KalmanFilter(6, 3)  # 6 state variables (x, y, z, vx, vy, vz), 3 measurement variables (x, y, z)

# Set up the Kalman filter
dt = 0.1  # time step
F = np.array([
    [1, 0, 0, dt, 0, 0],
    [0, 1, 0, 0, dt, 0],
    [0, 0, 1, 0, 0, dt],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1]
])
kf.set_state_transition(F)

H = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0]
])
kf.set_measurement_matrix(H)

# Set process noise covariance
Q = np.eye(6) * 0.1
kf.set_process_noise(Q)

# Set measurement noise covariance
R = np.eye(3) * 1
kf.set_measurement_noise(R)

st.title('3D Gyroscopic Data Plot with Kalman Filter')

# Function to update data
def update_data(x, y, z):
    # Update raw data
    new_data = pd.DataFrame({'x': [x], 'y': [y], 'z': [z]})
    st.session_state.data = pd.concat([st.session_state.data, new_data], ignore_index=True)

    # Apply Kalman filter
    kf.predict()
    kf.update(np.array([x, y, z]))
    filtered_state = kf.get_state()
    
    # Update filtered data
    new_filtered_data = pd.DataFrame({'x': [filtered_state[0]], 'y': [filtered_state[1]], 'z': [filtered_state[2]]})
    st.session_state.filtered_data = pd.concat([st.session_state.filtered_data, new_filtered_data], ignore_index=True)

# Input fields for x, y, z coordinates
col1, col2, col3 = st.columns(3)
with col1:
    x = st.number_input('X coordinate', value=0.0)
with col2:
    y = st.number_input('Y coordinate', value=0.0)
with col3:
    z = st.number_input('Z coordinate', value=0.0)

if st.button('Add Point'):
    update_data(x, y, z)

# Plot the data
if not st.session_state.data.empty:
    fig = go.Figure()

    # Plot raw data
    fig.add_trace(go.Scatter3d(
        x=st.session_state.data['x'],
        y=st.session_state.data['y'],
        z=st.session_state.data['z'],
        mode='lines+markers',
        name='Raw Data'
    ))

    # Plot filtered data
    fig.add_trace(go.Scatter3d(
        x=st.session_state.filtered_data['x'],
        y=st.session_state.filtered_data['y'],
        z=st.session_state.filtered_data['z'],
        mode='lines+markers',
        name='Filtered Data'
    ))

    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ))

    st.plotly_chart(fig)

# Display the data tables
st.subheader('Raw Data')
st.write(st.session_state.data)

st.subheader('Filtered Data')
st.write(st.session_state.filtered_data)