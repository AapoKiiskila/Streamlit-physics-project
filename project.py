import streamlit as st
import pandas as pd
import folium
import numpy as np
from streamlit_folium import st_folium
from scipy.signal import butter, filtfilt
from math import radians, cos, sin, asin, sqrt

url_acceleration = 'https://raw.githubusercontent.com/AapoKiiskila/Streamlit-physics-project/refs/heads/main/Linear%20Accelerometer.csv'
df_acceleration = pd.read_csv(url_acceleration)

url_location = 'https://raw.githubusercontent.com/AapoKiiskila/Streamlit-physics-project/refs/heads/main/Location.csv'
df_location = pd.read_csv(url_location)



#Low-pass filter -->

def butter_lowpass_filter(acceleration, cutoff, fs, nyq, order):
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype = 'low', analog = False)
    y = filtfilt(b, a, acceleration)
    return y

acceleration = df_acceleration['Z (m/s^2)']
T = df_acceleration['Time (s)'].max()
n = len(df_acceleration['Time (s)'])
fs = n / T
nyq = fs / 2 
order = 3
cutoff = 1 / 0.5

filtered_signal = butter_lowpass_filter(acceleration, cutoff, fs, nyq, order)

chart_filtered_signal = pd.DataFrame(
    {
        'Time (s)': df_acceleration['Time (s)'],
        'Filtered acceleration': filtered_signal
    }
).set_index('Time (s)')

signal_periods = 0
for i in range(n - 1):
    if filtered_signal[i] / filtered_signal[i + 1] < 0:
        signal_periods = signal_periods + 1

filtered_signal_steps = signal_periods / 2



#Fourier analysis -->

f = df_acceleration['Z (m/s^2)']
t = df_acceleration['Time (s)']
N = len(f)
dt = np.max(t) / N

fourier = np.fft.fft(f, N)
psd = fourier * np.conj(fourier) / N
freq = np.fft.fftfreq(N, dt)
L = np.arange(1, int(N / 2))

chart_psd = pd.DataFrame(np.transpose(np.array([freq[L], psd[L].real])), columns=['freq', 'psd'])

f_max = freq[L][psd[L] == np.max(psd[L])][0]
fourier_steps = np.max(t) * f_max

#T = 1 / f_max
#fourier_steps = np.max(t) / T



#Total distance, average velocity, stride length & step length -->

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a)) 
    r = 6371
    return c * r

df_location['dist'] = np.zeros(len(df_location))
df_location['t_diff'] = np.zeros(len(df_location))

for i in range(len(df_location) - 1):
    df_location.loc[i + 1, 'dist'] = haversine(df_location['Longitude (°)'][i], df_location['Latitude (°)'][i], df_location['Longitude (°)'][i + 1], df_location['Latitude (°)'][i + 1]) * 1000
    df_location.loc[i + 1, 't_diff'] = df_location['Time (s)'][i + 1] - df_location['Time (s)'][i]

df_location['velocity'] = df_location['dist'] / df_location['t_diff']
df_location['cumulative_distance'] = np.cumsum(df_location['dist'])

distance_travelled = df_location['cumulative_distance'].max()
average_velocity = df_location['velocity'].mean()

filtered_signal_stride_length_centimeters = distance_travelled / filtered_signal_steps * 100
filtered_signal_step_length_centimeters = filtered_signal_stride_length_centimeters / 2

fourier_stride_length_centimeters = distance_travelled / fourier_steps * 100
fourier_step_length_centimeters = fourier_stride_length_centimeters / 2



#Map -->

start_lat = df_location['Latitude (°)'].mean()
start_long = df_location['Longitude (°)'].mean()
map = folium.Map(location = [start_lat, start_long], zoom_start = 15)

folium.PolyLine(df_location[['Latitude (°)', 'Longitude (°)']], color = 'red', weight = 2, opacity = 1).add_to(map)



#Display results -->

st.title('Walking from a movie theater to a bus stop')
st.write('Step count by using low-pass filtering: ', filtered_signal_steps)
st.write('Step count by using Fourier analysis: ', fourier_steps)
st.write('Average velocity: ', round(average_velocity, 2), 'm/s')
st.write('Total distance: ', round(distance_travelled, 2), 'm')
st.header("Stride length & step length")
st.subheader("Calculated by using the step count from low pass filtering:", divider = 'gray')
st.write('Stride length: ', round(filtered_signal_stride_length_centimeters, 1), 'cm')
st.write('Step length: ', round(filtered_signal_step_length_centimeters, 1), 'cm')
st.subheader("Calculated by using the step count from Fourier analysis:", divider = 'gray')
st.write('Stride length: ', round(fourier_stride_length_centimeters, 1), 'cm')
st.write('Step length: ', round(fourier_step_length_centimeters, 1), 'cm')
st.header("Filtered acceleration (z-component)")
st.line_chart(chart_filtered_signal, x_label = 'Time (s)', y_label = 'Filtered acceleration (m/s²)')
st.header("Power spectral density")
st.line_chart(chart_psd, x = 'freq', y = 'psd' , y_label = 'Power', x_label = 'Frequency [Hz]')
st.header("Map")
st_map = st_folium(map, width = 900, height = 650)
