import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.title("Візуалізація Калман-фільтра")

st.sidebar.header("Параметри сигналу")
frequency = st.sidebar.slider("Частота (Гц)", 0.1, 5.0, 1.0)
amplitude = st.sidebar.slider("Амплітуда", 0.1, 10.0, 5.0)
offset = st.sidebar.slider("Зсув", 0.0, 20.0, 10.0)
sampling_interval = st.sidebar.slider("Інтервал вибірки (с)", 0.001, 0.1, 0.001)
total_time = st.sidebar.slider("Загальний час (с)", 0.1, 5.0, 1.0)

st.sidebar.header("Параметри шуму")
noise_variance = st.sidebar.slider("Дисперсія шуму", 1.0, 50.0, 16.0)
noise_std_dev = np.sqrt(noise_variance)

st.sidebar.header("Параметри Калман-фільтра")
F_value = st.sidebar.number_input("Матриця переходів станів (F)", value=1.0, step=0.1)
H_value = st.sidebar.number_input("Матриця вимірювань (H)", value=1.0, step=0.1)
Q_value = st.sidebar.number_input("Коваріація шуму процесу (Q)", value=1.0, step=0.1)
R_value = st.sidebar.number_input("Коваріація шуму вимірювань (R)", value=10.0, step=0.1)
P_value = st.sidebar.number_input("Коваріація помилки оцінки (P)", value=1.0, step=0.1)
initial_state = st.sidebar.number_input("Початковий стан (x)", value=0.0, step=0.1)

class KalmanFilter:
    def __init__(self, F, H, Q, R, P, x):
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.P = P
        self.x = x

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        return self.x

    def update(self, z):
        K = np.dot(self.P, self.H.T) / (np.dot(self.H, np.dot(self.P, self.H.T)) + self.R)
        self.x = self.x + K * (z - np.dot(self.H, self.x))
        self.P = (np.eye(len(self.P)) - K * self.H) @ self.P
        return self.x

time_steps = np.arange(0, total_time, sampling_interval)
true_signal = offset + amplitude * np.sin(2 * np.pi * frequency * time_steps)
noisy_signal = [val + np.random.normal(0, noise_std_dev) for val in true_signal]

F = np.array([[F_value]])
H = np.array([[H_value]])
Q = np.array([[Q_value]])
R = np.array([[R_value]])
P = np.array([[P_value]])
x = np.array([[initial_state]])

kf = KalmanFilter(F, H, Q, R, P, x)
kalman_estimates = []

for measurement in noisy_signal:
    kf.predict()
    estimate = kf.update(measurement)
    kalman_estimates.append(estimate[0][0])

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(time_steps, noisy_signal, label='Загачений сигнал', color='orange', linestyle='-', alpha=0.6)
ax.plot(time_steps, true_signal, label='Справжній сигнал (синусоїда)', linestyle='--', color='blue')
ax.plot(time_steps, kalman_estimates, label='Оцінка Калман-фільтра', color='green')
ax.set_xlabel('Час (с)')
ax.set_ylabel('Значення')
ax.set_title('Застосування Калман-фільтра до загаченого синусоїдального сигналу')
ax.legend()
ax.grid()

st.pyplot(fig)


