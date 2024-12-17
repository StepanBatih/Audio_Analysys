import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pickle
from tensorflow.keras.models import load_model
import scipy.io.wavfile as wav

# Завантаження моделі, scaler і label encoder
model_path = "fft.h5"
scaler_path = "scaler_fft.pkl"
label_encoder_path = "label_encoder_fft.pkl"

model = load_model(model_path)

with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

with open(label_encoder_path, 'rb') as file:
    encoder = pickle.load(file)

# Конфігурація
FFT_COMPONENTS = 100
wlen = 0.025  # Довжина вікна (у секундах)
wstep = 0.05  # Крок вікна (у секундах)

# Функція для обробки аудіофайлу
def preprocess_audio(file_path):
    rate, sig = wav.read(file_path)

    # Довжина файлу в секундах
    duration = len(sig) / rate
    print(f"Довжина аудіофайлу: {duration:.2f} секунд")

    # Розрахунок кількості семплів на вікно
    window_length = int(wlen * rate)
    window_step = int(wstep * rate)

    # Розбиття сигналу на вікна і обчислення FFT
    fft_features = []
    for start in range(0, len(sig) - window_length + 1, window_step):
        window = sig[start:start + window_length]
        fft_magnitude = np.abs(np.fft.rfft(window))[:FFT_COMPONENTS]
        fft_features.append(fft_magnitude)

    fft_features = np.array(fft_features)

    # Кількість вікон
    n_windows = len(fft_features)
    print(f"Кількість вікон: {n_windows}")

    # Видалення зайвого виміру, якщо присутній
    if fft_features.ndim == 3 and fft_features.shape[-1] == 2:
        fft_features = fft_features[..., 0]

    # Нормалізація
    fft_features = scaler.transform(fft_features)
    return fft_features

# Шлях до тестового аудіофайлу
test_file_path = "/Users/stepan_batih/ML/TEST/test_dog/5.wav"

# Обробка файлу
X_test = preprocess_audio(test_file_path)

# Перевірка форми
print(f"Форма X_test: {X_test.shape}")  # Очікується (n_windows, 100)

# Передбачення
predictions = model.predict(X_test)

# Отримання класу з найбільшою ймовірністю для кожного вікна
predicted_classes = np.argmax(predictions, axis=1)

# Декодування міток у назви класів
decoded_predictions = encoder.inverse_transform(predicted_classes)
print(decoded_predictions)

# Виведення результатів
print(f"Файл: {test_file_path}")
print("Передбачені класи для кожного вікна:")
print(decoded_predictions)
unique_classes, counts = np.unique(decoded_predictions, return_counts=True)
total_windows = len(decoded_predictions)
print("\nСтатистика передбачень:")
for cls, count in zip(unique_classes, counts):
    percentage = (count / total_windows) * 100
    #print(f"Клас: {cls}, Кількість: {count}, Частка: {percentage:.2f}%")
    print( f"{percentage:.1f}%")
    
final_prediction = unique_classes[np.argmax(counts)]
print(f"\nПідсумковий передбачений клас: {final_prediction}")

# Підсумковий клас (найчастіший)
final_prediction = max(set(decoded_predictions), key=decoded_predictions.tolist().count)
print(f"Підсумковий передбачений клас: {final_prediction}")
