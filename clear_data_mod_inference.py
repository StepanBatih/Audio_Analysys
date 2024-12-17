import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pickle
from tensorflow.keras.models import load_model
import scipy.io.wavfile as wav
from scipy.signal import resample

# Завантаження моделі, scaler і label encoder
model_path = "clear_data.h5"
scaler_path = "scaler_clear.pkl"
label_encoder_path = "label_encoder_clear.pkl"

model = load_model(model_path)

with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

with open(label_encoder_path, 'rb') as file:
    encoder = pickle.load(file)

# Конфігурація
wlen_seconds = 0.025  # Довжина вікна (сек)
wstep_seconds = 0.05  # Крок між вікнами (сек)
expected_length = 200  # Очікувана кількість ознак (вхід для моделі)

# Функція для обробки аудіофайлу
def preprocess_audio(file_path):
    rate, sig = wav.read(file_path)

    # Довжина файлу в секундах
    duration = len(sig) / rate
    print(f"Довжина аудіофайлу: {duration:.2f} секунд, частота дискретизації: {rate} Гц")

    # Розрахунок кількості семплів на вікно
    window_length_samples = int(wlen_seconds * rate)
    window_step_samples = int(wstep_seconds * rate)

    # Розбиття сигналу на вікна
    raw_features = []
    for start in range(0, len(sig) - window_length_samples + 1, window_step_samples):
        window = sig[start:start + window_length_samples]

        # Масштабування кожного вікна до expected_length (800 ознак)
        if len(window) != expected_length:
            window = resample(window, expected_length)
        raw_features.append(window)

    raw_features = np.array(raw_features)

    # Перетворення в 2D масив
    raw_features = raw_features.reshape(raw_features.shape[0], -1)

    # Повторна перевірка розміру вікон
    print(f"Форма після ресемплінгу: {raw_features.shape}")
    if raw_features.shape[1] != expected_length:
        print(f"Повторне масштабування для кожного вікна...")
        raw_features = np.array([resample(window, expected_length) for window in raw_features])

    # Гарантія форми після масштабування
    if raw_features.shape[1] != expected_length:
        raise ValueError(
            f"Після ресемплінгу кількість ознак у вікні: {raw_features.shape[1]}, очікується: {expected_length}."
        )

    # Нормалізація
    raw_features = scaler.transform(raw_features)
    return raw_features

# Шлях до тестового аудіофайлу
test_file_path = "/Users/stepan_batih/ML/TEST/test_dog/5.wav"

# Обробка файлу
try:
    X_test = preprocess_audio(test_file_path)

    # Перевірка форми
    print(f"Форма X_test: {X_test.shape}")  # Очікується (n_windows, expected_length)

    # Передбачення
    predictions = model.predict(X_test)

    # Отримання класу з найбільшою ймовірністю для кожного вікна
    predicted_classes = np.argmax(predictions, axis=1)
    print(predicted_classes)

    # Декодування міток у назви класів
    decoded_predictions = encoder.inverse_transform(predicted_classes)

    # Виведення результатів
    print(f"Файл: {test_file_path}")
    print("Передбачені класи для кожного вікна:")
    print(decoded_predictions)
    unique_classes, counts = np.unique(decoded_predictions, return_counts=True)
    total_windows = len(decoded_predictions)
    print("\nСтатистика передбачень:")
    for cls, count in zip(unique_classes, counts):
        percentage = (count / total_windows) * 100
        print(f"Клас: {cls}, Кількість: {count}, Частка: {percentage:.2f}%")
    final_prediction = unique_classes[np.argmax(counts)]
    print(f"\nПідсумковий передбачений клас: {final_prediction}")
    
    
    
    '''
    # Підсумковий клас (найчастіший)
    final_prediction = max(set(decoded_predictions), key=decoded_predictions.tolist().count)
    print(f"Підсумковий передбачений клас: {final_prediction}")'''

except Exception as e:
    print(f"Помилка обробки: {e}")
