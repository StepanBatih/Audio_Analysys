from python_speech_features import mfcc
import wave
import pickle
import numpy as np
import librosa

# Завантаження натренованої моделі
pkl_mlp = "pickle_model_mlp_clear.pkl"
with open(pkl_mlp, 'rb') as file:
    pickle_model_mlp = pickle.load(file)

# Функція для інференсу

def inference(input_file):
    print(f"Processing file: {input_file}")
    wlen = 0.05  # Довжина сегменту 0.05 секунд
    segment_length = int(44100 * wlen)  # Очікувана кількість семплів у сегменті (44100 Гц * 0.05 сек)
    arr_x = []

    # Зчитування аудіо файлу та перезразкування до 44100 Гц
    sig, rate = librosa.load(input_file, sr=44100)
    
    # Розбиття сигналу на сегменти
    for start in range(0, len(sig), segment_length):
        end = start + segment_length
        segment = sig[start:end]
        # Нормалізація розміру сегмента
        if len(segment) < segment_length:
            segment = np.pad(segment, (0, segment_length - len(segment)), mode='constant')
        elif len(segment) > segment_length:
            segment = segment[:segment_length]
        arr_x.append(segment)
    
    # Перетворення на 2D масив для прогнозування
    inf_x = np.array(arr_x)
    
    # Виведення кількості сегментів
    num_segments = len(inf_x)
    print(f"Кількість сегментів: {num_segments}")
    
    # Прогнозування на основі завантаженої моделі для кожного сегменту окремо
    y_pred = []
    for segment in inf_x:
        segment = segment.reshape(1, -1)  # Перетворення на 2D масив для моделі
        y_pred_segment = pickle_model_mlp.predict(segment)
        y_pred.append(y_pred_segment[0])
    
    # Аналіз прогнозів
    unique, counts = np.unique(y_pred, return_counts=True)
    sum_pred = sum(counts).astype(float)

    result = dict(zip(unique, counts))
    prediction = max(result, key=result.get)
    per = (max(counts) / sum_pred) * 100
    
    # Визначення довжини файлу
    duration = len(sig) / rate

    print(f"Довжина файлу: {duration:.2f} секунд")
    return y_pred, prediction, '{:.2f}'.format(per), '%'

# Перевірка на окремому файлі
wav_f = "D:/ML/TEST/test_people/4.wav"
print(inference(wav_f))
