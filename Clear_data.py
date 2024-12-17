from python_speech_features import mfcc
from sklearn.model_selection import train_test_split
import wave
import pathlib
import pickle 
import numpy as np
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

wlen = 0.5  # Зменшено до 0.05 секунди

OUTPUT_DIR_PATH_44KHZ = "D:/ML/44KHZ"

output_dir_names = ["/Cats","/Kettles","/Dogs","/People","/Music","/Silence"]
output_dirs = [OUTPUT_DIR_PATH_44KHZ]

# Генератор для зчитування аудіо сегментами та створення міток
def audio_segment_generator():
    iterator = 1
    labels = ["cat", "kettles", "dog", "people", "music", "silence"]
    for dir in output_dirs:
        for dir_name in output_dir_names:
            for path in pathlib.Path(dir+dir_name).glob('**/*.wav'):
                print(path)
                with wave.open(str(path), 'rb') as wav_file:
                    rate = wav_file.getframerate()
                    frames_per_segment = int(rate * wlen)  # Кількість кадрів для сегменту тривалістю 0.05 сек
                    while True:
                        frames = wav_file.readframes(frames_per_segment)
                        if not frames:
                            break
                        segment = np.frombuffer(frames, dtype=np.int16)
                        yield segment, labels[iterator - 1]
            iterator += 1

# Формування навчальної та тестової вибірок
X = []
y = []

segment_length = int(44100 * wlen)  # Очікувана кількість семплів у сегменті (44100 Гц * 0.05 сек)

for segment, label in audio_segment_generator():
    # Нормалізація розміру сегмента
    if len(segment) < segment_length:
        # Доповнення нулями, якщо сегмент коротший за очікуваний розмір
        segment = np.pad(segment, (0, segment_length - len(segment)), mode='constant')
    elif len(segment) > segment_length:
        # Обрізання, якщо сегмент довший за очікуваний розмір
        segment = segment[:segment_length]
    X.append(segment)
    y.append(label)

# Перетворення в numpy масиви
X = np.array(X)
y = np.array(y)

##############################################################################

# "MLP ALGORITHM "

X, y = shuffle(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

mlp_model = MLPClassifier(hidden_layer_sizes=10)
mlp_model_fit = mlp_model.fit(X_train, y_train)

y_pred = mlp_model.predict(X_test)

# Метрики для MLP
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:")
print(result1)
result2 = accuracy_score(y_test, y_pred)
print("MLP Accuracy:", result2)

# Збереження моделі
pkl_mlp = "pickle_model_mlp_clear.pkl"
with open(pkl_mlp, 'wb') as file:
    pickle.dump(mlp_model, file)
