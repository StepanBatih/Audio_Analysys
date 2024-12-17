import numpy as np
from sklearn.model_selection import train_test_split
import scipy.io.wavfile as wav
import pathlib
import pickle
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau
import time  # Для вимірювання часу

# Замір часу початку виконання
start_time = time.time()

# Параметри
wlen = 0.025  # Довжина вікна (сек)
wstep = 0.05  # Крок між вікнами (сек)

OUTPUT_DIR_PATH_44KHZ = "/Users/stepan_batih/ML/44KHZ/"

# Ініціалізація списків для кожного класу
class_data = {name: [] for name in ["Cats", "Kettles", "Dogs", "People", "Music", "Silence"]}

# Збір даних
output_dir_names = list(class_data.keys())
output_dirs = [OUTPUT_DIR_PATH_44KHZ]

for dir in output_dirs:
    for class_name in output_dir_names:
        for path in pathlib.Path(dir + "/" + class_name).glob('**/*.wav'):
            print(path)
            rate, sig = wav.read(path)

            # Розрахунок кількості семплів на вікно
            window_length = int(wlen * rate)
            window_step = int(wstep * rate)

            # Розбиття сигналу на вікна
            raw_features = []
            for start in range(0, len(sig) - window_length + 1, window_step):
                window = sig[start:start + window_length]
                raw_features.append(window)

            if raw_features:
                class_data[class_name].append(np.array(raw_features))

# Обробка даних
for key in class_data.keys():
    class_data[key] = [x.reshape(-1, window_length) for x in class_data[key]]
    class_data[key] = np.concatenate(class_data[key]) if class_data[key] else np.empty((0, window_length))

X = np.concatenate(list(class_data.values()))
y = np.concatenate([[class_name.lower()] * len(class_data[class_name]) for class_name in class_data])

# Нормалізація даних
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Перемішування та поділ даних
X, y = shuffle(X, y, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Перетворення міток у числовий формат
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

# Створення нейронної мережі
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),  # Вхідний шар тепер приймає довжину вікна (сигнал)
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(np.unique(y_train)), activation='softmax')
])

# Компіляція моделі
model.compile(
    optimizer=RMSprop(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callback для динамічного зменшення learning rate
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6
)

# Навчання моделі
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=128,
    validation_split=0.2,
    callbacks=[reduce_lr]
)

# Оцінка моделі
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Метрики
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_classes))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, target_names=encoder.classes_))
print("\nAccuracy:", accuracy_score(y_test, y_pred_classes))

# Збереження моделі
model.save("clear_data.h5")
with open("scaler_clear.pkl", 'wb') as file:
    pickle.dump(scaler, file)
with open("label_encoder_clear.pkl", 'wb') as file:
    pickle.dump(encoder, file)

# Замір часу завершення виконання
end_time = time.time()
execution_time = (end_time - start_time) / 60
print(f"Час виконання програми: {execution_time:.2f} хвилин")
