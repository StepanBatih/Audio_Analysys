import numpy as np
from sklearn.model_selection import train_test_split
import scipy.io.wavfile as wav
import pathlib
import pickle
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

wlen = 0.5
wstep = 0.5  # Зменшили крок вікна, щоб більше накладення вікон

OUTPUT_DIR_PATH_44KHZ = "/Users/stepan_batih/ML/44KHZ/"

iterator = 1

cats_44kHz = []
kettles_44kHz = []
dogs_44kHz = []
people_44kHz = []
music_44kHz = []
silence_44kHz = []

output_dir_names = ["/Cats", "/Kettles", "/Dogs", "/People", "/Music", "/Silence"]
output_dirs = [OUTPUT_DIR_PATH_44KHZ]

for dir in output_dirs:
    for dir_name in output_dir_names:
        for path in pathlib.Path(dir + dir_name).glob('**/*.wav'):
            print(path)
            rate, sig = wav.read(path)

            # Calculate the number of samples per window
            window_length = int(wlen * rate)
            window_step = int(wstep * rate)

            # Split the signal into windows and calculate FFT for each window
            fft_features = []
            for start in range(0, len(sig) - window_length + 1, window_step):
                window = sig[start:start + window_length]
                fft_magnitude = np.abs(np.fft.rfft(window))[:50]  # Збільшили до 50 компонент FFT
                fft_features.append(fft_magnitude)

            fft_features = np.array(fft_features)
            if fft_features.ndim == 2:
                fft_features = fft_features[np.newaxis, ...]

            if iterator == 1:
                cats_44kHz.append(fft_features)
            elif iterator == 2:
                kettles_44kHz.append(fft_features)
            elif iterator == 3:
                dogs_44kHz.append(fft_features)
            elif iterator == 4:
                people_44kHz.append(fft_features)
            elif iterator == 5:
                music_44kHz.append(fft_features)
            elif iterator == 6:
                silence_44kHz.append(fft_features)

        iterator += 1

# Ensure all arrays have the same dimensions before concatenation
cats_44kHz = [x.reshape(-1, 50) for x in cats_44kHz]
kettles_44kHz = [x.reshape(-1, 50) for x in kettles_44kHz]
dogs_44kHz = [x.reshape(-1, 50) for x in dogs_44kHz]
people_44kHz = [x.reshape(-1, 50) for x in people_44kHz]
music_44kHz = [x.reshape(-1, 50) for x in music_44kHz]
silence_44kHz = [x.reshape(-1, 50) for x in silence_44kHz]

cat_x = np.concatenate(cats_44kHz)
kettles_x = np.concatenate(kettles_44kHz)
dog_x = np.concatenate(dogs_44kHz)
people_x = np.concatenate(people_44kHz)
music_x = np.concatenate(music_44kHz)
silence_x = np.concatenate(silence_44kHz)

X = np.concatenate([cat_x, kettles_x, dog_x, people_x, music_x, silence_x])

cat_y = ["cat" for _ in range(cat_x.shape[0])]
kettles_y = ["kettles" for _ in range(kettles_x.shape[0])]
dog_y = ["dog" for _ in range(dog_x.shape[0])]
people_y = ["people" for _ in range(people_x.shape[0])]
music_y = ["music" for _ in range(music_x.shape[0])]
silence_y = ["silence" for _ in range(silence_x.shape[0])]

y = np.concatenate([cat_y, kettles_y, dog_y, people_y, music_y, silence_y])

##############################################################################

"""MLP ALGORITHM"""

# Normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

X, y = shuffle(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Improved MLP model with more layers, regularization, and increased iterations
mlp_model = MLPClassifier(hidden_layer_sizes=(100, 50), alpha=0.001, max_iter=1000, activation='relu')

mlp_model_fit = mlp_model.fit(X_train, y_train)

y_pred = mlp_model.predict(X_test)

# Metrics for MLP
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:")
print(result1)
result2 = accuracy_score(y_test, y_pred)
print("MLP Accuracy:", result2)

pkl_mlp = "pickle_model_mlp_fft.pkl"
with open(pkl_mlp, 'wb') as file:
    pickle.dump(mlp_model, file)
