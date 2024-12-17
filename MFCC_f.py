from python_speech_features import mfcc
from sklearn.model_selection import train_test_split
import scipy.io.wavfile as wav
import pathlib
import pickle 
import numpy as np
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

wlen = 0.5
wstep = 0.5

OUTPUT_DIR_PATH_44KHZ = "/Users/stepan_batih/ML/44KHZ/"


iterator = 1
rate_iterator = 1

cats_44kHz=[]
kettles_44kHz=[]
dogs_44kHz=[]
people_44kHz=[]
music_44kHz=[]
silence_44kHz=[]

output_dir_names = ["/Cats","/Kettles","/Dogs","/People","/Music","/Silence"]
output_dirs = [OUTPUT_DIR_PATH_44KHZ]



for dir in output_dirs:
    #print(dir)
    for dir_name in output_dir_names:
        #print(dir_name)
        for path in pathlib.Path(dir+dir_name).glob('**/*.wav'):
            print(path)
            rate,sig = wav.read(path)
            mfcc_option = mfcc(signal=sig,samplerate=rate,winlen=wlen,winstep=wstep,nfilt=26*2,nfft=22048,numcep=52)
            if (iterator == 1):
                mfcc_feat = mfcc_option
                cats_44kHz.append(mfcc_feat)
            elif (iterator == 2):
                mfcc_feat = mfcc_option
                kettles_44kHz.append(mfcc_feat)
            elif (iterator == 3):
                mfcc_feat = mfcc_option
                dogs_44kHz.append(mfcc_feat)
            elif (iterator == 4):
                mfcc_feat = mfcc_option
                people_44kHz.append(mfcc_feat)
            elif (iterator == 5):
                mfcc_feat = mfcc_option
                music_44kHz.append(mfcc_feat)
            elif (iterator == 6):
                mfcc_feat = mfcc_option
                silence_44kHz.append(mfcc_feat)
           
 
        iterator += 1

cat_x = np.concatenate(cats_44kHz)
kettles_x = np.concatenate(kettles_44kHz)
dog_x = np.concatenate(dogs_44kHz)
people_x = np.concatenate(people_44kHz)
music_x = np.concatenate(music_44kHz)
silence_x = np.concatenate(silence_44kHz)

X = np.concatenate([cat_x,kettles_x,dog_x,people_x,music_x,silence_x])

cat_y = ["cat" for _ in range(cat_x.shape[0])]
kettles_y = ["kettles" for _ in range(kettles_x.shape[0])]
dog_y = ["dog" for _ in range(dog_x.shape[0])]
people_y = ["people" for _ in range(people_x.shape[0])]
music_y = ["music" for _ in range(music_x.shape[0])]
silence_y = ["silence" for _ in range(silence_x.shape[0])]

y = np.concatenate([cat_y,kettles_y,dog_y,people_y,music_y,silence_y])

##############################################################################

"""MLP ALGORITM """

X, y = shuffle(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)


mlp_model = MLPClassifier(hidden_layer_sizes=10)

mlp_model_fit = mlp_model.fit(X_train, y_train,)


y_pred = mlp_model.predict(X_test)

#merrics for MLP
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("MLP Accuracy:",result2)

pkl_knn = "pickle_model_MFCC_MAC.pkl"
with open(pkl_knn, 'wb') as file:
    pickle.dump(mlp_model, file)