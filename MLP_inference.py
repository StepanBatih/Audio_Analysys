from python_speech_features import mfcc
from sklearn.model_selection import train_test_split
import scipy.io.wavfile as wav
import pathlib
import pickle 
import numpy as np
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
from pathlib import Path

import wave


pkl_knn = "pickle_model_mlp_MFCC0511.pkl"

with open(pkl_knn, 'rb') as file:
    pickle_model_mlp = pickle.load(file)
    

wav_f="/Users/stepan_batih/ML/TEST/test_cat/1.wav" # 


'''
input_folder = Path("D:/ML/TEST/test_silence" )
input_files = list(input_folder.glob("*.wav"))'''

def inference(input_file):
    print(input_file)
    wlen = 0.5
    wstep = 1
    arr_x=[]

    rate,sig = wav.read(input_file)
    mfcc_feat = mfcc(sig,rate,winlen=wlen,winstep=wstep,nfilt=52,nfft=22050,numcep=52).astype(np.float32)
    temp = arr_x.append(mfcc_feat)
    inf_x = np.concatenate(arr_x)
    
    y_prediction = pickle_model_mlp.predict(inf_x)
    print(y_prediction)
    unique, counts = np.unique(y_prediction, return_counts=True)
    sum_pred = sum(counts).astype(float)

    result=dict(zip(unique, counts))
    #print(result)
    prediction = max(result, key=result.get)
    #print(prediction)
    per = (max(counts) / sum_pred) * 100
    #print(per)    
    #print('Всього вікон :',sum_pred)
    
    with wave.open(str(input_file), 'rb') as f:  # Convert to string
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)

    #print(f"Довжина файлу: {duration:.2f} секунд")

    
    return (prediction,'{:.2f}'.format(per),'%')

   
print(inference(wav_f))

'''
for i, file in enumerate(input_files):
     # Визначаємо назву вихідного файлу
    print(inference(file))
    print()'''
    
