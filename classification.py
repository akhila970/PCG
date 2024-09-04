import numpy as np
import pandas as pd
import os
import scipy.io.wavfile as wav
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from python_speech_features import mfcc
from sklearn.svm import SVC
from keras.preprocessing.sequence import pad_sequences
import audiomentations as AA
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import pickle
tqdm.pandas()

max_sound_duration=12


train_data = pd.read_csv("sounds_pcg_csv.csv")
print(len(train_data))


train_audio = []
train_labels = []
augmenter = AA.Compose([
    AA.AddGaussianNoise(p=0.5), 
    AA.TimeStretch(p=0.3),
    AA.PitchShift(p=0.3),
    AA.Shift(p=0.5),

])


num_augmented_samples = 80
label_encoder = LabelEncoder()
train_data['label_encoded'] = label_encoder.fit_transform(train_data['label'])

for x, label_encoded in tqdm(zip(train_data['fname'], train_data['label_encoded'])):
    sample_rate, audio_data = wav.read("sounds_both//" + x)
    max_samples = int(max_sound_duration * sample_rate)
    audio_data = audio_data[:max_samples]

    audio_data = np.array(audio_data, dtype=np.float32)

    mfcc_features = mfcc(audio_data, sample_rate,
                         numcep=13, nfilt=26, nfft=4000)
    train_audio.append(mfcc_features)
    train_labels.append(label_encoded)

    for _ in range(num_augmented_samples - 1):
        
        augmented_audio = augmenter(
            samples=audio_data, sample_rate=sample_rate)
        mfcc_features = mfcc(augmented_audio, sample_rate,
                             numcep=13, nfilt=26, nfft=4000)
        train_audio.append(mfcc_features)
        train_labels.append(label_encoded)

max_length = max(len(audio_data) for audio_data in train_audio)
train_audio = pad_sequences(train_audio, maxlen=max_length,
                            dtype='float32', padding='post', truncating='post')

train_labels = np.array(train_labels)

print(len(train_audio))
print(len(train_labels))

x_train, x_test, y_train, y_test = train_test_split(
    train_audio.reshape(len(train_audio), -1), train_labels, train_size=0.8, random_state=42
)


knn_classifier = KNeighborsClassifier(n_neighbors=4, metric='minkowski', p=2)
knn_classifier.fit(x_train, y_train)

with open('knn_classifier_test.pkl', 'wb') as f:
    pickle.dump(knn_classifier, f)

with open('y_test_test.pkl', 'wb') as f:
    pickle.dump(y_test, f)

with open('x_test_test.pkl', 'wb') as f:
    pickle.dump(x_test, f)



y_pred_knn = knn_classifier.predict(x_test.tolist())
ac_knn = accuracy_score(y_test, y_pred_knn)
print("KNN model accuracy is", ac_knn)