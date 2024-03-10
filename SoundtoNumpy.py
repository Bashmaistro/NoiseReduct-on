from glob import glob
import pandas as pd
import librosa
import numpy    as  np
from openpyxl.workbook import Workbook

import pandas as pd
from glob import glob
import librosa

noisy_audio_files = glob('noisy_1sn/*.wav')[:1000]
clean_audio_files = glob('clean_1sn/*.wav')[:1000]

X_train = []
Y_train = []



for noisy_file, clean_file in zip(noisy_audio_files, clean_audio_files):
    noisy_data, sr = librosa.load(noisy_file)
    clean_data, sr = librosa.load(clean_file)




    print(noisy_file)
    print(clean_file)

    X_train.append(noisy_data)
    Y_train.append(clean_data)


np.save('X_train1000.npy', X_train)
np.save('Y_train1000.npy', Y_train)
