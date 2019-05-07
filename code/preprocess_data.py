import librosa
import pandas as pd
import pickle
import time


def load_sample(fname, data_path='../input/train_curated/', SR=22050):
    x, sr = librosa.load(data_path + '/' + fname)
    if sr != SR:
        raise Exception('Unexpected Sample Rate (not {}'.format(SR))
    return x


def get_logMel(y):
    mel = librosa.feature.melspectrogram(y, power=2)
    return (librosa.power_to_db(mel))


def get_tempogram(y):
    return (librosa.feature.tempogram(y))


wavs = pd.read_csv("../input/train_curated.csv")

mel = []
spec = []
tgram = []
fnames = []
tic = time.time()
for i, row in wavs.iterrows():
    y = load_sample(row.fname)
    mel.append(get_logMel(y))
    tgram.append(get_tempogram(y))
    fnames.append(row.fname)
    if (i % 500 == 0) or ((i + 1) == len(wavs)):
        print(i)
        print("File {} processed after {}s".format(i, time.time() - tic))
        with open('fname.pkl', 'wb') as fn:
            pickle.dump(fnames, fn, pickle.HIGHEST_PROTOCOL)
        with open('mel.pkl', 'wb') as fn:
            pickle.dump(mel, fn, pickle.HIGHEST_PROTOCOL)
        with open('tgram.pkl', 'wb') as fn:
            pickle.dump(tgram, fn, pickle.HIGHEST_PROTOCOL)
print(time.time() - tic)
