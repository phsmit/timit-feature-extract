from __future__ import print_function
import matplotlib.mlab
import numpy as np
from os.path import join
import os
import scipy.io.wavfile
import sys
import tempfile
import subprocess
from sklearn.preprocessing import normalize

def main(dir):
    wav_files = []
    phn_files = []

    for root, dirs, files in os.walk(dir):
        for f in files:
            if f.endswith('.wav'):
                wav_files.append(join(root, f))
            if f.endswith('.phn'):
                phn_files.append(join(root, f))


    train_samples = np.zeros((0,201),dtype=np.float64)
    train_labels = []
    for wav_file, phn_file in zip(sorted(wav_files),
                                  sorted(phn_files)):
        #print(wav_file)
        #print(phn_file)
        with tempfile.TemporaryFile() as fp:
            subprocess.check_call(['sox', wav_file, '-t', 'wav', '-'], stdout=fp)
            fp.seek(0)
            framerate, data = scipy.io.wavfile.read(fp)

            data_f = data / np.iinfo(data.dtype).max

            # what kind of window width and overlap should we use?
            spec, freqs, t = matplotlib.mlab.specgram(data_f, 400, Fs=1, window=np.hamming(400), noverlap=200)
            train_samples = np.append(train_samples, spec.T, axis=0)



            t = list(t)
            for line in open(phn_file):
                s, e, phn = line.split()
                while len(t) > 0 and t[0] < int(e):
                    train_labels.append(phn)
                    t = t[1:]

        while train_samples.shape[0] > len(train_labels):
            print(phn_file)
            train_labels.append("h#")


    train_samples = (train_samples - train_samples.mean(axis=0)) / train_samples.std(axis=0)

    print(train_samples.std(axis=0))

    print(train_samples.shape)
    print(len(train_labels))

    with open('out.labels', 'w') as fd:
        for label in train_labels:
            print(label, file=fd)

    np.save('out.features', train_samples)

if __name__ == "__main__":
    timit_dir = sys.argv[1]
    main(timit_dir)