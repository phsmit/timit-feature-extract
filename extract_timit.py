from __future__ import print_function
import matplotlib.mlab
import numpy as np
from os.path import join
import os
import scipy.io.wavfile
import sys
import tempfile
import subprocess
import pandas as pd

WINDOW_WIDTH=400
WINDOW_NOVERLAP=200


def main(dir, hdf_file):
    if os.path.exists(hdf_file):
        os.remove(hdf_file)

    store = pd.HDFStore(hdf_file)

    y_frames = []
    meta_frames = []

    wav_files = []
    phn_files = []

    for root, dirs, files in os.walk(dir):
        for f in files:
            if f.endswith('.wav'):
                wav_files.append(join(root, f))
            if f.endswith('.phn'):
                phn_files.append(join(root, f))

    for file_id, files in  enumerate(zip(sorted(wav_files),
                                  sorted(phn_files))):
        wav_file, phn_file = files
        print(wav_file)

        with tempfile.SpooledTemporaryFile() as fp:
            subprocess.check_call(['sox', wav_file, '-t', 'wav', '-'], stdout=fp)
            fp.seek(0)
            framerate, data = scipy.io.wavfile.read(fp)

            data_f = data / np.iinfo(data.dtype).max

            # what kind of window width and overlap should we use?
            spec, freqs, t = matplotlib.mlab.specgram(data_f,
                                                      WINDOW_WIDTH,
                                                      Fs=1,
                                                      window=np.hamming(WINDOW_WIDTH),
                                                      noverlap=WINDOW_NOVERLAP)

            indices = pd.MultiIndex.from_tuples([(file_id, i) for i in range(spec.shape[1])], names=['file','frame'])

            store.append('X', pd.DataFrame(spec.T.astype(np.float32), index=indices, columns=range(WINDOW_WIDTH//2+1)))

            labels = []
            t = list(t)
            for line in open(phn_file):
                s, e, phn = line.split()
                while len(t) > 0 and t[0] < int(e):
                    labels.append(phn)
                    t = t[1:]
            while spec.shape[1] > len(labels):
                labels.append("h#")

            y_frames.append(pd.DataFrame(labels, columns=["timit_label"], index=indices))

            _, set_type, dialect_region, speaker, sentence = wav_file[wav_file.rfind("timit"):].split('/')

            file_info={
                "speaker": speaker,
                "male": speaker[0] == 'M',
                "dialect_region":dialect_region,
                "frame_count": spec.shape[1],
                "sent_type": sentence[:2],
                "sent_id": int(sentence[2:sentence.find('.')]),
                "train": set_type == "train",
                "core_test": speaker in ('DAB0', 'WBT0', 'ELC0', 'TAS1', 'WEW0', 'PAS0', 'JMP0', 'LNT0', 'PKT0', 'LLL0', 'TLS0', 'JLM0', 'BPM0', 'KLT0', 'NLP0', 'CMJ0', 'JDH0', 'MGD0', 'GRT0', 'NJM0', 'DHC0', 'JLN0', 'PAM0', 'MLD0')
            }
            meta_frames.append(pd.DataFrame(file_info, index=[file_id]))

    store.append('y',pd.concat(y_frames))

    meta_df = pd.concat(meta_frames)
    meta_df.info()
    store.append('meta', meta_df)


if __name__ == "__main__":
    timit_dir = sys.argv[1]
    hdf_file = sys.argv[2]
    main(timit_dir, hdf_file)
