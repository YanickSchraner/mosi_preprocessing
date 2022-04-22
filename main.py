import subprocess
from collections import defaultdict

from constants import SDK_PATH, DATA_PATH, WORD_EMB_PATH, CACHE_PATH
import sys

if SDK_PATH is None:
    print("SDK path is not specified! Please specify first in constants/paths.py")
    exit(0)
else:
    sys.path.append(SDK_PATH)

import mmsdk
import os
import re
import numpy as np
from mmsdk import mmdatasdk as md
from subprocess import check_call, CalledProcessError

# create folders for storing the data
if not os.path.exists(DATA_PATH):
    check_call(' '.join(['mkdir', '-p', DATA_PATH]), shell=True)

# download highlevel features, low-level (raw) data and labels for the dataset MOSI
# if the files are already present, instead of downloading it you just load it yourself.
# here we use CMU_MOSI dataset as example.

DATASET = md.cmu_mosi


def get_file_paths():
    wav_files = [os.path.join(DATA_PATH, 'Raw', 'Audio', 'WAV_16000', 'Full', file) for file in
                 os.listdir(os.path.join(DATA_PATH, 'Raw', 'Audio', 'WAV_16000', 'Full')) if file.endswith('.wav')]
    transcript_files = [os.path.join(DATA_PATH, 'Raw', 'Transcript', 'Full', file) for file in
                        os.listdir(os.path.join(DATA_PATH, 'Raw', 'Transcript', 'Full'))]
    video_files = [os.path.join(DATA_PATH, 'Raw', 'Video', 'Full', file) for file in
                   os.listdir(os.path.join(DATA_PATH, 'Raw', 'Video', 'Full'))]
    id_to_paths = defaultdict(dict)
    for wav_file, transcript_file, video_file in zip(wav_files, transcript_files, video_files):
        id_to_paths[os.path.basename(wav_file).split('.')[0]]['audio'] = wav_file
        id_to_paths[os.path.basename(transcript_file).split('.')[0]]['text'] = transcript_file
        id_to_paths[os.path.basename(video_file).split('.')[0]]['video'] = video_file
    return id_to_paths


def calculate_alignments(paths, out_path):
    for id, path_dict in paths.items():
        print(id)
        try:
            file_out_path = os.path.join(out_path, id)
            check_call(
                ' '.join(['python', './external/p2fa/p2fa/align.py', path_dict['audio'], path_dict['text'], file_out_path, '--sampling_rate=16000', '--verbose=0']),
                shell=True)
        except CalledProcessError as e:
            print(e)
            continue


def unify_modality_frequencies():
    pass


def open_face_features():
    pass


def covarep_features():
    pass


def get_glove_embedding_dict() -> dict:
    with open(WORD_EMB_PATH, 'r') as f:
        lines = f.readlines()
        glove_dict = {}
        for line in lines:
            line = line.strip()
            line = line.split(' ')
            glove_dict[line[0]] = np.array(line[1:]).astype(np.float32)
        return glove_dict


def main():
    paths = get_file_paths()
    calculate_alignments(paths, os.path.join(DATA_PATH, 'alignments'))


if __name__ == '__main__':
    main()
