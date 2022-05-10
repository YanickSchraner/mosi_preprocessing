import difflib
import subprocess
from collections import defaultdict

from moviepy.video.io.VideoFileClip import VideoFileClip
from oct2py import octave
from pydub import AudioSegment

from constants import SDK_PATH, DATA_PATH, WORD_EMB_PATH, OPENFACE_FEATURE_EXTRACTION, COVAREP_PATH
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

ANNOTATED_TRANSCRIPT_REGEX = re.compile(r'[0-9]*_DELIM_')


def get_file_paths():
    wav_files = [os.path.join(DATA_PATH, 'Raw', 'Audio', 'WAV_16000', 'Full', file) for file in
                 os.listdir(os.path.join(DATA_PATH, 'Raw', 'Audio', 'WAV_16000', 'Full')) if file.endswith('.wav')]
    transcript_files = [os.path.join(DATA_PATH, 'Raw', 'Transcript', 'Full', file) for file in
                        os.listdir(os.path.join(DATA_PATH, 'Raw', 'Transcript', 'Full'))]
    transcript_segmented_files = [os.path.join(DATA_PATH, 'Raw', 'Transcript', 'Segmented', file) for file in
                                  os.listdir(os.path.join(DATA_PATH, 'Raw', 'Transcript', 'Segmented'))]
    video_files = [os.path.join(DATA_PATH, 'Raw', 'Video', 'Full', file) for file in
                   os.listdir(os.path.join(DATA_PATH, 'Raw', 'Video', 'Full'))]
    id_to_paths = defaultdict(dict)
    for wav_file, transcript_file, video_file, trans_seg_file in zip(wav_files, transcript_files, video_files,
                                                                     transcript_segmented_files):
        id_to_paths[os.path.basename(wav_file).split('.')[0]]['audio'] = wav_file
        id_to_paths[os.path.basename(transcript_file).split('.')[0]]['text'] = transcript_file
        id_to_paths[os.path.basename(video_file).split('.')[0]]['video'] = video_file
        id_to_paths[os.path.basename(trans_seg_file).split('.')[0]]['segmented_text'] = trans_seg_file
    return id_to_paths


def calculate_alignments(paths, out_path):
    for id, path_dict in paths.items():
        try:
            file_out_path = os.path.join(out_path, id)
            if not os.path.exists(file_out_path):
                check_call(
                    ' '.join(
                        ['python', './external/p2fa/align.py', path_dict['audio'], path_dict['text'], file_out_path,
                         '--sampling_rate=16000', '--verbose=0']),
                    shell=True)
            paths[id]['alignments'] = file_out_path
        except CalledProcessError as e:
            print(e)
            continue
    return paths


def get_word_timings(path_dict):
    words = []
    word_start_end_times = []
    with open(path_dict['alignments'], 'r') as f:
        lines = f.readlines()[3:]  # The first 3 lines are not needed
        # Get the start and end times of each word
        for idx in range(0, len(lines), 3):
            word = lines[idx + 2].split()[0].replace('"', '').lower()
            # Ignore pauses
            if word == 'sp':
                continue
            words.append(word)
            start_time, end_time = float(lines[idx].split()[0]), float(lines[idx + 1].split()[0])
            word_start_end_times.append((start_time, end_time))
    return words, word_start_end_times


def get_sentence_timings(path_dict, words, word_start_end_times):
    sentence_start_end_times = []
    # Filter sentence based on annotated transcript
    with open(path_dict['segmented_text'], 'r') as f:
        for line in f.readlines():
            words_sequence = ' '.join(words).lower()
            words_annotated = ' '.join(line.lower().split()[1:])
            # find overlapping sequence of words and words in annotated transcript
            d = difflib.SequenceMatcher(None, words_sequence, words_annotated)
            match = max(d.get_matching_blocks(), key=lambda x: x.size)

            # Get the time stamps of the sentence
            first_word_in_sentence_index = match.a
            last_word_in_sentence_index = first_word_in_sentence_index + match.size - 1

            # Convert sequence index to word index
            first_word_in_sentence_index = len(words_sequence[:first_word_in_sentence_index].split()) - 1
            last_word_in_sentence_index = len(words_sequence[:last_word_in_sentence_index].split()) - 1
            if first_word_in_sentence_index < 0:
                continue
            if last_word_in_sentence_index < 0:
                continue
            if first_word_in_sentence_index >= last_word_in_sentence_index:
                continue

            sentence_start_time = word_start_end_times[first_word_in_sentence_index][0]
            sentence_end_time = word_start_end_times[min(last_word_in_sentence_index, len(word_start_end_times) - 1)][1]

            if sentence_start_time == sentence_end_time:
                print('Sentence start and end time are same for sentence: {}'.format(words_annotated))
                print(words_sequence)
                print(match)
                continue
            sentence_start_end_times.append((sentence_start_time, sentence_end_time))
    return sentence_start_end_times


def create_aligned_data(paths):
    for id, path_dict in paths.items():
        already_dropped_phoneme_alignments = False
        with open(path_dict['alignments'], 'r') as f:
            already_dropped_phoneme_alignments = not f.readline().strip().startswith('File type')
        if not already_dropped_phoneme_alignments:
            # Delete phoneme level alignments
            check_call(['sed', '-i', '-e', '0,/"word"/d', path_dict['alignments']])
        words, word_start_end_times = get_word_timings(path_dict)
        sentence_start_end_times = get_sentence_timings(path_dict, words, word_start_end_times)

        for idx, (sentence_start_time, sentence_end_time) in enumerate(sentence_start_end_times):
            # Create the aligned data
            segment_audio_path = create_audio_segment(path_dict['audio'], sentence_start_time, sentence_end_time, idx)
            if 'audio_segments' not in paths[id].keys():
                paths[id]['audio_segments'] = []
            paths[id]['audio_segments'].append(segment_audio_path)
            video_segment_path = create_video_segment(path_dict['video'], sentence_start_time, sentence_end_time, idx)
            if 'video_segments' not in paths[id].keys():
                paths[id]['video_segments'] = []
            paths[id]['video_segments'].append(video_segment_path)
    return paths


def create_audio_segment(audio_path, start_time, end_time, idx):
    audio_segment = AudioSegment.from_wav(audio_path)
    audio_segment = audio_segment[int(start_time * 1000):int(end_time * 1000)]
    new_audio_path = audio_path
    new_audio_path = new_audio_path.replace('Full', 'Aligned')
    new_audio_path = new_audio_path.replace('.wav', f'_{idx}.wav')
    if not os.path.exists(new_audio_path):
        os.makedirs(os.path.dirname(new_audio_path), exist_ok=True)
        audio_segment.export(new_audio_path, format='wav')
    return new_audio_path


def create_video_segment(video_path, start_time, end_time, idx):
    video_segment = VideoFileClip(video_path)
    video_segment = video_segment.subclip(start_time, end_time)
    new_video_path = video_path
    new_video_path = new_video_path.replace('Full', 'Aligned')
    new_video_path = new_video_path.replace('.mp4', f'_{idx}.mp4')
    if not os.path.exists(new_video_path):
        os.makedirs(os.path.dirname(new_video_path), exist_ok=True)
        video_segment.write_videofile(new_video_path, fps=video_segment.fps, verbose=False, logger=None)
    return new_video_path


def get_glove_embedding_dict() -> dict:
    with open(WORD_EMB_PATH, 'r') as f:
        lines = f.readlines()
        glove_dict = {}
        for line in lines:
            line = line.strip()
            line = line.split(' ')
            glove_dict[line[0]] = np.array(line[1:]).astype(np.float32)
        return glove_dict


def generate_language_features(paths):
    glove_embeddings = get_glove_embedding_dict()
    for id, path_dict in paths.items():
        with open(path_dict['segmented_text'], 'r') as f:
            sentences = f.readlines()
            words_in_sentences = [ANNOTATED_TRANSCRIPT_REGEX.sub('', sentence.strip()).split() for sentence in
                                  sentences]
            # Words not part of the dictionary are ignored
            glove_embeddings_in_sentences = [glove_embeddings[word.lower()] for sentence in words_in_sentences for word
                                             in
                                             sentence if word in glove_embeddings.keys()]
            for i in range(len(glove_embeddings_in_sentences)):
                embeddings = np.array(glove_embeddings_in_sentences[i])
                path = path_dict['text'].replace('Full', 'Aligned')
                path = path.replace('.txt', f'_{i}.npy')
                os.makedirs(os.path.dirname(path), exist_ok=True)
                np.save(path, embeddings)


def main():
    paths = get_file_paths()
    paths = calculate_alignments(paths, os.path.join(DATA_PATH, 'alignments'))
    paths = create_aligned_data(paths)
    generate_language_features(paths)


if __name__ == '__main__':
    main()
