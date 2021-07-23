
import pyaudio
import os
import wave
import pickle
from sys import byteorder
from array import array
from struct import pack
from sklearn.neural_network import MLPClassifier

from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model

from utils import extract_feature

import argparse
from predict import make_prediction
import speech_recognition as sr


Sr1 = sr.Recognizer()
Sr2 = sr.Recognizer()

THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 16000

SILENCE = 30


def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD

def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    "Trim the blank spots at the start and end"
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def add_silence(snd_data, seconds):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    r = array('h', [0 for i in range(int(seconds*RATE))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds*RATE))])
    return r

def record():
    """
    Record a word or words from the microphone and
    return the data as an array of signed shorts.
    Normalizes the audio, trims silence from the
    start and end, and pads with 0.5 seconds of
    blank sound to make sure VLC et al can play
    it without getting chopped off.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > SILENCE:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r

def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()
    print('Finish Recording')


if __name__ == "__main__":

    print("Please talk")

    filename = "Audio_testing/Unknow/test.wav"

    # record the file (start talking)
    #features = extract_feature(filename, mfcc=True, chroma=True, mel=True).reshape(1, -1)
    record_to_file(filename)



    parser = argparse.ArgumentParser(description='Audio Classification Predict')
    parser.add_argument('--model_fn', type=str, default='models_users/conv1d.h5', help='model to make predictions')
    #change between user(models_users/conv1d.h5) and keywords (models/conv1d.h5)
    #alterar pasta logs e clean na função make_prediction no predict.py


    parser.add_argument('--pred_fn', type=str, default='y_pred', help='fn to write predictions in logs dir')

    parser.add_argument('--src_dir', type=str, default='Audio_testing', help='directory containing wavfiles to predict')

    parser.add_argument('--dt', type=float, default=2, help='time in seconds to sample audio')

    parser.add_argument('--sr', type=int, default=16000, help='sample rate of clean audio')

    parser.add_argument('--threshold', type=str, default=20, help='threshold magnitude for np.int16 dtype')

    args, _ = parser.parse_known_args()


    result = make_prediction(args)



