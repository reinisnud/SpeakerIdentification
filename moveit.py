# from python_speech_features import mfcc

# import things we're going to need
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import numpy
import shutil
import os
import random
import pickle

# directory where we your .wav files are
# directory to put our results in, you can change the name if you like
directoryName = "C:\\Users\\reini\\Desktop\\SpeakerRecognition_tutorial-master" + "\\MFCCresults"
newdirectory = "C:\\Users\\reini\\Desktop\\SpeakerRecognition_tutorial-master\\sorted"
train = "C:\\Users\\reini\\Desktop\\SpeakerRecognition_tutorial-master\\sorted\\train"
test = "C:\\Users\\reini\\Desktop\\SpeakerRecognition_tutorial-master\\sorted\\test"
# make a new folder in this directory to save our results in
if not os.path.exists(train):
    os.makedirs(train)

if not os.path.exists(newdirectory):
    os.makedirs(newdirectory)

if not os.path.exists(test):
    os.makedirs(test)

# get MFCCs for every .wav file in our specified directory 
for folder in os.listdir(directoryName):
    print(folder)
    i=0

    for filename in os.listdir(directoryName + "\\" + folder):
        if filename.endswith('.p'): # only get MFCCs from .wavs
 # read in our file
            if not os.path.exists(train + "\\" + folder):
                os.makedirs(train + "\\" + folder)
            if not os.path.exists(test + "\\" + folder):
                os.makedirs(test + "\\" + folder)

            outcome = random.randint(1,10)
            if outcome >= 2:
                shutil.move(directoryName + "\\" + folder + "\\" + filename, train + "\\" + folder + "\\" + filename)
            else:
                shutil.move(directoryName + "\\" + folder + "\\" + filename, test + "\\" + folder + "\\" + filename)