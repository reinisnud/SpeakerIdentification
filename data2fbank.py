# from python_speech_features import mfcc
# from python_speech_features import delta
# from python_speech_features import logfbank
# import scipy.io.wavfile as wav

# (rate,sig) = wav.read("testwav.wav")
# mfcc_feat = mfcc(sig,rate)
# d_mfcc_feat = delta(mfcc_feat, 2)
# fbank_feat = logfbank(sig,rate)

# print(fbank_feat)


# This python script preforms an MFCC analysis of every .wav file in a specified directory and saves the filterbank
# energies for each file in a new directory (as a .csv).
#
# Please note: running this script multiple times will overwrite earlier analyses
#
# Requires python_speech_features. Documentation: https://github.com/jameslyons/python_speech_features)
# Requires scipy. Scipy documentation: https://www.scipy.org/install.html
#
# Script written by Rachael Tatman (rachael.tatman@gmail.com), supported by National Science Foundation grant DGE-1256082

# import things we're going to need
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import numpy
import os
import pickle

# directory where we your .wav files are
directoryName = "C:\\Users\\reini\\Desktop\\SpeakerRecognition_tutorial-master\\wav" # put your own directory here
# directory to put our results in, you can change the name if you like
resultsDirectory = "C:\\Users\\reini\\Desktop\\SpeakerRecognition_tutorial-master" + "\\MFCCresults"

# make a new folder in this directory to save our results in
if not os.path.exists(resultsDirectory):
    os.makedirs(resultsDirectory)

# get MFCCs for every .wav file in our specified directory 
for folder in os.listdir(directoryName):
	print(folder)
	i=0

	for subfolder in os.listdir(directoryName + "\\" + folder):
		for filename in os.listdir(directoryName + "\\" + folder + "\\" + subfolder):
		    if filename.endswith('.wav'): # only get MFCCs from .wavs
		        # read in our file
		        (rate,sig) = wav.read(directoryName + "\\" + folder + "\\" + subfolder + "\\" + filename)

		        # get mfcc
		        mfcc_feat = mfcc(sig,rate)
		        dictt = {}

		        # get filterbank energies
		        fbank_feat = logfbank(sig,rate)
		        i=i+1
		        filename = folder + "_{}".format(i)
		        # create a file to save our results in
		        if not os.path.exists(resultsDirectory + "\\" + folder):
		            os.makedirs(resultsDirectory + "\\" + folder)
		        outputFile = resultsDirectory + "\\" + os.path.splitext(folder + "\\" + filename)[0] + ".p"
		        dictt['feat'] = fbank_feat
		        dictt['label'] = folder

		        pickle.dump( dictt, open( outputFile, "wb" ) )
		        #file = open(outputFile, 'w+') # make file/over write existing file
		        #numpy.savetxt(file, fbank_feat, delimiter=",") #save MFCCs as .csv
		        #file.close() # close file