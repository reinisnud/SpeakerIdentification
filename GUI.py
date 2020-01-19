# import tkinter

# window = tkinter.Tk()
# # to rename the title of the window
# window.title("Speaker Recognition APP")
# # pack is used to show the object in the window
# label = tkinter.Label(window, text = "Speaker Recognition APP V1.0").pack()
# window.mainloop()
import tkinter
import sounddevice as sd
from scipy.io.wavfile import write    
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import random
import numpy
import os
import pickle
from DB_wav_reader import read_feats_structure
from SR_Dataset import read_MFB, ToTensorTestInput
from model.model import background_resnet
from identification import load_model, split_enroll_and_test, load_enroll_embeddings, perform_identification
import configure as c




log_dir = 'model_saved' # Where the checkpoints are saved
embedding_dir = 'enroll_embeddings' # Where embeddings are saved
test_dir = 'sorted\\test\\' # Where test features are saved

# Settings
use_cuda = True # Use cuda or not
embedding_size = 128 # Dimension of speaker embeddings
cp_num = 30 # Which checkpoint to use?
n_classes = 1211 # How many speakers in training data?
test_frames = 100 # Split the test utterance 

# Load model from checkpoint
model = load_model(use_cuda, log_dir, cp_num, embedding_size, n_classes)

# Get the dataframe for test DB
enroll_DB, test_DB = split_enroll_and_test(c.TEST_FEAT_DIR)

# Load enroll embeddings
embeddings = load_enroll_embeddings(embedding_dir)


def recordAudio():


    fs = 44100  # Sample rate
    seconds = 3  # Duration of recording

    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    write('output.wav', fs, myrecording)  # Save as WAV file 


    # Let's create the Tkinter window


# creating a function called DataCamp_Tutorial()
def rec():
    recordAudio()
    (rate,sig) = wav.read('output.wav')

    # get mfcc
    mfcc_feat = mfcc(sig,rate)
    dictt={}
    fbank_feat = logfbank(sig,rate)
    outputFile = "output.p"
    dictt['feat'] = fbank_feat
    dictt['label'] = folder
    pickle.dump( dictt, open( outputFile, "wb" ) )


    tkinter.Label(window, text = "Recorded, identifying").pack()
    
    spk_list = []
    # Perform the test 
    best_spk = perform_identification(use_cuda, model, embeddings, outputFile, test_frames, spk_list)
    tkinter.Label(window, text = "Predicted speaker:").pack()
    tkinter.Label(window, text = best_spk).pack()

def test():
    tkinter.Label(window, text = "Running a random speaker test... ").pack()
    # testspeakers = embeddings[:100]
    sk = random.randint(11,99)
    test_speaker = "id100" + str(sk)
    # test_speaker = random.choice(testspeakers)
    speaker_path = os.path.join(test_dir, test_speaker)
    for root, dirs, files in os.walk(speaker_path):
        if files:
            spoken_path = sorted(files)[-1]
   
    test_path = os.path.join(test_dir, test_speaker, spoken_path)
    
    # Perform the test 
    spk_list = []
    best_spk = perform_identification(use_cuda, model, embeddings, test_path, test_frames, spk_list)
    tkinter.Label(window, text = "Predicted speaker:").pack()
    tkinter.Label(window, text = best_spk).pack()
    tkinter.Label(window, text = "Actual speaker:").pack()
    tkinter.Label(window, text = test_speaker).pack()
    # if best_spk == test_speaker:
    #     positive=poisitve +1
    # else:
    #     negatove=negative + 1
    # print("true:" + positive + " false: " + negative)

def stats():
    tkinter.Label(window, text = "Training set 120525 utts (90.0%)").pack()
    tkinter.Label(window, text = "Validation set 13391 utts (10.0%)").pack()
    tkinter.Label(window, text = "Total 133916 utts").pack()
    tkinter.Label(window, text = "Number of classes (speakers): 1211").pack()
    tkinter.Label(window, text = "Accuracy: 92.51%").pack()





window = tkinter.Tk()
window.title("GUI")

tkinter.Button(window, text = "Record Audio for Speaker Identification", command = rec).pack()
tkinter.Button(window, text = "Test random speaker with a random test file", command = test).pack()
tkinter.Button(window, text = "Statistics", command = stats).pack()

window.mainloop()