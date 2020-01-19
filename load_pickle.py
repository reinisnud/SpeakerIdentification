
import os
import pickle # For python3 

with open('testfile4.p', 'rb') as f:
    feat_and_label = pickle.load(f)
    print(feat_and_label)
    print()
