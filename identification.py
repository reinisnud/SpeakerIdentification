import torch
import torch.nn.functional as F
from torch.autograd import Variable

import pandas as pd
import math
import os
import configure as c

from DB_wav_reader import read_feats_structure
from SR_Dataset import read_MFB, ToTensorTestInput
from model.model import background_resnet

def load_model(use_cuda, log_dir, cp_num, embedding_size, n_classes):
    model = background_resnet(embedding_size=embedding_size, num_classes=n_classes)
    if use_cuda:
        model.cuda()
    print('=> loading checkpoint')
    # original saved file with DataParallel
    checkpoint = torch.load(log_dir + '/checkpoint_' + str(cp_num) + '.pth')
    # create new OrderedDict that does not contain `module.`
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def split_enroll_and_test(dataroot_dir):
    DB_all = read_feats_structure(dataroot_dir)
    enroll_DB = pd.DataFrame()
    test_DB = pd.DataFrame()
    
    enroll_DB = DB_all[DB_all['filename'].str.contains('enroll.p')]
    test_DB = DB_all[DB_all['filename'].str.contains('test.p')]
    
    # Reset the index
    enroll_DB = enroll_DB.reset_index(drop=True)
    test_DB = test_DB.reset_index(drop=True)
    return enroll_DB, test_DB

def load_enroll_embeddings(embedding_dir):
    embeddings = {}
    for f in os.listdir(embedding_dir):
        spk = f.replace('.pth','')
        # Select the speakers who are in the 'enroll_spk_list'
        embedding_path = os.path.join(embedding_dir, f)
        tmp_embeddings = torch.load(embedding_path)
        embeddings[spk] = tmp_embeddings
        
    return embeddings

def get_embeddings(use_cuda, filename, model, test_frames):
    input, label = read_MFB(filename) # input size:(n_frames, n_dims)
    
    tot_segments = math.ceil(len(input)/test_frames) # total number of segments with 'test_frames' 
    activation = 0
    with torch.no_grad():
        for i in range(tot_segments):
            temp_input = input[i*test_frames:i*test_frames+test_frames]
            
            TT = ToTensorTestInput()
            temp_input = TT(temp_input) # size:(1, 1, n_dims, n_frames)
    
            if use_cuda:
                temp_input = temp_input.cuda()
            temp_activation,_ = model(temp_input)
            activation += torch.sum(temp_activation, dim=0, keepdim=True)
    
    activation = l2_norm(activation, 1)
                
    return activation

def l2_norm(input, alpha):
    input_size = input.size()  # size:(n_frames, dim)
    buffer = torch.pow(input, 2)  # 2 denotes a squared operation. size:(n_frames, dim)
    normp = torch.sum(buffer, 1).add_(1e-10)  # size:(n_frames)
    norm = torch.sqrt(normp)  # size:(n_frames)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
    output = output * alpha
    return output

def perform_identification(use_cuda, model, embeddings, test_filename, test_frames, spk_list):
    test_embedding = get_embeddings(use_cuda, test_filename, model, test_frames)
    max_score = -10**8
    speaker_count = 100
    best_spk = None
    i=0
    for spk in embeddings: #spk_list
        if i < speaker_count:
            score = F.cosine_similarity(test_embedding, embeddings[spk])
            score = score.data.cpu().numpy() 
            if score > max_score:
                max_score = score
                best_spk = spk
        i=i+1
    #print("Speaker identification result : %s" %best_spk)
    true_spk = test_filename.split('\\')[-2]
    print("\n=== Speaker identification ===")
    print("True speaker : %s\nPredicted speaker : %s\nResult : %s\n" %(true_spk, best_spk, true_spk==best_spk))
    
    print(max_score)
    return best_spk

def main():
    
    log_dir = 'model_saved' # Where the checkpoints are saved
    embedding_dir = 'enroll_embeddings' # Where embeddings are saved
    test_dir = 'sorted\\test\\' # Where test features are saved
    
    # Settings
    use_cuda = True # Use cuda or not
    embedding_size = 128 # Dimension of speaker embeddings
    cp_num = 23 # Which checkpoint to use?
    n_classes = 1211 # How many speakers in training data?
    test_frames = 100 # Split the test utterance 

    # Load model from checkpoint
    model = load_model(use_cuda, log_dir, cp_num, embedding_size, n_classes)

    # Get the dataframe for test DB
    enroll_DB, test_DB = split_enroll_and_test(c.TEST_FEAT_DIR)
    
    # Load enroll embeddings
    embeddings = load_enroll_embeddings(embedding_dir)
    
    """ Test speaker list
    '103F3021', '207F2088', '213F5100', '217F3038', '225M4062',  id10001
Save the embeddings for id10002
Save the embeddings for id10003
Save the embeddings for id10004
Save the embeddings for id10005
Save the embeddings for id10006
Save the embeddings for id10007
Save the embeddings for id10008
Save the embeddings for id10009
Save the embeddings for id10010
Save the embeddings for id10011
Save the embeddings for id10012
Save the embeddings for id10013
Save the embeddings for id10014
    '229M2031', '230M4087', '233F4013', '236M3043', '240M3063'
    """ 
    
    spk_list = ['id10001', 'id10002', 'id10003', 'id10004', 'id10005',\
    'id10006', 'id10007', 'id10008', 'id10009', 'id10010', 'id10011' , 'id10012', 'id10013', 'id10014']
    
    # Set the test speaker
    test_speaker = 'id10010' 
    speaker_path = os.path.join(test_dir, test_speaker)
    for root, dirs, files in os.walk(speaker_path):
        if files:
            spoken_path = sorted(files)[-1]
   
    test_path = os.path.join(test_dir, test_speaker, spoken_path)
    
    # Perform the test 
    best_spk = perform_identification(use_cuda, model, embeddings, test_path, test_frames, spk_list)

if __name__ == '__main__':
    main()