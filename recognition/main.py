
from utils import best_path_improved, Lextree, State
import hmm
#import pydot
import numpy as np
from tqdm import tqdm
import librosa
import segk
from matplotlib import pyplot as plt
from segk import HMM
from mfcc_s import get_MFCC
import os
import SGMM
import pickle
from sgmm_p2 import find_true_seq
def save_model(models, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(models, file)


def load_model(filename):
    with open(filename, 'rb') as fr:
        model = pickle.load(fr)
    return model
def extract_recognition_result(sequence):
    out=''
    flag=True
    for char in sequence:

        if isinstance(char,State):
          #  print(char.name)
            if char.name.startswith('r') or char.name.startswith('s'):
                continue
            elif flag:
                flag=False
                out+=char.name[2]
        elif char=='NE':
            flag=True
    return out

def split_MFCC(mfcc, sequence_label,data):

    i=0
    for st in sequence_label[1:]:
        if not isinstance(st,State):
            continue
        elif st.name.startswith('silence'):
            i+=1
            continue
        else:
            if st.name in data.keys():
                data[st.name].append(mfcc[i,:])
            else:
                data[st.name]=[mfcc[i,:]]
            i+=1
    # for name, dts in data.items():
    #     means[name]=np.mean(np.array(dts),axis=0)
    #     vars[name]=np.vars(np.array(dts),axis=0)
    #     print(means['0'].shape)
   # print(data['0_state_0'])
   #  for k, v in data.items():
   #      print(f"{k}:{np.array(v).shape}")
    return data
  #  return means, vars

def Test():

    tree= load_model('lex_tree_final.pkl')
    audio_path = 'template_conitnue\\0123456789.wav'
    mfccs = get_MFCC(audio_path)

    best_path_result = best_path_improved(mfccs, tree, pruning_val=1.005)
    print(extract_recognition_result(best_path_result.sequence))
# 1.Path.sequence str -> list of states
# 2.构建 lextree 由定标签决定
# 3.best path -> {7-1: [[1,39], [1,39],...], }
def wer(reference, hypothesis):
    ref = list(reference)
    hyp = list(hypothesis)

    d = np.zeros((len(ref) + 1) * (len(hyp) + 1), dtype=np.uint8).reshape((len(ref) + 1, len(hyp) + 1))

    # Initialize the matrix for Levenshtein distance
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j

    # Compute the Levenshtein distance
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(d[i - 1][j] + 1,  # deletion
                          d[i][j - 1] + 1,  # insertion
                          d[i - 1][j - 1] + cost)  # substitution

    # Calculate WER
    wer_value = d[len(ref)][len(hyp)] / float(len(ref))
    return wer_value
tree= load_model('lex_tree.pkl')
total_wer=0
nums=0
for audio_path in os.listdir('.\\'):
    if audio_path.endswith('.wav'):
        nums+=1
        reference = audio_path.split('.')[0]
        mfccs = get_MFCC(audio_path)

        best_path_result = best_path_improved(mfccs, tree, pruning_val=1.005)
        hypothesis = extract_recognition_result(best_path_result.sequence)
        W = wer(reference,hypothesis)
        total_wer+=W
        print (reference)
        print(hypothesis)
        print(f"WER: {W}")
print("Average_WER:", total_wer/nums)
