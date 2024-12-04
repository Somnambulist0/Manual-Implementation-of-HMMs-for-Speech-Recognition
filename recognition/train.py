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
def train():
    flag=True


    cost=np.inf
    #HMM_list={1: hmm, 2:hmm}
    ## INITIALIZE
    HMM_list = SGMM.build_shmm()
    silence_data = []
    file = f"template/silence"
    for wav_name in os.listdir(file):
        if wav_name.endswith(".wav"):
            f = os.path.join(f"template/silence", wav_name)
            silence_data.append(get_MFCC(f))
    silence_HMM = SGMM.train_model_sghmm(silence_data, 2)
    lex_tree = Lextree(HMM_list,silence_HMM=silence_HMM,silence=True)
    lex_tree.create_states_for_digits()
   # lex_tree = load_model('lextree_ol.pkl')

    save_model(lex_tree,'lex_tree_origin.pkl')
    ## TRAIN
    costs=[]
    for i in tqdm(range(15)):

        new_data={}
        new_cost=0
        n=0
        means,vars={},{}
        for i,samples in enumerate(os.listdir("template_conitnue")):

            if samples.endswith('.wav'):
                name= samples[:-4].split('_')[0]
              #  print('name',name)

                lex_tree.create_states_for_digits(name)

                n=i
                audio_path = f'template_conitnue/{samples}'
                mfccs=get_MFCC(audio_path)
                # 使用

                best_path_result = best_path_improved(mfccs,LexTree=lex_tree, pruning_val=1.005)
              #  print('path:',extract_recognition_result(best_path_result.sequence))
                new_cost+=best_path_result.dtw

                seq=best_path_result.sequence
                new_data= split_MFCC(mfcc=mfccs,sequence_label=seq,data=new_data)
        for name, dts in new_data.items():
            means[name]=np.mean(np.array(dts),axis=0)
            vars[name]=np.var(np.array(dts),axis=0)

        for name, data in means.items():
            lex_tree.trained_hmms[name[0]].means[int(name[-1])]=data
        for name, data in vars.items():
            lex_tree.trained_hmms[name[0]].var[int(name[-1])] = data

        #save_model(lex_tree, f'lex_tree{int(new_cost/n)}.pkl')
        costs.append(new_cost/n)
      #  print('cur_err:', new_cost/n)
        if not ((cost-new_cost>(2*n)) or (new_cost>cost)):
            break
        cost=new_cost
    lex_tree.create_states_for_digits()
    save_model(lex_tree,'lex_tree.pkl')
train()
