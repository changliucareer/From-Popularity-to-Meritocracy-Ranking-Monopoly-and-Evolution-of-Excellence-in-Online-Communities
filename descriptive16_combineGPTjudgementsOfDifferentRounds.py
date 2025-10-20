import os
import sys
#First print the current working directory
print("Current Working Directory", os.getcwd())
###############################################################
from toolFunctions import format_time, writeIntoLog, writeIntoResult,saveModel,savePlot
import time
import datetime
import glob
import multiprocessing as mp
import math
from collections import defaultdict
import pickle
import csv
import pandas as pd
import operator
import copy
from itertools import groupby
import re
import psutil
from scipy.sparse import csr_matrix, lil_matrix
import numpy as np
import torch
from tqdm import tqdm
from statistics import mean
from sklearn.model_selection import train_test_split
import sklearn
from CustomizedNN import LRNN_1layer, LRNN_1layer_bias, LRNN_1layer_bias_specify,LRNN_1layer_bias_withoutRankTerm
from torch.utils.data import IterableDataset, DataLoader
from itertools import cycle, islice
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix 
from sklearn.preprocessing import normalize
import random
import math
import scipy.stats
from adjustText import adjust_text

   
def main():

    t0=time.time()
    root_dir = os.getcwd()

    
    # load GPT judgement of all comm
    commName2GPTjudgement_allRounds = defaultdict()

    with open('GPTjudgementOfTopComms_prompt3_round1.dict', 'rb') as inputFile:
        commName2GPTjudgement_round1 = pickle.load( inputFile)

        for commName, d in commName2GPTjudgement_round1.items():
            commName2GPTjudgement_allRounds[commName] = {'knowledge': [d['knowledge']],
                                                         'experience': [d['experience']],
                                                         'belief': [d['belief']],
                                                         'opinion': [d['opinion']]}
    
    with open('GPTjudgementOfTopComms_prompt3_round2.dict', 'rb') as inputFile:
        commName2GPTjudgement_round2 = pickle.load( inputFile)
        for commName, d in commName2GPTjudgement_round2.items():
            commName2GPTjudgement_allRounds[commName] = {'knowledge': commName2GPTjudgement_allRounds[commName]['knowledge'] + [d['knowledge']],
                                                         'experience': commName2GPTjudgement_allRounds[commName]['experience'] +  [d['experience']],
                                                         'belief': commName2GPTjudgement_allRounds[commName]['belief'] + [d['belief']],
                                                         'opinion': commName2GPTjudgement_allRounds[commName]['opinion'] + [d['opinion']]}
    
    with open('GPTjudgementOfTopComms_prompt3_round3.dict', 'rb') as inputFile:
        commName2GPTjudgement_round3 = pickle.load( inputFile)
        for commName, d in commName2GPTjudgement_round3.items():
            commName2GPTjudgement_allRounds[commName] = {'knowledge': commName2GPTjudgement_allRounds[commName]['knowledge'] + [d['knowledge']],
                                                         'experience': commName2GPTjudgement_allRounds[commName]['experience'] +  [d['experience']],
                                                         'belief': commName2GPTjudgement_allRounds[commName]['belief'] + [d['belief']],
                                                         'opinion': commName2GPTjudgement_allRounds[commName]['opinion'] + [d['opinion']]}
    
    with open('GPTjudgementOfTopComms_prompt3_round4.dict', 'rb') as inputFile:
        commName2GPTjudgement_round4 = pickle.load( inputFile)
        for commName, d in commName2GPTjudgement_round4.items():
            commName2GPTjudgement_allRounds[commName] = {'knowledge': commName2GPTjudgement_allRounds[commName]['knowledge'] + [d['knowledge']],
                                                         'experience': commName2GPTjudgement_allRounds[commName]['experience'] +  [d['experience']],
                                                         'belief': commName2GPTjudgement_allRounds[commName]['belief'] + [d['belief']],
                                                         'opinion': commName2GPTjudgement_allRounds[commName]['opinion'] + [d['opinion']]}
    
    with open('GPTjudgementOfTopComms_prompt3_round5.dict', 'rb') as inputFile:
        commName2GPTjudgement_round5 = pickle.load( inputFile)
        for commName, d in commName2GPTjudgement_round5.items():
            commName2GPTjudgement_allRounds[commName] = {'knowledge': commName2GPTjudgement_allRounds[commName]['knowledge'] + [d['knowledge']],
                                                         'experience': commName2GPTjudgement_allRounds[commName]['experience'] +  [d['experience']],
                                                         'belief': commName2GPTjudgement_allRounds[commName]['belief'] + [d['belief']],
                                                         'opinion': commName2GPTjudgement_allRounds[commName]['opinion'] + [d['opinion']]}

    # save
    with open(f'commName2GPTjudgement_allRounds.dict', 'wb') as outputFile:
        pickle.dump(commName2GPTjudgement_allRounds, outputFile)
        print(f"saved commName2GPTjudgement_allRounds")

    # save as cvs
    csvfile = open('commName2GPTjudgement_allRounds.csv', 'w', newline='')
    writer = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow( ["commName","knowledge_round1", "experience_round1","belief_round1","opinion_round1",
                      "knowledge_round2", "experience_round2","belief_round2","opinion_round2",
                      "knowledge_round3", "experience_round3","belief_round3","opinion_round3",
                      "knowledge_round4", "experience_round4","belief_round4","opinion_round4",
                      "knowledge_round5", "experience_round5","belief_round5","opinion_round5"])
    for commName, d in commName2GPTjudgement_allRounds.items():
        writer.writerow( [commName, d["knowledge"][0], d["experience"][0],d["belief"][0],d["opinion"][0],
                          d["knowledge"][1], d["experience"][1],d["belief"][1],d["opinion"][1],
                          d["knowledge"][2], d["experience"][2],d["belief"][2],d["opinion"][2],
                          d["knowledge"][3], d["experience"][3],d["belief"][3],d["opinion"][3],
                          d["knowledge"][4], d["experience"][4],d["belief"][4],d["opinion"][4]])
    csvfile.close()

    commName2GPTjudgement_allRounds_averageProb = defaultdict()
    for commName, d in commName2GPTjudgement_allRounds.items():
        commName2GPTjudgement_allRounds_averageProb[commName] = {'knowledge': mean(d['knowledge']),
                                                                 'experience': mean(d['experience']),
                                                                 'belief': mean(d['belief']),
                                                                 'opinion': mean(d['opinion'])}
        
    # save
    with open(f'commName2GPTjudgement_allRounds_averageProb.dict', 'wb') as outputFile:
        pickle.dump(commName2GPTjudgement_allRounds_averageProb, outputFile)
        print(f"saved commName2GPTjudgement_allRounds_averageProb")

    # select round for selected comms
    selectedCommName2round = {
        "workplace.stackexchange" : 3,
        "travel.stackexchange" : 1,
        "worldbuilding.stackexchange": 2,
        "sqa.stackexchange": 2,
        "aviation.stackexchange": 1,
        "stats.stackexchange": 5,
        "retrocomputing.stackexchange": 1,
        "math.stackexchange":5,
        "softwareengineering.stackexchange": 3,
        "security.stackexchange": 3,
        "academia.meta.stackexchange": 1,
        "stats.meta.stackexchange": 2,
        "unix.meta.stackexchange": 3,
        "photo.meta.stackexchange": 5,
        "electronics.meta.stackexchange":1,
        "pt.meta.stackoverflow":2,
        "codereview.meta.stackexchange":2,
        "worldbuilding.meta.stackexchange":2,
        "meta.serverfault": 2,
        "softwareengineering.meta.stackexchange": 4,
        "physics.meta.stackexchange": 3,
        "scifi.meta.stackexchange":2
    }
    commName2GPTjudgement_selectedRound = defaultdict()
    for commName, d in commName2GPTjudgement_allRounds.items():
        if commName not in selectedCommName2round.keys():
            commName2GPTjudgement_selectedRound[commName] = {'knowledge': mean(d['knowledge']),
                                                                    'experience': mean(d['experience']),
                                                                    'belief': mean(d['belief']),
                                                                    'opinion': mean(d['opinion'])}
        else:
            round = selectedCommName2round[commName]
            commName2GPTjudgement_selectedRound[commName] = {'knowledge': d['knowledge'][round-1],
                                                                    'experience': d['experience'][round-1],
                                                                    'belief': d['belief'][round-1],
                                                                    'opinion': d['opinion'][round-1]}

    # save
    with open(f'commName2GPTjudgement_selectedRound.dict', 'wb') as outputFile:
        pickle.dump(commName2GPTjudgement_selectedRound, outputFile)
        print(f"saved commName2GPTjudgement_selectedRound")

    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('descriptive 16 Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
