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
import json

def myFun_parallel(commName, commDir, return_dict):
    print(f"comm {commName} running on {mp.current_process().name}")

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data files
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')

    GPTresponse_files_directory = os.path.join(commDir, r'GPTresponse_folder')
    if not os.path.exists(GPTresponse_files_directory):
        print("no GPTresponse_files_directory, skip")
        return

    my_model = "gpt-4-turbo-preview"
    try:
        # get total voteCount
        with open('GPTresponse_folder/descriptive14_prompt3_round5_'+my_model+'.json') as json_file:
            GPT_judgement,res_text = json.load(json_file)
        
        for cat, prob in GPT_judgement.items():
            if prob == None:
                print(f"Given {res_text}, Enter prob of {cat}:")
                humanInput = input("prob = ")
                if humanInput != 'None':
                    GPT_judgement[cat] = float(humanInput)
        return_dict[commName] = GPT_judgement
    except Exception as e:
        print(f"{e} for {commName}")

def myFun(commName, commDir):
    print(f"comm {commName} running on {mp.current_process().name}")

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data files
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')

    GPTresponse_files_directory = os.path.join(commDir, r'GPTresponse_folder')
    if not os.path.exists(GPTresponse_files_directory):
        print("no GPTresponse_files_directory, skip")
        return

    my_model = "gpt-4-turbo-preview"
    try:
        # get total voteCount
        with open('GPTresponse_folder/descriptive14_prompt3_round5_'+my_model+'.json') as json_file:
            GPT_judgement,res_text = json.load(json_file)
        
        for cat, prob in GPT_judgement.items():
            if prob == None:
                print(f"Given {res_text}, Enter prob of {cat}:")
                humanInput = input("prob = ")
                if humanInput != 'None':
                    GPT_judgement[cat] = float(humanInput)
        return commName, GPT_judgement
    except Exception as e:
        print(f"{e} for {commName}")
   
def main():

    t0=time.time()
    root_dir = os.getcwd()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("sorted CommDir loaded.")

    ## Load top 120 selected comm names for Map
    with open('topSelectedCommNames_descriptive.dict', 'rb') as inputFile:
        topSelectedCommNames = pickle.load( inputFile)
    print("seleted comm names loaded.")

    """
    # use shared variable to communicate among all comm's process
    manager = mp.Manager()
    return_dict = manager.dict() # to save the used train mode (wholebatch or minibatch) of each community

    # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for commIndex, tup in enumerate(commDir_sizes_sortedlist[:359]):
        commName = tup[0]
        commDir = tup[1]

        if commName not in topSelectedCommNames:
            print(f"{commName} is not seleted, skip")
            continue
        
        try:
            p = mp.Process(target=myFun, args=(commName,commDir,return_dict))
            p.start()
        except Exception as e:
            print(e)
            pscount = sum(1 for proc in psutil.process_iter() if proc.name() == 'python3')
            print(f"current python3 processes count {pscount}.")
            sys.exit()

        processes.append(p)
        if len(processes)==24:
            # make sure all p finish before main process finish
            for p in processes:
                p.join()
                finishedCount +=1
                print(f"finished {finishedCount} comm.")
            # clear processes
            processes = []
    
    # join the last batch of processes
    if len(processes)>0:
        # make sure all p finish before main process finish
        for p in processes:
            p.join()
            finishedCount +=1
            print(f"finished {finishedCount} comm.")
    
    print(f"got {len(return_dict)} comms' GPT judgements, converting to normal dict...")
    # save 
    return_normalDict = defaultdict()
    for commName, d in return_dict.items():
        return_normalDict[commName] = {'knowledge':d['knowledge'], 'experience':d['experience'], 'belief':d['belief'], 'opinion':d['opinion']}
    os.chdir(root_dir) # go back to root directory
    with open('GPTjudgementOfTopComms_prompt3_round5.dict', 'wb') as outputFile:
        pickle.dump(return_normalDict, outputFile)
        print(f"saved return_normalDict")
    """

    return_dict = defaultdict()
    finishedCount = 0
    for commIndex, tup in enumerate(commDir_sizes_sortedlist):
        commName = tup[0]
        commDir = tup[1]

        if commName not in topSelectedCommNames:
            print(f"{commName} is not seleted, skip")
            continue
    
        commName, GPT_judgement = myFun(commName, commDir)
        return_dict[commName] = GPT_judgement
    
    os.chdir(root_dir) # go back to root directory
    with open('GPTjudgementOfTopComms_prompt3_round5.dict', 'wb') as outputFile:
        pickle.dump(return_dict, outputFile)
        print(f"saved return_normalDict")

    import csv
    print(f"start to save the results as csv...")
    with open('GPTjudgementOfTopComms_prompt3_round5.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow( ["commName",
                          "knowledge",
                          "experience",
                          "belief",
                          "opinion"])

        for commName, d in return_dict.items():
            writer.writerow((commName,d['knowledge'], d['experience'], d['belief'], d['opinion'] ))
    
    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('descriptive 15 Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
