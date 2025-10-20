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

def plotMap(year, SampleSize_commName_List, Trendiness_dict,Conformity_dict,root_dir):
    selectCommCount = len(Trendiness_dict)
    topSelectedCommNames = [tup[1] for tup in SampleSize_commName_List[:selectCommCount+1]]

    #only plot top selected communities
    topSelected_trendiness=[]
    topSelected_conformity=[]
    topSelected_sizes = []
    topSelected_cn =[]
    for i, commName in enumerate(topSelectedCommNames):
        cn = commName.replace('.stackexchange','')
        if cn =='meta':
            cn = 'meta.stackexchange'
        topSelected_cn.append(cn)
        topSelected_trendiness.append(Trendiness_dict[commName])
        topSelected_conformity.append(Conformity_dict[commName])
        topSelected_sizes.append(SampleSize_commName_List[i][0])



    minSize = sorted(topSelected_sizes)[0]
    norm_sizes =  [(float(i)/len(topSelected_sizes)) for i in topSelected_sizes]
    print(f"Filtered commu count {len(topSelected_trendiness)}")      

    colors = []
    for i in range(len(topSelected_trendiness)):
        colors.append(np.random.rand(3,))

    plt.figure(figsize=(50,30))

    plt.scatter(topSelected_trendiness, topSelected_conformity, s=norm_sizes, c=colors,alpha=0.5, cmap='viridis')
    plt.xlabel('Trendiness',fontsize=60)
    plt.ylabel('Conformity',fontsize=60)

    # plt.xlim(xmin=1.95,xmax=2.15)
    # plt.ylim(ymin=20,ymax=53)   
    plt.xlim(xmin=min(topSelected_trendiness),xmax=max(topSelected_trendiness)+0.05)
    plt.ylim(ymin=min(topSelected_conformity),ymax=max(topSelected_conformity)+1)

    plt.yticks(fontsize=60)
    plt.xticks(fontsize=60)
    
    texts = [plt.text(topSelected_trendiness[i],topSelected_conformity[i],topSelected_cn[i],fontsize=60) for i in range(len(topSelected_trendiness))]
    adjust_text(texts) 

    # go back to root dir
    os.chdir(root_dir)
    savePlot(plt, f"descriptive_plotSubComms_SimplifiedPowerLaw_b_asTrendiness_tillyear_{year}.pdf")
    print(f"saved descriptive_plotSubComms_SimplifiedPowerLaw_b_asTrendiness_tillyear_{year}.pdf")
   
def main():

    t0=time.time()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("sorted CommDir loaded.")

    # go to StackOverflow data directory
    parentCommName = commDir_sizes_sortedlist[359][0]
    parentCommDir = commDir_sizes_sortedlist[359][1]
    os.chdir(parentCommDir)
    print(os.getcwd())

    subComms_data_folder = os.path.join(parentCommDir, f'subCommunities_folder')
    if not os.path.exists( subComms_data_folder):
        print("Exception: no subComms_data_folder!")

    # load Trendiness fitting and Conformity results of all sub comms
    with open(subComms_data_folder+'/'+f'subCommsTrendinessConformitySize.dict', 'rb') as inputFile:
        return_normalDict = pickle.load( inputFile)
    print(f"return_normalDict loaded. length {len(return_normalDict)}")

    for year in range(2008,2023):
        # extract T and C for current year
        Trendiness_dict = defaultdict()
        Conformity_dict = defaultdict()
        SampleSize_commName_List = []
        for commName, content in return_normalDict.items():
            year2fittingResults = content['TrendinessFittingResults'] 
            year2ConformitySize = content['ConformitySizeResults']
            if len(year2ConformitySize) == 0: # without Conformity for any year, skip
                continue
            if year not in year2ConformitySize.keys(): # current comm doesn't have conformity for current year
                continue
            c = year2ConformitySize[year]['conformtiy']
            if c == None: # conformity is None and size is 0
                continue
            Conformity_dict[commName] = c
            SampleSize_commName_List.append((year2ConformitySize[year]['size'], commName))

            if year in year2fittingResults.keys():
                b = year2fittingResults[year]['spl_estParams'][0]
                Trendiness_dict[commName] = b
            else:
                print(f"{commName} doesn't have trendiness.")
                Trendiness_dict[commName] = None


        SampleSize_commName_List.sort(key=lambda tup: tup[0], reverse=True)

        # plot
        if year == 2022:
            plotMap(year, SampleSize_commName_List, Trendiness_dict,Conformity_dict,subComms_data_folder)
    

    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('plot trendiness and conformity Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
