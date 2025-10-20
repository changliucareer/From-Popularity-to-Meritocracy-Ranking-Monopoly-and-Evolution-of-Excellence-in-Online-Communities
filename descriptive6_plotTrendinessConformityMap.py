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
from statistics import mean, median
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
import csv

def plotMap(year, SampleSize_commName_List, Trendiness_dict,Conformity_dict,root_dir):
    ylim_max = 60
    selectCommCount = 120
    topSelectedCommNames = [tup[1] for tup in SampleSize_commName_List[:selectCommCount+1]]

    # MEAN_Trendiness = Trendiness_dict['MEAN'] # 1.5179208251970313
    # MEAN_Conformity = Conformity_dict['MEAN'] # 13.630976471538755
    # print(f"MEAN trendiness:{MEAN_Trendiness}, MEAN conformity:{MEAN_Conformity}")

    # # remove MEAN from list if not the last year
    # if year != 2023:
    #     topSelectedCommNames.remove('MEAN')

    # save topSelectedCommNames
    with open('topSelectedCommNames_descriptive.dict', 'wb') as outputFile:
        pickle.dump(topSelectedCommNames, outputFile)

    #only plot top selected communities
    topSelected_trendiness=[]
    topSelected_conformity=[]
    topSelected_sizes = []
    topSelected_cn =[]
    for i, commName in enumerate(topSelectedCommNames):
        cn = commName.replace('.stackexchange','')
        if cn =='meta':
            cn = 'meta.stackexchange'

        # if ('.meta' in cn) or ('meta.' in cn):
        #     continue # skip meta comms

        if Trendiness_dict[commName]==None or Conformity_dict[commName]== None:
            continue
        
        topSelected_cn.append(cn)
        topSelected_trendiness.append(Trendiness_dict[commName])
        if Conformity_dict[commName] > ylim_max:
            topSelected_conformity.append(ylim_max)
        else:
            topSelected_conformity.append(Conformity_dict[commName])
        topSelected_sizes.append(SampleSize_commName_List[i][0])

    medianTrendiness = median(sorted(topSelected_trendiness))
    medianConformity = median(sorted(topSelected_conformity))

    minSize = sorted(topSelected_sizes[:120])[0]
    # minSize = sorted(topSelected_sizes)[len(topSelected_sizes)-120]
    # norm_sizes =  [(float(i)/minSize)*15 for i in topSelected_sizes]
    norm_sizes =  [np.log2(i/minSize +1)**2*100 for i in topSelected_sizes]
    print(f"Filtered commu count {len(topSelected_trendiness)}")      

    colors = []
    for i in range(len(topSelected_trendiness)):
        colors.append(np.random.rand(3,))

    plt.figure(figsize=(50,30))

    plt.scatter(topSelected_trendiness, topSelected_conformity, s=norm_sizes, c=colors,alpha=0.5, cmap='viridis')
    plt.xlabel('Trendiness',fontsize=40)
    plt.ylabel('Conformity',fontsize=40)

    secondMaxOfTrendiness = sorted(topSelected_trendiness,reverse=True)[1]
    secondMaxOfConformity = sorted(topSelected_conformity,reverse=True)[1]
    plt.xlim(xmin=min(topSelected_trendiness),xmax=secondMaxOfTrendiness+0.1)
    print(f'xlim (T): {min(topSelected_trendiness)} to {secondMaxOfTrendiness+0.1}')
    plt.ylim(ymin=min(topSelected_conformity)-0.1,ymax=ylim_max)
    print(f'ylim (C): {min(topSelected_conformity)-0.1} to {ylim_max}')
    """
    if year !=2022 :
        plt.xlim(xmin=min(topSelected_trendiness),xmax=max(topSelected_trendiness)+0.1)
        plt.ylim(ymin=min(topSelected_conformity)-0.1,ymax=max(topSelected_conformity))
    else:
        plt.xlim(xmin=min(topSelected_trendiness)-0.01,xmax=2.23)
        plt.ylim(ymin=min(topSelected_conformity)-0.1,ymax=3)
    """
    
    # # plot quadrant by MEAN from all communities for the last year
    # plt.axvline(x=MEAN_Trendiness,color='k', lw=0.8, ls='--')
    # plt.axhline(y=MEAN_Conformity,color='k', lw=0.8, ls='--')


    plt.text(medianTrendiness,medianConformity-1,'MEDIAN',fontsize=25, color='green')
    plt.axvline(x=medianTrendiness,color='g', lw=0.8, ls='--')
    plt.axhline(y=medianConformity,color='g', lw=0.8, ls='--')
    print(f"medianTrendiness:{medianTrendiness}, medianConformity:{medianConformity}")

    plt.yticks(fontsize=25)
    plt.xticks(fontsize=25)

    # add quadrant types
    plt.text(x=0.85, y=40, s="Knowledge-type", horizontalalignment='center', verticalalignment='center', fontdict={'color':'red', 'size':25,'weight':'bold'})
    plt.text(x=1.85, y=47, s="Experience-type", horizontalalignment='center', verticalalignment='center', fontdict={'color':'red', 'size':25,'weight':'bold'})
    plt.text(x=1.9, y=13, s="Opinion-type", horizontalalignment='center', verticalalignment='center', fontdict={'color':'red', 'size':25,'weight':'bold'})
    plt.text(x=1, y=10, s="Belief-type", horizontalalignment='center', verticalalignment='center', fontdict={'color':'red', 'size':25,'weight':'bold'})

    # for i in range(len(topSelected_trendiness)):
    #     lab = topSelected_cn[i]
    #     plt.text(topSelected_trendiness[i],topSelected_conformity[i],lab,fontsize='x-small')
    
    texts = [plt.text(topSelected_trendiness[i],topSelected_conformity[i],topSelected_cn[i],fontsize=25) for i in range(len(topSelected_trendiness))]
    adjust_text(texts) 

    # go back to root dir
    os.chdir(root_dir)
    savePlot(plt, f"descriptive_plottop{selectCommCount}_SimplifiedPowerLaw_b_asTrendiness_tillyear_{year}_new.pdf")
    print(f"saved descriptive_plottop{selectCommCount}_SimplifiedPowerLaw_b_asTrendiness_tillyear_{year}.pdf")
    # savePlot(plt, f"descriptive_plotNonMeta_SimplifiedPowerLaw_b_asTrendiness_tillyear_{year}.pdf")
    # print(f"saved descriptive_plotNonMeta_SimplifiedPowerLaw_b_asTrendiness_tillyear_{year}.pdf")

    # # save csv
    with open(root_dir +'/'+'allComm_descriptive6_TrendinessAndConformity.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i,commName in enumerate(topSelected_cn):
            sampleCount = topSelected_sizes[i]
            trendiness = topSelected_trendiness[i]
            conformity = topSelected_conformity[i]
            
            quadrant = None
            if trendiness >= medianTrendiness and conformity >= medianConformity:
                quadrant = 'experience'
            elif trendiness >= medianTrendiness and conformity < medianConformity:
                quadrant = 'opinion'
            elif trendiness < medianTrendiness and conformity >= medianConformity:
                quadrant = 'knowledge'
            else:
                quadrant = 'belief'

            writer.writerow( [commName, sampleCount, trendiness, conformity, quadrant])
   
def main():

    t0=time.time()
    root_dir = os.getcwd()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("sorted CommDir loaded.")

    # # save csv
    with open(root_dir +'/'+'allComm_descriptive6_TrendinessAndConformity.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow( ["commName","totalSampleCount","Trendiness","Conformity", "quadrant"])


    # load Trendiness fitting results of all comm
    with open('descriptive_TrendinessFittingResults_allComm.dict', 'rb') as inputFile:
        return_trendiness_normalDict = pickle.load( inputFile)
    print(f"return_trendiness_normalDict loaded. length {len(return_trendiness_normalDict)}")
    
    # with open('MEAN_samples_10each_descriptive_TrendinessFittingResults.dict', 'rb') as inputFile:
    #     return_trendiness_MEAN_dict = pickle.load( inputFile)
    # print("return train success MEAN dict loaded.")

    # return_trendiness_normalDict['MEAN'] = return_trendiness_MEAN_dict['MEAN']
    
    # print(f"updated return train success dict loaded. length {len(return_trendiness_normalDict)}")

    ## load all non-splitted communities' step by step trained for conformity results dictionary
    with open('descriptive_ConformityAndSize_allComm.dict', 'rb') as inputFile:
        return_conformity_dict = pickle.load( inputFile)
    print("return train success for Conformity dict loaded.")

    # with open('MEAN_samples_10each_descriptive_ConformityAndSize.dict', 'rb') as inputFile:
    #     return_conformity_dict_forMEAN = pickle.load( inputFile)
    #     print("return train success for Conformity forMEAN dict loaded.")
    # return_conformity_dict['MEAN'] = return_conformity_dict_forMEAN['MEAN']
    
    ## load SOF conformity results dictionary
    with open('descriptive_ConformityAndSize_SOF.dict', 'rb') as inputFile:
        return_conformity_dict_forSOF = pickle.load( inputFile)
        print("return train success for Conformity forSOF loaded.")
    return_conformity_dict['stackoverflow'] = return_conformity_dict_forSOF['stackoverflow']

    print(f"length of conformity dict {len(return_conformity_dict)}")

    for year in range(2008,2023):
        # extract T and C for current year
        Conformity_dict = defaultdict()
        SampleSize_commName_List = []
        for commName, content in return_conformity_dict.items():
            if len(content) == 0: # without Conformity for any year, skip
                continue
            if year not in content.keys(): # current comm doesn't have conformity for current year
                continue
            c = content[year]['conformtiy']
            if c == None: # conformity is None and size is 0
                continue
            Conformity_dict[commName] = c
            SampleSize_commName_List.append((content[year]['size'], commName))

        Trendiness_dict = defaultdict()
        for commName in Conformity_dict.keys():
            if commName in return_trendiness_normalDict.keys():
                if year in return_trendiness_normalDict[commName].keys():
                    b = return_trendiness_normalDict[commName][year]['spl_estParams'][0]
                    Trendiness_dict[commName] = b
                
            else:
                print(f"{commName} doesn't have trendiness.")
                Trendiness_dict[commName] = None


        SampleSize_commName_List.sort(key=lambda tup: tup[0], reverse=True)

        # # plot
        if year == 2022:
            plotMap(year, SampleSize_commName_List, Trendiness_dict,Conformity_dict,root_dir)

        if year == 2022:
            print(f"start to save the results till 2022 as csv...")
            with open(f'descriptive6_TrendinessConformityOfAllcomms.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow( ["commName","sampleCount", "Trendiness (position bias sensitivity)","Conformity (herding bias sensitivity)"])

                for tup in SampleSize_commName_List:
                    sampleSize, commName = tup
                    conformity = Conformity_dict[commName]
                    trendiness = Trendiness_dict[commName]
                    writer.writerow((commName,sampleSize, trendiness, conformity))
        
    

    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('plot trendiness and conformity Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
