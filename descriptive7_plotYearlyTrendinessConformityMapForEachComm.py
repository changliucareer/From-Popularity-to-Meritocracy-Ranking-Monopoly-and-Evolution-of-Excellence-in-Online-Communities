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
from scipy.interpolate import make_interp_spline

def plotMap(commName, commDir, TCYearly, year2MedianTCS):
    os.chdir(commDir)
    print(os.getcwd())

    # get total vot count of each year
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')
    
    with open(intermediate_directory+'/'+'yearlyVoteCountAtEachRank.dict', 'rb') as inputFile:
        year2voteCountAtEachRank_total = pickle.load( inputFile)

    year2totalVoteCount = defaultdict()
    for year, rank2voteCount in year2voteCountAtEachRank_total.items():
        totalVoteCount = 0
        for r, vc in rank2voteCount.items():
            totalVoteCount += vc['pos']+vc['neg']
        year2totalVoteCount[year] = totalVoteCount

    year2totalVoteCount = dict(sorted(year2totalVoteCount.items()))

    TCYearly = dict(sorted(TCYearly.items()))

    colorT = 'seagreen'
    colorC = 'slateblue'
    try:
        Years = list(TCYearly.keys())
        sizes = []
        T = []
        C = []
        voteCount = []
        medianT = []
        medianC = []
        medianSize = []
        for y, tcDict in TCYearly.items():
            T.append(tcDict['Trendiness'])
            C.append(tcDict['Conformity'])
            sizes.append(tcDict['size'])
            voteCount.append(year2totalVoteCount[y])
            medianT.append(year2MedianTCS[y]['MedianT'])
            medianC.append(year2MedianTCS[y]['MedianC'])
            medianSize.append(year2MedianTCS[y]['MedianS'])

        norm_sizes =  [np.log2(s/min(sizes) +1)**2*100 for s in sizes]  
        # norm_sizes =  [float(i)/50 for i in sizes] 
        # norm_sizes = sizes 

        # remove Nones
        removed = []
        for i,t in enumerate(T):
            if (t is None) or (C[i] is None):
                removed.append(i)

        T = [t for i,t in enumerate(T) if i not in removed]
        C = [c for i,c in enumerate(C) if i not in removed]
        voteCount = [c for i,c in enumerate(voteCount) if i not in removed]
        norm_sizes = [s for i,s in enumerate(norm_sizes) if i not in removed]
        Years = [y for i,y in enumerate(Years) if i not in removed]
        medianT = [t for i,t in enumerate(medianT) if i not in removed]
        medianC = [c for i,c in enumerate(medianC) if i not in removed] 

        maxSize = max(norm_sizes)
        minSize = min(norm_sizes)

        if len(T)==0 or len(C)==0: # empty
            return
        
        

        maxYearOfMedian = max(year2MedianTCS.keys())
        finalMedianT = year2MedianTCS[maxYearOfMedian]['MedianT']
        finalMedianC = year2MedianTCS[maxYearOfMedian]['MedianC']

        
        # curve plot
        plt.cla()
        
        # # Adding Xticks
        # plt.xlabel('Year', fontsize = 15)
        # plt.ylabel('Normalized Measurement', fontsize = 15)

        #Creating adjacent subplots
        fig, ax1 = plt.subplots(figsize=(12, 9))
        ax2 = ax1.twinx()

        # ax1.set_title(f"{commName.replace('.stackexchange','')}",fontsize=20)

        ax1.grid(color='lightgrey', linestyle='--', linewidth=0.2, alpha = 0.5)


        X = []
        previousYearVoteCount = 0
        for year, vc in year2totalVoteCount.items():
            X.append(previousYearVoteCount+vc)
            previousYearVoteCount = X[-1]

        ax1.set_xticks(X)
        
        if commName in ['cstheory.stackexchange']:
            xticklabels = [str(y) for y in Years]  # show all years
        elif commName in ['math.meta.stackexchange','mathoverflow.net','philosophy.stackexchange']: # only omit the second year
            xticklabels = [str(y) for y in Years]
            xticklabels[1] = ""
        elif commName in ['politics.stackexchange']:
            # when omit some early years before 2016, but show the first year
            xticklabels = [str(y) if y>=2016 else "" for y in Years]
            xticklabels[0] = str(Years[0])
        else:
            # when omit some early years before 2014, but show the first year
            xticklabels = [str(y) if y>=2014 else "" for y in Years]
            xticklabels[0] = str(Years[0])  
        ax1.set_xticklabels(xticklabels, rotation=90, fontdict={'horizontalalignment': 'center', 'size':15})
        ax1.set_xlabel("Year", fontsize=25)
        ax1.set_ylabel("Trendiness", color=colorT, fontsize=25)
        ax1.tick_params(axis="y", labelcolor=colorT, labelsize=15)

        ax2.set_ylabel("Conformity", color=colorC, fontsize=25)
        ax2.tick_params(axis="y", labelcolor=colorC, labelsize=15)

        X_Y_Spline_0 = make_interp_spline(X, T)
        # Returns evenly spaced numbers
        # over a specified interval.
        X_ = np.linspace(min(X), max(X), 500)
        Y_ = X_Y_Spline_0(X_)
        ax1.plot(X_, Y_, linestyle='-', color=colorT, label='Trendiness')
        

        X_Y_Spline_1 = make_interp_spline(X, C)
        # Returns evenly spaced numbers
        # over a specified interval.
        X_ = np.linspace(min(X), max(X), 500)
        Y_ = X_Y_Spline_1(X_)
        ax2.plot(X_, Y_, linestyle='-', color=colorC, label='Conformity')
        # ax2.legend(loc="lower right",fontsize=20, frameon=False)
        
        ax1.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)

        # axis limits for 2D map
        # xlim (T): 0.23385321007486987 to 2.1721463115301227
        # ylim (C): 3.812792484764052 to 60

        ax1.set_ylim(0.2, 3.7)
        ax2.set_ylim(3.8, 53)

        xmiddle = (min(X) + max(X))/2
        ax1.hlines(y=finalMedianT, xmin= min(X), xmax=0.75*xmiddle, color=colorT, lw=0.8, ls='--')
        ax1.text(x=0.75*xmiddle, y=finalMedianT, s="median T", horizontalalignment='left', 
                 verticalalignment='center', fontdict={'color':colorT, 'size':15})
        ax2.hlines(y=finalMedianC, xmin= 1.25*xmiddle, xmax=max(X),color=colorC, lw=0.8, ls='--')
        ax2.text(x=1.25*xmiddle, y=finalMedianC, s="median C", horizontalalignment='right', 
                 verticalalignment='center', fontdict={'color':colorC, 'size':15})
        
        picDir = f'yearlyMeasurementsChanging.pdf'
        savePlot(plt, picDir)
        print(f"save {picDir} for {commName}")


        # plot scatter plot 
        # scatter plot
        plt.figure(figsize=(14,10.5))
        Fontsize=25
        plt.scatter(T, C, s=norm_sizes, c='g',alpha=0.2, cmap='viridis_r')
        plt.scatter(T, C,s=50,c='r',alpha=1, cmap='viridis_r')
        plt.xlabel('Trendiness',fontsize = Fontsize)
        plt.ylabel('Conformity',fontsize = Fontsize)
        plt.xticks(fontsize= Fontsize)
        plt.yticks(fontsize= Fontsize)

        # plot quadrant by Median from top communities for the last year
        plt.axvline(x=finalMedianT,color='k', lw=0.8, ls='--')
        plt.axhline(y=finalMedianC,color='k', lw=0.8, ls='--')
        plt.text(x=finalMedianT+0.01, y=finalMedianC-0.1, s="MEDIAN", horizontalalignment='left', 
                 verticalalignment='top', fontdict={'color':colorT, 'size':14})

        from adjustText import adjust_text
        texts = [plt.text(T[i],C[i],Years[i],fontsize=Fontsize) for i in range(len(T))]
        adjust_text(texts)

        # manually adjust for mathoverflow
        if commName == 'mathoverflow.net':
            for t in texts:
                if t.get_text() == '2021':
                    t._y += 1

        text_Positions = [[t._x, t._y] for t in texts]
        for i in range(len(T)):
            point_T = T[i]
            point_C = C[i]

            if math.dist([point_T,point_C],text_Positions[i]) >= 0.2: # adding arrow if distance of point and text is too far
                plt.annotate("",
                            xy=(point_T, point_C), xycoords='data',
                            xytext=text_Positions[i], textcoords='data',
                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

        my_ymin = min(min(C),min([t[1] for t in text_Positions])) - 1
        my_ymax = max(max(C),max([t[1] for t in text_Positions])) + 5
        
        if finalMedianC < my_ymin:
            my_ymin = finalMedianC -1
        elif finalMedianC > my_ymax:
            my_ymax = finalMedianC + 1

        plt.ylim(ymin=my_ymin,ymax=my_ymax)
        plt.xlim(xmin= min(min(T),finalMedianT)-0.1,xmax=max(max(T),finalMedianT)+0.1)
        
        # go back to comm dir
        os.chdir(commDir)
        savePlot(plt, f"descriptive_plotTrendinessConformityYearly.pdf")
        print(f"saved descriptive_plotTrendinessConformityYearly.pdf for {commName}")
        
    except Exception as e:
        print(f"failed to plot for {commName}, {e}. ")
   
def main():

    t0=time.time()
    root_dir = os.getcwd()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("sorted CommDir loaded.")

    # load Trendiness fitting results of all comm
    with open('descriptive_TrendinessFittingResults_allComm.dict', 'rb') as inputFile:
        return_trendiness_normalDict = pickle.load( inputFile)
    print(f"return_trendiness_normalDict loaded. length {len(return_trendiness_normalDict)}")

    ## load all non-splitted communities' step by step trained for conformity results dictionary
    with open('descriptive_ConformityAndSize_allComm.dict', 'rb') as inputFile:
        return_conformity_dict = pickle.load( inputFile)
    print("return train success for Conformity dict loaded.")
    
    ## load SOF conformity results dictionary
    with open('descriptive_ConformityAndSize_SOF.dict', 'rb') as inputFile:
        return_conformity_dict_forSOF = pickle.load( inputFile)
        print("return train success for Conformity forSOF loaded.")
    return_conformity_dict['stackoverflow'] = return_conformity_dict_forSOF['stackoverflow']

    print(f"length of conformity dict {len(return_conformity_dict)}")

    # load top 120 commName list
    with open('topSelectedCommNames_descriptive.dict', 'rb') as inputFile:
        topSelectedCommNames = pickle.load( inputFile)

    minYear = 100000
    maxYear =0
    commName2TCYearly = defaultdict()
    for commName, content in return_conformity_dict.items():
        if commName not in topSelectedCommNames: # skip non-selected comms
            continue
        TCYearly = defaultdict()
        if len(content) == 0: # without Conformity for any year, skip
            continue
        for year, cs in content.items():
            c = cs['conformtiy']
            s = cs['size']
            if len(return_trendiness_normalDict[commName])==0: # no T for this comm, skip
                continue
            if year not in return_trendiness_normalDict[commName].keys(): # no T for this year, skip
                continue
            t = return_trendiness_normalDict[commName][year]['spl_estParams'][0]
            TCYearly[year] = {'Trendiness':t, 'Conformity':c, 'size':s}

            if year < minYear:
                minYear = year
            
            if year > maxYear:
                maxYear = year
        
        commName2TCYearly[commName] = TCYearly

    # find the Median of each year
    year2MedianTCS = defaultdict()
    for year in range(minYear, maxYear+1):
        Ts = [d[year]['Trendiness'] for cn, d in commName2TCYearly.items() if year in d.keys()]
        Cs = [d[year]['Conformity'] for cn, d in commName2TCYearly.items() if year in d.keys()]
        Ss = [d[year]['size'] for cn, d in commName2TCYearly.items() if year in d.keys()]

        if len(Ts) >0:
            year2MedianTCS[year]={'MedianT':median(Ts), 'MedianC':median(Cs), 'MedianS':median(Ss)}

    
    selected_comms = ['mathoverflow.net','stackoverflow','codegolf.meta.stackexchange','politics.stackexchange', 
                      'cstheory.stackexchange','askubuntu', 'math.meta.stackexchange', 'philosophy.stackexchange']
    
    # plot parellely
    commNames = [tup[0] for tup in commDir_sizes_sortedlist]
    commDirs = [tup[1] for tup in commDir_sizes_sortedlist]
    n_proc = mp.cpu_count()-2 # left 2 cores to do others
    with mp.Pool(processes=n_proc) as pool:
        args = []
        for commName, TCYearly in commName2TCYearly.items():
            if commName not in selected_comms: # skip non-selected comms
                continue
            # find commDir
            if commName =='MEAN':
                commDir = root_dir
            # elif commName == 'stackoverflow': #skip
            #     continue
            else:
                commIndex = commNames.index(commName)
                commDir = commDirs[commIndex]
            args.append((commName, commDir, TCYearly, year2MedianTCS))
        # issue tasks to the process pool and wait for tasks to complete
        pool.starmap(plotMap, args , chunksize=10)
    """
    plotMap(commDir_sizes_sortedlist[359][0], commDir_sizes_sortedlist[359][1], commName2TCYearly['stackoverflow'])
    """
    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('plot trendiness and conformity Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
