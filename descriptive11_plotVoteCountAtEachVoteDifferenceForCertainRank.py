import os
#First print the current working directory
print("Current Working Directory", os.getcwd())
###############################################################
from toolFunctions import format_time, writeIntoLog, insertEventInUniversalTimeStep, savePlot
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
from itertools import groupby
import re
import psutil
from scipy.sparse import csr_matrix, lil_matrix
import numpy as np
import sys
import copy
from statistics import mean
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from scipy.optimize import curve_fit

def func(x, a, b, c):
    return a * np.exp(- (b * x)) + c

def funcForMLE(a, b, c):
    return a * np.exp(- (b * x)) + c

def MLERegressionOfExponential (params):
    yhat = funcForMLE(params[0],params[1],params[2])

    # compute PDF of observed values normally distributed around mean(yhat)
    # with a standard deviation of sd
    negLL = -np.sum(stats.norm.logpdf(y,loc=yhat,scale=params[3]))
    return negLL

def plotVoteCountRatioAtEachVoteDifferenceForCertainRank(commName,commDir,voteDiff2voteCounts_comm_atCertainRank,logFileName, certainRank):
    # sort dict by vd
    sorted_perVoteDifferenceDict = {k: v for k,v in sorted(voteDiff2voteCounts_comm_atCertainRank.items(), key=lambda item: item[0])} 
    # plot bars
    print("start to extract y_pos and y_neg ratios...")
    global x,y
    x = list (range( min(sorted_perVoteDifferenceDict.keys()), max(sorted_perVoteDifferenceDict.keys())+1 ))
    y = [0]*len(x) # ratio

    colors  = []
    for i,vd in enumerate(x):
        if vd in sorted_perVoteDifferenceDict.keys():
            vcDict = sorted_perVoteDifferenceDict[vd]
            if vd >= 0: # right side
                y[i] = vcDict['pos']/(vcDict['pos']+vcDict['neg'])
                if y[i] ==1 and vcDict['pos']==1:
                    y[i] = 0
                colors.append('red')
            else: # left side
                y[i] = vcDict['neg']/(vcDict['pos']+vcDict['neg'])
                if y[i] ==1 and vcDict['neg']==1:
                    y[i] = 0
                colors.append('blue')


    # only plot a segment of whole list
    print("start to plot a segment of whole list...")
    start_diff = x[0]
    if start_diff<0:
        start_diff= int(-start_diff)
    else: 
        start_diff = 0

    ### slide-window smoothing
    window_size = 1
    stride = 1
    startIndex = 0
    while (start_diff % stride !=0):
        start_diff -= 1 
        startIndex += 1

    if commName == 'stackoverflow':
        start_diff = 35
        startIndex = 10

    end_index = startIndex + start_diff * 2 +1
    x = x[startIndex:end_index]
    y = y[startIndex:end_index]


    x_window = []
    y_window = []
    x_window_right = []
    y_window_right = []
    x_window_left = []
    y_window_left = []

    colors_window = []
    cur_x_window = []
    cur_y_window = []
    for i,vd in enumerate(x):
        if vd == 0: # zero vd
            # store previous window
            x_window.append(mean(cur_x_window))
            if vd > 0: # right side
                y_window.append(mean(cur_y_window))
                colors_window.append('red')
                # add current point to right side
                x_window_right.append(x_window[-1])
                y_window_right.append(y_window[-1])
            else: # vd <0
                y_window.append(mean(cur_y_window))
                colors_window.append('blue')
                # add current point to left side
                x_window_left.append(x_window[-1])
                y_window_left.append(y_window[-1])
            # clear cur window
            cur_x_window= []
            cur_y_window = []
            # append cur vd
            x_window.append(vd)
            y_window.append(y[i])
            colors_window.append('grey')
            # x_window_right.append(x_window[-1])
            # y_window_right.append(y_window[-1])
            continue

        if len(cur_x_window)<window_size:
            cur_x_window.append(vd)
            cur_y_window.append(y[i])
        else: # reach window size
            x_window.append(mean(cur_x_window))
            if vd > 0: # right side
                y_window.append(mean(cur_y_window))
                colors_window.append('red')
                # add current point to right side
                x_window_right.append(x_window[-1])
                y_window_right.append(y_window[-1])
            else: # vd <0, left side
                y_window.append(mean(cur_y_window))
                colors_window.append('blue')
                # add current point to left side
                x_window_left.append(x_window[-1])
                y_window_left.append(y_window[-1])
            # re-initialized cur window
            cur_x_window= cur_x_window[stride:]
            cur_y_window = cur_y_window[stride:]
            # append cur vd
            cur_x_window.append(vd)
            cur_y_window.append(y[i])
    
    
    # visualize comment count at each rank
    plt.cla()
    
    # Make the plot
    # plt.bar(x, y, color =colors)
    # plt.bar(x_bucket, y_bucket, color =colors_bucket)
    plt.bar(x_window, y_window, color =colors_window, alpha=0.5)
    
    # Adding Xticks
    plt.xlabel('vote difference', fontsize = 15)
    plt.ylabel('Vote Ratio', fontsize = 15)
    # plt.xticks([r for r in x if r%10==0], [str(i) for i in x if i%10==0])

    ##### to fit left side ###################################
    #  # remove the zero values of y
    removedIndexList = []
    for i in range(len(y_window_left)):
        if y_window_left[i]<=0:
            removedIndexList.append(i)
        else:
            break

    x_window_left= [j for i, j in enumerate(x_window_left) if i not in removedIndexList]
    y_window_left= [j for i, j in enumerate(y_window_left) if i not in removedIndexList]

    # in order to fit MLE exponential, x, y must to be global
    x = np.array(x_window_left,dtype=np.float64)
    y = np.array(y_window_left,dtype=np.float64)
    
    # Exponential using MLE 
    # let’s start with some random coefficient guesses and optimize
    initParams = np.array([1,1,1,1])
    ep_results = minimize(MLERegressionOfExponential, initParams, method = 'Nelder-Mead', options={'disp': True})
    ep_estParams_left = ep_results.x
    ep_yhat = funcForMLE(ep_estParams_left[0],ep_estParams_left[1],ep_estParams_left[2])
    ep_residuals = abs(y -ep_yhat)
    ep_avgRes_left = np.sum(ep_residuals)/len(x)

    print(f"Left side fit, a={ep_estParams_left[0]}, b={ep_estParams_left[1]}, c={ep_estParams_left[2]}. avg Residual={ep_avgRes_left}\n")


    plt.plot(x_window_left, funcForMLE(ep_estParams_left[0],ep_estParams_left[1],ep_estParams_left[2]), 'b-',
    label='a=%5.5f, b=%5.5f, c=%5.5f' % (ep_estParams_left[0],ep_estParams_left[1],ep_estParams_left[2]) )

    ##### to fit right side ###################################
    #  # remove the zero values of y
    # removedIndexList = [i for i,y_item in enumerate(y_window_right) if y_item==0]
    # x_window_right= [j for i, j in enumerate(x_window_right) if i not in removedIndexList]
    # y_window_right= [j for i, j in enumerate(y_window_right) if i not in removedIndexList]

    # in order to fit MLE exponential, x, y must to be global
    x = np.array(x_window_right,dtype=np.float64)
    y = np.array(y_window_right,dtype=np.float64)
    
    # Exponential using MLE 
    # let’s start with some random coefficient guesses and optimize
    initParams = np.array([1,1,1,1])
    ep_results = minimize(MLERegressionOfExponential, initParams, method = 'Nelder-Mead', options={'disp': True})
    ep_estParams_right = ep_results.x
    ep_yhat = funcForMLE(ep_estParams_right[0],ep_estParams_right[1],ep_estParams_right[2])
    ep_residuals = abs(y -ep_yhat)
    ep_avgRes_right = np.sum(ep_residuals)/len(x)

    print(f"Right side fit, a={ep_estParams_right[0]}, b={ep_estParams_right[1]}, c={ep_estParams_right[2]}. avg Residual={ep_avgRes_right}\n")

    
    # # curve fitting 
    # # for exponential
    # ep_popt_right, ep_pcov = curve_fit(func, x, y)
    # ep_cf_yhat = func(x, *ep_popt_right)
    # ep_cf_avgRes_right = np.sum(abs(y - ep_cf_yhat))/len(x)

    plt.plot(x_window_right, funcForMLE(ep_estParams_right[0],ep_estParams_right[1],ep_estParams_right[2]), 'r-',
    label='a=%5.5f, b=%5.5f, c=%5.5f' % (ep_estParams_right[0],ep_estParams_right[1],ep_estParams_right[2]) )
    
    plt.legend(loc="lower center")
    plt.title(f"{commName} till2022 (at rank {certainRank}) window{window_size} stride{stride}")


    # plt.ylim(ymin=min(y),ymax=max(y)+0.02)
    # plt.xlim(-64,xmax=100)
    
    picDir = f'VoteRatioAtEachVoteDifference_Till2022_atRank{certainRank}_window{window_size}stride{stride}.pdf'
    savePlot(plt, picDir)
    print(f"save {picDir} for {commName}")


def myFun(commName, commDir):
    print(f"comm {commName} running on {mp.current_process().name}")
    logFileName = 'descriptive11_Log.txt'

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data files
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')

    certainRank = 6
    # load vote diff to voteCounts at certain rank
    with open(intermediate_directory+'/'+f'voteDiff2voteCounts_comm_atCertainRank{certainRank}.dict', 'rb') as inputFile:
        voteDiff2voteCounts_comm_atCertainRank = pickle.load( inputFile)
    
    # plot for averageVoteCountAtEachVoteDiffJumpInterval
    plotVoteCountRatioAtEachVoteDifferenceForCertainRank(commName,commDir,voteDiff2voteCounts_comm_atCertainRank,logFileName, certainRank)
        
        
def plot4inOne(commDir_sizes_sortedlist, selected_comms):
    commName2voteDiff2voteCounts_comm_atCertainRank = defaultdict()
    
    for commName in selected_comms: # keep the order of the selected comms
        for tup in commDir_sizes_sortedlist:
            if tup[0] == commName:
                commDir = tup[1]
                break

        # load intermediate_data files
        intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')

        certainRank = 6
        # load vote diff to voteCounts at certain rank
        with open(intermediate_directory+'/'+f'voteDiff2voteCounts_comm_atCertainRank{certainRank}.dict', 'rb') as inputFile:
            voteDiff2voteCounts_comm_atCertainRank = pickle.load( inputFile)
        
        commName2voteDiff2voteCounts_comm_atCertainRank[commName] = voteDiff2voteCounts_comm_atCertainRank
    
    # plot
    fig, axs = plt.subplots(2, 2)

    for i in range(4):
        commName, voteDiff2voteCounts_comm_atCertainRank = list(commName2voteDiff2voteCounts_comm_atCertainRank.items())[i]
        if i == 0:
            cur_axs = axs[0,0]
        elif i == 1:
            cur_axs = axs[0,1]
        elif i == 2:
            cur_axs = axs[1,0]
        else:
            cur_axs = axs[1,1]

        # prepare data
        # sort dict by vd
        sorted_perVoteDifferenceDict = {k: v for k,v in sorted(voteDiff2voteCounts_comm_atCertainRank.items(), key=lambda item: item[0])} 
        # plot bars
        print("start to extract y_pos and y_neg ratios...")
        global x,y
        x = list (range( min(sorted_perVoteDifferenceDict.keys()), max(sorted_perVoteDifferenceDict.keys())+1 ))
        y = [0]*len(x) # ratio

        colors  = []
        for i,vd in enumerate(x):
            if vd in sorted_perVoteDifferenceDict.keys():
                vcDict = sorted_perVoteDifferenceDict[vd]
                if vd >= 0: # right side
                    y[i] = vcDict['pos']/(vcDict['pos']+vcDict['neg'])
                    if y[i] ==1 and vcDict['pos']==1:
                        y[i] = 0
                    colors.append('red')
                else: # left side
                    y[i] = vcDict['neg']/(vcDict['pos']+vcDict['neg'])
                    if y[i] ==1 and vcDict['neg']==1:
                        y[i] = 0
                    colors.append('blue')


        # only plot a segment of whole list
        print("start to plot a segment of whole list...")
        start_diff = x[0]
        if start_diff<0:
            start_diff= int(-start_diff)
        else: 
            start_diff = 0

        ### slide-window smoothing
        window_size = 1
        stride = 1
        startIndex = 0
        while (start_diff % stride !=0):
            start_diff -= 1 
            startIndex += 1

        if commName == 'stackoverflow':
            start_diff = 35
            startIndex = 10

        end_index = startIndex + start_diff * 2 +1
        x = x[startIndex:end_index]
        y = y[startIndex:end_index]


        x_window = []
        y_window = []
        x_window_right = []
        y_window_right = []
        x_window_left = []
        y_window_left = []

        colors_window = []
        cur_x_window = []
        cur_y_window = []
        for i,vd in enumerate(x):
            if vd == 0: # zero vd
                # store previous window
                x_window.append(mean(cur_x_window))
                if vd > 0: # right side
                    y_window.append(mean(cur_y_window))
                    colors_window.append('red')
                    # add current point to right side
                    x_window_right.append(x_window[-1])
                    y_window_right.append(y_window[-1])
                else: # vd <0
                    y_window.append(mean(cur_y_window))
                    colors_window.append('blue')
                    # add current point to left side
                    x_window_left.append(x_window[-1])
                    y_window_left.append(y_window[-1])
                # clear cur window
                cur_x_window= []
                cur_y_window = []
                # append cur vd
                x_window.append(vd)
                y_window.append(y[i])
                colors_window.append('grey')
                x_window_right.append(x_window[-1])
                y_window_right.append(y_window[-1])
                continue

            if len(cur_x_window)<window_size:
                cur_x_window.append(vd)
                cur_y_window.append(y[i])
            else: # reach window size
                x_window.append(mean(cur_x_window))
                if vd > 0: # right side
                    y_window.append(mean(cur_y_window))
                    colors_window.append('red')
                    # add current point to right side
                    x_window_right.append(x_window[-1])
                    y_window_right.append(y_window[-1])
                else: # vd <0, left side
                    y_window.append(mean(cur_y_window))
                    colors_window.append('blue')
                    # add current point to left side
                    x_window_left.append(x_window[-1])
                    y_window_left.append(y_window[-1])
                # re-initialized cur window
                cur_x_window= cur_x_window[stride:]
                cur_y_window = cur_y_window[stride:]
                # append cur vd
                cur_x_window.append(vd)
                cur_y_window.append(y[i])
        
        # plot cur_axs
        cur_axs.bar(x_window, y_window, color =colors_window, alpha=0.5)
    
        # Adding Xticks
        cur_axs.set_xlabel('vote difference', fontsize = 10)
        cur_axs.set_ylabel('Agreeing Vote Ratio', fontsize = 10)

        ##### to fit left side ###################################
        #  # remove the zero values of y
        removedIndexList = []
        for i in range(len(y_window_left)):
            if y_window_left[i]<=0:
                removedIndexList.append(i)
            else:
                break

        x_window_left= [j for i, j in enumerate(x_window_left) if i not in removedIndexList]
        y_window_left= [j for i, j in enumerate(y_window_left) if i not in removedIndexList]

        # in order to fit MLE exponential, x, y must to be global
        x = np.array(x_window_left,dtype=np.float64)
        y = np.array(y_window_left,dtype=np.float64)
        
        # Exponential using MLE 
        # let’s start with some random coefficient guesses and optimize
        initParams = np.array([1,1,1,1])
        ep_results = minimize(MLERegressionOfExponential, initParams, method = 'Nelder-Mead', options={'disp': True})
        ep_estParams_left = ep_results.x
        ep_yhat = funcForMLE(ep_estParams_left[0],ep_estParams_left[1],ep_estParams_left[2])
        ep_residuals = abs(y -ep_yhat)
        ep_avgRes_left = np.sum(ep_residuals)/len(x)

        print(f"Left side fit for {commName}, a={ep_estParams_left[0]}, b={ep_estParams_left[1]}, c={ep_estParams_left[2]}. avg Residual={ep_avgRes_left}\n")


        cur_axs.plot(x_window_left, funcForMLE(ep_estParams_left[0],ep_estParams_left[1],ep_estParams_left[2]), 'b-')

        ##### to fit right side ###################################
        #  # remove the zero values of y
        # removedIndexList = [i for i,y_item in enumerate(y_window_right) if y_item==0]
        # x_window_right= [j for i, j in enumerate(x_window_right) if i not in removedIndexList]
        # y_window_right= [j for i, j in enumerate(y_window_right) if i not in removedIndexList]

        # in order to fit MLE exponential, x, y must to be global
        x = np.array(x_window_right,dtype=np.float64)
        y = np.array(y_window_right,dtype=np.float64)
        
        # Exponential using MLE 
        # let’s start with some random coefficient guesses and optimize
        initParams = np.array([1,1,1,1])
        ep_results = minimize(MLERegressionOfExponential, initParams, method = 'Nelder-Mead', options={'disp': True})
        ep_estParams_right = ep_results.x
        ep_yhat = funcForMLE(ep_estParams_right[0],ep_estParams_right[1],ep_estParams_right[2])
        ep_residuals = abs(y -ep_yhat)
        ep_avgRes_right = np.sum(ep_residuals)/len(x)

        print(f"Right side fit for {commName}, a={ep_estParams_right[0]}, b={ep_estParams_right[1]}, c={ep_estParams_right[2]}. avg Residual={ep_avgRes_right}\n")


        cur_axs.plot(x_window_right, funcForMLE(ep_estParams_right[0],ep_estParams_right[1],ep_estParams_right[2]), 'r-')
        
        # cur_axs.legend(loc="lower center")
        cur_axs.set_title(f"{commName.replace('.stackexchange','')}", fontsize = 15)
    
    plt.subplots_adjust(wspace=0.3,hspace=0.3)
    fig.set_figheight(10)
    fig.set_figwidth(10)

    picDir = f'VoteRatioAtEachVoteDifference_4CommsInOneFigure_Till2022_atRank{certainRank}_window{window_size}stride{stride}.pdf'
    savePlot(fig, picDir)
    print(f"4 in one figure saved {picDir} for {selected_comms}")






def main():

    t0=time.time()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("other sorted CommDir loaded.")
    
    # # test on comm "coffee.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[166][0], commDir_sizes_sortedlist[166][1])
    # # test on comm "datascience.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[301][0], commDir_sizes_sortedlist[301][1])
    # test on comm "english.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[349][0], commDir_sizes_sortedlist[349][1])
    # test on comm "stackoverflow" to debug
    # myFun(commDir_sizes_sortedlist[359][0], commDir_sizes_sortedlist[359][1])
    
    # skip splitted communities
    splitted_comms = ['tex.stackexchange','meta.stackexchange','softwareengineering.stackexchange','superuser','math.stackexchange','worldbuilding.stackexchange','codegolf.stackexchange','stackoverflow']
    selected_comms = ['mathoverflow.net','stackoverflow','unix.meta.stackexchange','politics.stackexchange']
    """
    # # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for tup in commDir_sizes_sortedlist:
        commName = tup[0]
        commDir = tup[1]
        
        # if commName not in selected_comms: # skip non-selected comms
        #     continue
        
        try:
            p = mp.Process(target=myFun, args=(commName,commDir))
            p.start()
        except Exception as e:
            print(e)
            pscount = sum(1 for proc in psutil.process_iter() if proc.name() == 'python3')
            print(f"current python3 processes count {pscount}.")
            return

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
    """

    # plot four communities in one figure
    plot4inOne(commDir_sizes_sortedlist, selected_comms)
          
    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('descriptive10 Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
