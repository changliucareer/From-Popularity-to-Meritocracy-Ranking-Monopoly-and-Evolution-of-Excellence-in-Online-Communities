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


def plotVoteCountRatioAtEachVoteDifference(commName,commDir,voteDiff2AgreeingVoteRatio,logFileName):
    # sort dict by vd
    sorted_perVoteDifferenceDict = {k: v for k,v in sorted(voteDiff2AgreeingVoteRatio.items(), key=lambda item: item[0])} 
    # plot bars
    print("start to extract y_pos and y_neg ratios...")
    
    x = list (range( min(sorted_perVoteDifferenceDict.keys()), max(sorted_perVoteDifferenceDict.keys())+1 ))
    y = [0]*len(x) # ratio

    for i,vd in enumerate(x):
        if vd in sorted_perVoteDifferenceDict.keys():
            y[i] = sorted_perVoteDifferenceDict[vd]

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
        start_diff = 49
        startIndex = 18


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
    
    if len(cur_x_window)== window_size: # the remaining also reach the window size
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

    
    
    # visualize comment count at each rank
    plt.cla()
    
    # Adding Xticks
    plt.xlabel('vote difference', fontsize = 15)
    plt.ylabel('Agreeing Vote Ratio', fontsize = 15)
    # plt.xticks([r for r in x if r%10==0], [str(i) for i in x if i%10==0])

    
    ##### to fit left side ###################################
    #  # remove the zero values of y
    removedIndexList = []
    for i in range(len(y_window_left)):
        if (y_window_left[i]<=0) or (y_window_left[i] < y_window_left[i+1]):
            removedIndexList.append(i)
        else:
            break

    x_window_left= [j for i, j in enumerate(x_window_left) if i not in removedIndexList]
    y_window_left= [j for i, j in enumerate(y_window_left) if i not in removedIndexList]

    x_window= [j for i, j in enumerate(x_window) if i not in removedIndexList]
    y_window= [j for i, j in enumerate(y_window) if i not in removedIndexList]
    colors_window= [j for i, j in enumerate(colors_window) if i not in removedIndexList]

    # Make the bar plot
    plt.bar(x_window, y_window, color =colors_window, alpha=0.5)

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

    # # # curve fitting for exponential
    # ep_popt_left, ep_pcov = curve_fit(func, x, y, maxfev = 1000)
    # ep_cf_yhat = func(x, *ep_popt_left)
    # ep_cf_avgRes_left = np.sum(abs(y - ep_cf_yhat))/len(x)

    # print(f"Left side fit, a={ep_popt_left[0]}, b={ep_popt_left[1]}, c={ep_popt_left[2]}. avg Residual={ep_cf_avgRes_left}\n")
    # plt.plot(x_window_left, func(x, *ep_popt_left), 'b-',
    # label='a=%5.5f, b=%5.5f, c=%5.5f' % (ep_popt_left[0],ep_popt_left[1],ep_popt_left[2]) )
    
    ##### to fit right side ###################################
    #  # remove the zero values of y
    # removedIndexList = [i for i,y_item in enumerate(y_window_right) if y_item==0]
    # x_window_right= [j for i, j in enumerate(x_window_right) if i not in removedIndexList]
    # y_window_right= [j for i, j in enumerate(y_window_right) if i not in removedIndexList]

    # in order to fit MLE exponential, x, y must to be global
    x = np.array(x_window_right,dtype=np.float64)
    y = np.array(y_window_right,dtype=np.float64)
    
    # # Exponential using MLE 
    # # let’s start with some random coefficient guesses and optimize
    # initParams = np.array([1,1,1,1])
    # ep_results = minimize(MLERegressionOfExponential, initParams, method = 'Nelder-Mead', options={'disp': True})
    # ep_estParams_right = ep_results.x
    # ep_yhat = funcForMLE(ep_estParams_right[0],ep_estParams_right[1],ep_estParams_right[2])
    # ep_residuals = abs(y -ep_yhat)
    # ep_avgRes_right = np.sum(ep_residuals)/len(x)

    # print(f"Right side fit, a={ep_estParams_right[0]}, b={ep_estParams_right[1]}, c={ep_estParams_right[2]}. avg Residual={ep_avgRes_right}\n")

    # plt.plot(x_window_right, funcForMLE(ep_estParams_right[0],ep_estParams_right[1],ep_estParams_right[2]), 'r-',
    # label='a=%5.5f, b=%5.5f, c=%5.5f' % (ep_estParams_right[0],ep_estParams_right[1],ep_estParams_right[2]) )

     # # curve fitting for exponential
    ep_popt_right, ep_pcov = curve_fit(func, x, y)
    ep_cf_yhat = func(x, *ep_popt_right)
    ep_cf_avgRes_right = np.sum(abs(y - ep_cf_yhat))/len(x)

    print(f"Right side fit, a={ep_popt_right[0]}, b={ep_popt_right[1]}, c={ep_popt_right[2]}. avg Residual={ep_cf_avgRes_right}\n")


    plt.plot(x_window_right, func(x, *ep_popt_right), 'r-',
    label='a=%5.5f, b=%5.5f, c=%5.5f' % (ep_popt_right[0],ep_popt_right[1],ep_popt_right[2]) )
    
    plt.legend(loc="lower center")
    plt.title(f"{commName} till2022 ConditionOnRank window{window_size} stride{stride}")


    # plt.ylim(ymin=min(y),ymax=max(y)+0.02)
    # plt.xlim(-64,xmax=100)
    
    picDir = f'VoteRatioAtEachVoteDifference_Till2022_ConditionOnRank_window{window_size}stride{stride}.pdf'
    savePlot(plt, picDir)
    print(f"save {picDir} for {commName}")


def plotVoteCountRatioAtEachVoteDifferencePercentile(commName,commDir,voteDiff2AgreeingVoteRatio,logFileName):
    # sort dict by vd
    sorted_perVoteDifferenceDict = {k: v for k,v in sorted(voteDiff2AgreeingVoteRatio.items(), key=lambda item: item[0])} 
    if len(sorted_perVoteDifferenceDict)==0:
        print(f"no sorted_perVoteDifferenceDict for {commName}")
        return
    # plot bars
    print("start to extract y_pos and y_neg ratios...")
    
    X = list (range( min(sorted_perVoteDifferenceDict.keys()), max(sorted_perVoteDifferenceDict.keys())+1 ))
    Y = [0]*len(X) # ratio

    for i,vd in enumerate(X):
        if vd in sorted_perVoteDifferenceDict.keys():
            Y[i] = sorted_perVoteDifferenceDict[vd]
    
    #  # remove the zero values of y from the most left
    removedIndexList = []
    for i in range(len(Y)):
        if (Y[i]<=0) or (Y[i] < Y[i+1]):
            removedIndexList.append(i)
        else:
            break

    X= [j for i, j in enumerate(X) if i not in removedIndexList]
    Y= [j for i, j in enumerate(Y) if i not in removedIndexList]

    
    zeroVDIndex = X.index(0)

    X_left = [X[i] for i in range(zeroVDIndex)]
    Y_left = [Y[i] for i in range(zeroVDIndex)]

    X_right = [X[i] for i in range(zeroVDIndex+1, len(X))]
    Y_right = [Y[i] for i in range(zeroVDIndex+1, len(Y))]

    if (len(X_left)> 40) and (len(X_right) > len(X_left) *3): # cut the long tail of right side if its longer than 3 times of left side
        X_right = X_right[:len(X_left) *3]
        Y_right = Y_right[:len(Y_left) *3]


    if commName == 'unix.meta.stackexchange': # manually cut
        X_right = X_right[:209]
        Y_right = Y_right[:209]
    
    
    # get 10,20,30,40,50,60,70,80,90,100 percentile of x
    Ps = [10,20,30,40,50,60,70,80,90,100]

    XP_left = [np.percentile(X_left, p) for p in Ps]
    XP_right = [np.percentile(X_right, p) for p in Ps]


    groupedX_left = []
    groupedY_left = []
    
    groupedX_right = []
    groupedY_right = []
    
    # left side
    startIndex = 0
    for i, XP in enumerate(XP_left):
        if math.floor(XP) not in X_left: # no X for current percentile, skip
            continue
        
        curIndex = X_left.index(math.floor(XP))
        if startIndex > curIndex: # copy the previous one
            groupedX_left.append(groupedX_left[-1])
            groupedY_left.append(groupedY_left[-1])

        else: # got average
            curXs = X_left[startIndex:curIndex+1]
            curYs = Y_left[startIndex:curIndex+1]

            groupedX_left.append(mean(curXs))
            groupedY_left.append(mean(curYs))

        startIndex = curIndex +1

    # right side
    startIndex = 0
    for i, XP in enumerate(XP_right):
        if math.floor(XP) not in X_right: # no X for current percentile, skip
            continue

        curIndex = X_right.index(math.floor(XP))
        if startIndex >= curIndex: # copy the previous one
            groupedX_right.append(groupedX_right[-1])
            groupedY_right.append(groupedY_right[-1])

        else: # got average
            curXs = X_right[startIndex:curIndex+1]
            curYs = Y_right[startIndex:curIndex+1]

            groupedX_right.append(mean(curXs))
            groupedY_right.append(mean(curYs))
        
        startIndex = curIndex +1

    
    
    # visualize comment count at each rank
    plt.cla()
    plt.rcParams["font.family"] = "Times New Roman"
    plt.grid(color='grey', linestyle='--', linewidth=0.2, alpha = 0.5)
    
    # Adding Xticks
    plt.xlabel('vote difference percentile', fontsize = 15)
    plt.ylabel('Agreeing Vote Ratio', fontsize = 15)
    # plt.xticks([r for r in x if r%10==0], [str(i) for i in x if i%10==0])

    
    ##### to fit left side ###################################

    # Make the bar plot
    groupedXPosition_left = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1]
    # plt.bar(groupedXPosition_left, groupedY_left, color ='blue', alpha=0.5)

    # make a Lollipop Chart
    plt.vlines(x=groupedXPosition_left, ymin=0, ymax=groupedY_left, color='skyblue', alpha=1, linewidth=1)
    plt.scatter(x=groupedXPosition_left, y=groupedY_left, s=30, color='skyblue', alpha=1)


    # in order to fit MLE exponential, x, y must to be global
    global x,y
    x = np.array(groupedXPosition_left,dtype=np.float64)
    y = np.array(groupedY_left,dtype=np.float64)

    if commName == 'math.meta.stackexchange': # ignore the left most point for regression
        x = x[1:]
        y = y[1:]
    
    # Exponential using MLE 
    # let’s start with some random coefficient guesses and optimize
    initParams = np.array([1,1,1,1])
    ep_results = minimize(MLERegressionOfExponential, initParams, method = 'Nelder-Mead', options={'disp': True})
    ep_estParams_left = ep_results.x
    ep_yhat = funcForMLE(ep_estParams_left[0],ep_estParams_left[1],ep_estParams_left[2])
    ep_residuals = abs(y -ep_yhat)
    ep_avgRes_left = np.sum(ep_residuals)/len(x)
    print(f"Left side fit, a={ep_estParams_left[0]}, b={ep_estParams_left[1]}, c={ep_estParams_left[2]}. avg Residual={ep_avgRes_left}\n")

    # plt.plot(x, funcForMLE(ep_estParams_left[0],ep_estParams_left[1],ep_estParams_left[2]), 'b-',
    # label='a=%5.5f, b=%5.5f, c=%5.5f' % (ep_estParams_left[0],ep_estParams_left[1],ep_estParams_left[2]) )
    
    x = np.array(groupedXPosition_left,dtype=np.float64) # recover the x
    ep_yhat = funcForMLE(ep_estParams_left[0],ep_estParams_left[1],ep_estParams_left[2]) # recover the yhat
    plt.fill_between(x, ep_yhat, 0, where= ep_yhat >= 0, facecolor='skyblue', interpolate=True, alpha=0.5)

    # # # curve fitting for exponential
    # ep_popt_left, ep_pcov = curve_fit(func, x, y, maxfev = 1000)
    # ep_cf_yhat = func(x, *ep_popt_left)
    # ep_cf_avgRes_left = np.sum(abs(y - ep_cf_yhat))/len(x)

    # print(f"Left side fit, a={ep_popt_left[0]}, b={ep_popt_left[1]}, c={ep_popt_left[2]}. avg Residual={ep_cf_avgRes_left}\n")
    # plt.plot(x_window_left, func(x, *ep_popt_left), 'b-',
    # label='a=%5.5f, b=%5.5f, c=%5.5f' % (ep_popt_left[0],ep_popt_left[1],ep_popt_left[2]) )
    
    ##### add zero Vote Diff Bar ##############################
    # Make the bar plot
    # plt.bar([0], Y[zeroVDIndex], color ='grey', alpha=0.5)
    plt.vlines(x=[0], ymin=0, ymax=Y[zeroVDIndex], color='grey', alpha=0.7, linewidth=2)
    plt.scatter(x=[0], y=Y[zeroVDIndex], s=50, color='grey', alpha=0.7)
    
    ##### to fit right side ###################################
    groupedXPosition_right = [1,2,3,4,5,6,7,8,9,10]
    # Make the bar plot
    # plt.bar(groupedXPosition_right, groupedY_right, color ='red', alpha=0.5)
    # make a Lollipop Chart
    plt.vlines(x=groupedXPosition_right, ymin=0, ymax=groupedY_right, color='firebrick', alpha=1, linewidth=1)
    plt.scatter(x=groupedXPosition_right, y=groupedY_right, s=30, color='firebrick', alpha=1)

    # in order to fit MLE exponential, x, y must to be global
    x = np.array([0]+ groupedXPosition_right,dtype=np.float64)
    y = np.array([Y[zeroVDIndex]] +groupedY_right,dtype=np.float64)

    if commName == 'math.meta.stackexchange': # ignore the 2 right most point for regression
        x = x[:-2]
        y = y[:-2]
    
    # Exponential using MLE 
    # let’s start with some random coefficient guesses and optimize
    initParams = np.array([1,1,1,1])
    ep_results = minimize(MLERegressionOfExponential, initParams, method = 'Nelder-Mead', options={'disp': True})
    ep_estParams_right = ep_results.x
    ep_yhat = funcForMLE(ep_estParams_right[0],ep_estParams_right[1],ep_estParams_right[2])
    ep_residuals = abs(y -ep_yhat)
    ep_avgRes_right = np.sum(ep_residuals)/len(x)

    print(f"Right side fit, a={ep_estParams_right[0]}, b={ep_estParams_right[1]}, c={ep_estParams_right[2]}. avg Residual={ep_avgRes_right}\n")

    # plt.plot(x_window_right, funcForMLE(ep_estParams_right[0],ep_estParams_right[1],ep_estParams_right[2]), 'r-',
    # label='a=%5.5f, b=%5.5f, c=%5.5f' % (ep_estParams_right[0],ep_estParams_right[1],ep_estParams_right[2]) )
    plt.fill_between(x, ep_yhat, 0, where= ep_yhat >= 0, facecolor='firebrick', interpolate=True, alpha=0.5)

     # # curve fitting for exponential
    # ep_popt_right, ep_pcov = curve_fit(func, x, y)
    # ep_cf_yhat = func(x, *ep_popt_right)
    # ep_cf_avgRes_right = np.sum(abs(y - ep_cf_yhat))/len(x)

    # print(f"Right side fit, a={ep_popt_right[0]}, b={ep_popt_right[1]}, c={ep_popt_right[2]}. avg Residual={ep_cf_avgRes_right}\n")

    # plt.plot(x, func(x, *ep_popt_right), 'r-',
    # label='a=%5.5f, b=%5.5f, c=%5.5f' % (ep_popt_right[0],ep_popt_right[1],ep_popt_right[2]) )
    # plt.fill_between(x, ep_cf_yhat, 0, where= ep_cf_yhat >= 0, facecolor='firebrick', interpolate=True, alpha=0.5)
    
    # plt.legend(loc="lower center")
    # plt.title(f"{commName}")
    plt.xticks(groupedXPosition_left + [0] + groupedXPosition_right, ['$-$\n100%', '$-$\n90%', '$-$\n80%','$-$\n70%', '$-$\n60%', '$-$\n50%','$-$\n40%', '$-$\n30%', '$-$\n20%', '$-$\n10%', '0','+\n10%', '+\n20%', '+\n30%','+\n40%', '+\n50%', '+\n60%','+\n70%', '+\n80%', '+\n90%', '+\n100%'], fontsize=5)

    plt.ylim(ymin=0)
    # plt.xlim(-64,xmax=100)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    
    picDir = f'VoteRatioAtEachVoteDifferencePercentile_Till2022_ConditionOnRank.pdf'
    savePlot(plt, picDir)
    print(f"save {picDir} for {commName}")

def myFun(commName, commDir):
    print(f"comm {commName} running on {mp.current_process().name}")
    logFileName = 'descriptive20_herdingBiasVerification_log.txt'

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data files
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')

    """
    # load total_valuesForConformityComputation_forEachVote
    try:
        with open(intermediate_directory+'/'+'total_valuesForConformityComputation_forEachVote_original.dict', 'rb') as inputFile:
            total_valuesForConformityComputation_forEachVote = pickle.load( inputFile)
    except:
        logtext = f"fail to load total_valuesForConformityComputation_forEachVote_original.dict for {commName}.\n"
        print(logtext)
        writeIntoLog(logtext, commDir, logFileName)
    
    totalVoteCount = len(total_valuesForConformityComputation_forEachVote)
    
    # update values for conformity computation along with real time voteDiff2voteCounts_comm table
    voteDiff2voteCountsAndRank_comm = defaultdict()

    for i, tup in enumerate(total_valuesForConformityComputation_forEachVote):
        print(f"processing {i+1}th/{totalVoteCount} vote of {commName}...")
        cur_vote = tup[0]
        cur_n_pos = tup[1]
        cur_n_neg = tup[2]
        rank = tup[3]

        cur_vote_diff = cur_n_pos - cur_n_neg
        
        if cur_vote_diff in voteDiff2voteCountsAndRank_comm.keys():
            voteDiff2voteCountsAndRank_comm[cur_vote_diff].append({'vote':cur_vote, 'rank':rank}) 
        else:
            voteDiff2voteCountsAndRank_comm[cur_vote_diff] = [{'vote':cur_vote, 'rank':rank}] 

    
    total_valuesForConformityComputation_forEachVote = [] # clear to save memory
        
    # save voteDiff2voteCountAndRank_comm
    with open(intermediate_directory+'/'+f'voteDiff2voteCountAndRank_comm.dict', 'wb') as outputFile:
        pickle.dump( voteDiff2voteCountsAndRank_comm, outputFile) 
        logtext = f"saved  voteDiff2voteCountsAndRank_comm of {commName}.\n"
        print(logtext)
        writeIntoLog(logtext, commDir, logFileName)
    

    # load voteDiff2voteCountAndRank_comm
    with open(intermediate_directory+'/'+'voteDiff2voteCountAndRank_comm.dict', 'rb') as inputFile:
        voteDiff2voteCountsAndRank_comm = pickle.load( inputFile)


    # compute voteDiff2Rank2PosCountAndNegCount
    voteDiff2Rank2PosCountAndNegCount = defaultdict()

    for voteDiff, dList in voteDiff2voteCountsAndRank_comm.items():
        rank2PosCountAndNegCount = defaultdict()
        for d in dList:
            cur_vote = d['vote']
            cur_rank = d['rank']
            if cur_rank not in rank2PosCountAndNegCount.keys():
                if cur_vote == 1: # pos vote
                    rank2PosCountAndNegCount[cur_rank] = {'pos':1, 'neg':0}
                else: # neg vote
                    rank2PosCountAndNegCount[cur_rank] = {'pos':0, 'neg':1}
            else:
                if cur_vote == 1: # pos vote
                    rank2PosCountAndNegCount[cur_rank]['pos'] += 1
                else: # neg vote
                    rank2PosCountAndNegCount[cur_rank]['neg'] += 1
        
        voteDiff2Rank2PosCountAndNegCount[voteDiff] = rank2PosCountAndNegCount
    
    # save  voteDiff2Rank2PosCountAndNegCount
    with open(intermediate_directory+'/'+f' voteDiff2Rank2PosCountAndNegCount.dict', 'wb') as outputFile:
        pickle.dump( voteDiff2Rank2PosCountAndNegCount, outputFile) 
        logtext = f"saved  voteDiff2Rank2PosCountAndNegCount of {commName}.\n"
        print(logtext)
        writeIntoLog(logtext, commDir, logFileName)

    voteDiff2voteCountsAndRank_comm.clear() # to save memory
    """

    # load voteDiff2Rank2PosCountAndNegCount
    with open(intermediate_directory+'/'+' voteDiff2Rank2PosCountAndNegCount.dict', 'rb') as inputFile:
        voteDiff2Rank2PosCountAndNegCount = pickle.load( inputFile)
    
    # compute voteDiff2AgreeingVoteRatio
    voteDiff2AgreeingVoteRatio = defaultdict()

    for voteDiff, rank2PosCountAndNegCount in voteDiff2Rank2PosCountAndNegCount.items():
        agreeTotal = 0
        total = 0


        totalVoteCountOverAllRanks = 0

        for rank, d in rank2PosCountAndNegCount.items():
            posVoteCountAtCurRank = d['pos']
            negVoteCountAtCurRank = d['neg']
            totalVoteCountAtCurRank = posVoteCountAtCurRank + negVoteCountAtCurRank
            totalVoteCountOverAllRanks += totalVoteCountAtCurRank

            if voteDiff >= 0: # majority is positive
                agreeCountAtCurRank = posVoteCountAtCurRank
            else:
                agreeCountAtCurRank = negVoteCountAtCurRank

            agreeTotal += agreeCountAtCurRank/totalVoteCountAtCurRank
            total += totalVoteCountAtCurRank/totalVoteCountAtCurRank
        
        if (totalVoteCountOverAllRanks > 1) and agreeTotal >0:
            voteDiff2AgreeingVoteRatio[voteDiff] = agreeTotal / total
        else:
            print(f"omit vote Diff {voteDiff}'s agreeing ratio, because only one vote cast at this voteDiff")
    
    # plot
    # plotVoteCountRatioAtEachVoteDifference(commName,commDir,voteDiff2AgreeingVoteRatio,logFileName)
    plotVoteCountRatioAtEachVoteDifferencePercentile(commName,commDir,voteDiff2AgreeingVoteRatio,logFileName)

            

    
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
    # test on comm "webapps.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[305][0], commDir_sizes_sortedlist[305][1])
    # test on comm "stackoverflow" to debug
    # myFun(commDir_sizes_sortedlist[359][0], commDir_sizes_sortedlist[359][1])
    # test on comm "politics" to debug
    # myFun(commDir_sizes_sortedlist[283][0], commDir_sizes_sortedlist[283][1])
    # test on comm "travel" to debug
    # myFun(commDir_sizes_sortedlist[319][0], commDir_sizes_sortedlist[319][1])
    # test on comm "writers" to debug
    # myFun(commDir_sizes_sortedlist[286][0], commDir_sizes_sortedlist[286][1])
    # test on comm "mathoverflow" to debug
    # myFun(commDir_sizes_sortedlist[343][0], commDir_sizes_sortedlist[343][1])
    # test on comm "unix.meta" to debug
    # myFun(commDir_sizes_sortedlist[173][0], commDir_sizes_sortedlist[173][1])
    # test on comm "math.meta" to debug
    myFun(commDir_sizes_sortedlist[250][0], commDir_sizes_sortedlist[250][1])

    # skip splitted communities
    splitted_comms = ['tex.stackexchange','meta.stackexchange','softwareengineering.stackexchange','superuser','math.stackexchange','worldbuilding.stackexchange','codegolf.stackexchange','stackoverflow']

    """
    # # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for tup in commDir_sizes_sortedlist:
        commName = tup[0]
        commDir = tup[1]
        if commName != 'stackoverflow': # skip stackoverflow 
            try:
                p = mp.Process(target=myFun, args=(commName,commDir))
                p.start()
            except Exception as e:
                print(e)
                pscount = sum(1 for proc in psutil.process_iter() if proc.name() == 'python3')
                print(f"current python3 processes count {pscount}.")
                return
        else:
            stackoverflow_dir = commDir
            continue

        processes.append(p)
        if len(processes)==10:
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

    
    # # run stackoverflow at the last separately
    print(f"start to process stackoverflow alone...")
    myFun('stackoverflow', stackoverflow_dir)
    
    """
            
    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('descriptive20 get vote count and rank at each vote diff Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
