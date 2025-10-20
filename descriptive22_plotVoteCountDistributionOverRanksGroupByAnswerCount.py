import os
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
import matplotlib.pyplot as plt
from matplotlib import cm
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
from scipy import stats
from scipy.optimize import minimize
from scipy.optimize import curve_fit

def func(x, a, b, c):
    return a * np.exp(- (b * x)) + c

def funcForMLE(a, b, c):
    return a * np.exp(- (b * x)) + c

def powerlawFunc(x, a, b, c):
    return a/(x**b+1)+c

def powerlawFuncForMLE(a,b ,c):
    return a/(x**b+1)+c

def simplifiedpowerlawFuncForMLE(b,c):
    return 1/(x**b+1) + c

def getEndFitRank (ranks):
    for i in range(len(ranks)):
        if i==0:
            continue
        else:
            if ranks[i]>ranks[i-1]:
                return i
    return len(ranks)

def MLERegressionOfPowerLaw (params):
    yhat = powerlawFuncForMLE(params[0],params[1],params[2])

    # compute PDF of observed values normally distributed around mean(yhat)
    # with a standard deviation of sd
    negLL = -np.sum(stats.norm.logpdf(y,loc=yhat,scale=params[3]))
    return negLL

def MLERegressionOfSimplifiedPowerLaw (params):
    yhat = simplifiedpowerlawFuncForMLE(params[0],params[1])

    # compute PDF of observed values normally distributed around mean(yhat)
    # with a standard deviation of sd
    negLL = -np.sum(stats.norm.logpdf(y,loc=yhat,scale=params[2]))
    return negLL

def MLERegressionOfExponential (params):
    yhat = funcForMLE(params[0],params[1],params[2])

    # compute PDF of observed values normally distributed around mean(yhat)
    # with a standard deviation of sd
    negLL = -np.sum(stats.norm.logpdf(y,loc=yhat,scale=params[3]))
    return negLL


def myPlot (commName, commDir, answerCount2rank2voteCount,logFileName):
    # Plotting
    Fontsize=40
    fig1, axes1 = plt.subplots(ncols=1, nrows=1)
    plt.grid(color='grey', linestyle='--', linewidth=0.2, alpha = 0.5)
    fig1.set_size_inches(20, 10)
    axes1.set_xlabel('rank',fontsize=Fontsize)
    axes1.set_ylabel('Vote Proportion',fontsize=Fontsize)
    axes1.set_title(f"{commName.replace('.stackexchange','')}",fontsize=Fontsize)

    answerCountList = list(answerCount2rank2voteCount.keys())
    if len(answerCountList) <20:
        n_colors = len(answerCountList) -1 # don't plot for answerCount = 1
    else:
        n_colors = 19 # at most plot answerCount=2 to answerCount =20 (19 lines) 
    colours = cm.rainbow(np.linspace(0, 1, n_colors))
    myColorDict = dict([(answerCountList[i+1],colours[i]) for i in range(n_colors)])

    rank2voteCount_sofar = None
    maxRank_sofar = None
    maxRank_total = max(answerCount2rank2voteCount.keys())

    for answerCount, rank2voteCount in answerCount2rank2voteCount.items():
        cur_maxRank = max(rank2voteCount.keys())

        if rank2voteCount_sofar == None:
            rank2voteCount_sofar = rank2voteCount
            maxRank_sofar = cur_maxRank
        else: # need to combine
            for r, vc in rank2voteCount.items():
                if r in rank2voteCount_sofar.keys():
                    rank2voteCount_sofar[r]['pos'] += rank2voteCount[r]['pos']
                    rank2voteCount_sofar[r]['neg'] += rank2voteCount[r]['neg']
                else:
                    rank2voteCount_sofar[r] = rank2voteCount[r]
            if cur_maxRank > maxRank_sofar:
                maxRank_sofar = cur_maxRank

        ranksWithVote_sofar = len(rank2voteCount_sofar)

        # fill up rank2voteCount with ranks without vote
        # sort
        rank2voteCount_sofar = dict(sorted(rank2voteCount_sofar.items()))
        Longest = max(list(rank2voteCount_sofar.keys()))

        ranks_pos = []
        ranks_neg = []
        for r in range(1,Longest+1):
            if r in rank2voteCount_sofar.keys():
                ranks_pos.append(rank2voteCount_sofar[r]['pos'])
                ranks_neg.append(rank2voteCount_sofar[r]['neg'])
            else:
                ranks_pos.append(0)
                ranks_neg.append(0)

        ranks = [pvc+ ranks_neg[i] for i,pvc in enumerate(ranks_pos)]

        endRank = 15
        # endRank = int(0.8 * Longest)
        voteSum = sum(ranks)
        if voteSum==0: # no vote, skip
            return None

        #use partial data to fit
        #endFit = 50
        endFit = getEndFitRank (ranks) # dynamically fitting
        

        global x,y
        """
        x = np.array([i+1 for i in range(len(ranks))],dtype=np.float64)[:endFit]
        y = np.array([r/voteSum for r in ranks],dtype=np.float64)[:endFit]
        
        # simplified PowerLaw using MLE 
        # let’s start with some random coefficient guesses and optimize
        initParams = np.array([1,1,1])
        spl_results = minimize(MLERegressionOfSimplifiedPowerLaw, initParams, method = 'Nelder-Mead', 
        options={'disp': True})
        spl_estParams = spl_results.x
        spl_yhat = simplifiedpowerlawFuncForMLE(spl_estParams[0],spl_estParams[1])
        spl_residuals = abs(y-spl_yhat)
        spl_avgRes = np.sum(spl_residuals)/len(x)

        print(f"simplified PowerLaw use only first {endFit} ranks to fit, b={spl_estParams[0]}, c={spl_estParams[1]}\n")
        """
        # only plot the beginning
        x=[i+1 for i in range(len(ranks))][:endRank]
        y=[r/voteSum for r in ranks][:endRank]
        y_pos = np.array([r/voteSum for r in ranks_pos],dtype=np.float64)[:endRank]
        y_neg = np.array([r/voteSum for r in ranks_neg],dtype=np.float64)[:endRank]

        # rects = axes1.bar(x, y)
        weight_counts = {
            "Negative Vote": np.array(y_neg),
            "Positive Vote": np.array(y_pos),
        }
        mycolors = {
            "Negative Vote": 'skyblue',
            "Positive Vote": 'firebrick',
        }
        myalphas = {
            "Negative Vote": 1,
            "Positive Vote": 0.5,
        }
        width = 0.2
        bottom = np.zeros(len(x))

        # for boolean, weight_count in weight_counts.items():
        #     p = axes1.bar(x, weight_count, width, label=boolean, bottom=bottom, color=mycolors[boolean],alpha = myalphas[boolean])
        #     bottom += weight_count
        
        if answerCount == 1: #don't plot
            continue

        if answerCount >20 and answerCount< maxRank_total: # not show
            continue

        if answerCount == maxRank_total: # the last one, as whole
            axes1.plot(x, y,'o-' ,color='black', label=f'Exist {answerCount} answers (all), {ranksWithVote_sofar} ranks have votes, the Max Rank that has votes:{cur_maxRank}')
        else:
            axes1.plot(x, y,'o-' ,color=myColorDict[answerCount], label=f'Exist {answerCount} answers, {ranksWithVote_sofar} ranks have votes, the Max Rank that has votes:{cur_maxRank}')
        
        axes1.set_xticks(x)
    
        axes1.set_ylim(0, 0.75)

        x =np.arange(x[0], x[-1]+0.1, 0.2)
        # axes1.plot(x, simplifiedpowerlawFuncForMLE(spl_estParams[0],spl_estParams[1]), '-', color = myColorDict[answerCount], label=f'Exist {answerCount} answers, {ranksWithVote} ranks have votes, the Max Rank that has votes:{maxRank}')
        
        # axes1.plot(x, funcForMLE(ep_estParams[0],ep_estParams[1],ep_estParams[2]), 'g-',
        #      label='Exponential func MLE:\na=%5.5f, b=%5.5f, c=%5.5f\naverage residual=%5.5f' % (ep_estParams[0],ep_estParams[1],ep_estParams[2],ep_avgRes) )

        """
        axes1.plot(x, pl_cf_yhat[:endRank], 'r--',
            label='powelaw curve fit:\n a=%5.3f, b=%5.3f, c=%5.3f,\n avgRes=%5.5f' % (pl_popt[0],pl_popt[1],pl_popt[2],pl_cf_avgRes) )
        axes1.plot(x, ep_cf_yhat[:endRank], 'g--',
            label='exponential curve fit:\n a=%5.3f, b=%5.3f, c=%5.3f,\n avgRes=%5.5f' % (ep_popt[0],ep_popt[1],ep_popt[2],ep_cf_avgRes) )
        """
        
        axes1.legend(fontsize=10, frameon=False)
        plt.setp(axes1.get_xticklabels(),fontsize=20)
        plt.setp(axes1.get_yticklabels(),fontsize=20)

        axes1.spines['top'].set_visible(False)
        axes1.spines['right'].set_visible(False)

    # save plot
    savePlot(fig1, f'voteProportionAtEachRank_SimplifiedPowerLawGroupByAnswerCount.pdf')
    return 
    

def myFun(commName, commDir, root_dir):
    print(f"comm {commName} running on {mp.current_process().name}")
    logFileName = 'descriptive22_log.txt'

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data files
    current_directory = os.getcwd()
    intermediate_directory = os.path.join(current_directory, r'intermediate_data_folder')
    
    with open(intermediate_directory+'/'+'answerCount2VoteCountAtEachRank.dict', 'rb') as inputFile:
        answerCount2rank2voteCount = pickle.load( inputFile)

    myPlot (commName, commDir, answerCount2rank2voteCount,logFileName)


def main():

    t0=time.time()
    root_dir = os.getcwd()

    
    # # check whether already done
    # with open('descriptive_TrendinessFittingResults_allComm.dict', 'rb') as inputFile:
    #     return_trendiness_normalDict = pickle.load( inputFile)
    # print("return_trendiness_normalDict loaded.")

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("other sorted CommDir loaded.")

    # # test on comm "coffee.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[166][0], commDir_sizes_sortedlist[166][1], root_dir)
    # # test on comm "datascience.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[301][0], commDir_sizes_sortedlist[301][1], root_dir)
    # test on comm "webapps.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[305][0], commDir_sizes_sortedlist[305][1], root_dir)
    # test on comm "writers" to debug
    # myFun(commDir_sizes_sortedlist[286][0], commDir_sizes_sortedlist[286][1], root_dir)
    # test on comm "mathoverflow" to debug
    # myFun(commDir_sizes_sortedlist[343][0], commDir_sizes_sortedlist[343][1], root_dir)
    # test on comm "unix.meta" to debug
    # myFun(commDir_sizes_sortedlist[173][0], commDir_sizes_sortedlist[173][1], root_dir)

    # skip splitted communities
    splitted_comms = ['tex.stackexchange','meta.stackexchange','softwareengineering.stackexchange','superuser','math.stackexchange','worldbuilding.stackexchange','codegolf.stackexchange','stackoverflow']

    selected_comms = ['mathoverflow.net','stackoverflow','unix.meta.stackexchange','politics.stackexchange']
    
    # # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for tup in commDir_sizes_sortedlist:
        commName = tup[0]
        commDir = tup[1]

        # if commName not in selected_comms:
        #     continue

        try:
            p = mp.Process(target=myFun, args=(commName,commDir, root_dir))
            p.start()
        except Exception as e:
            print(e)
            pscount = sum(1 for proc in psutil.process_iter() if proc.name() == 'python3')
            print(f"current python3 processes count {pscount}.")
            return

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
    
    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('descriptive 22 plot vote count distribution over ranks grouped by answerCount Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
