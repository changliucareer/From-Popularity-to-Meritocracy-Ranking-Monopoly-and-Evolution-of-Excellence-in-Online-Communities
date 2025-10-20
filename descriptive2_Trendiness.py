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


def myAction (commName, commDir, year, rank2voteCount,logFileName):
    logtext = f'for year {year},\n'
    # fill up rank2voteCount with ranks without vote
    # sort
    rank2voteCount = dict(sorted(rank2voteCount.items()))
    Longest = max(list(rank2voteCount.keys()))

    ranks_pos = []
    ranks_neg = []
    for r in range(1,Longest+1):
        if r in rank2voteCount.keys():
            ranks_pos.append(rank2voteCount[r]['pos'])
            ranks_neg.append(rank2voteCount[r]['neg'])
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

    x = np.array([i+1 for i in range(len(ranks))],dtype=np.float64)[:endFit]
    y = np.array([r/voteSum for r in ranks],dtype=np.float64)[:endFit]

    # PowerLaw using MLE 
    # let’s start with some random coefficient guesses and optimize
    initParams = np.array([1,1,1,1])
    pl_results = minimize(MLERegressionOfPowerLaw, initParams, method = 'Nelder-Mead', 
    options={'disp': True})
    pl_estParams = pl_results.x
    pl_yhat = powerlawFuncForMLE(pl_estParams[0],pl_estParams[1],pl_estParams[2])
    pl_residuals = abs(y-pl_yhat)
    pl_avgRes = np.sum(pl_residuals)/len(x)

    logtext += f"PowerLaw use only first {endFit} ranks to fit, a={pl_estParams[0]}, b={pl_estParams[1]}, c={pl_estParams[2]}. avg Residual={pl_avgRes}\n"
    
  
    # simplified PowerLaw using MLE 
    # let’s start with some random coefficient guesses and optimize
    initParams = np.array([1,1,1])
    spl_results = minimize(MLERegressionOfSimplifiedPowerLaw, initParams, method = 'Nelder-Mead', 
    options={'disp': True})
    spl_estParams = spl_results.x
    spl_yhat = simplifiedpowerlawFuncForMLE(spl_estParams[0],spl_estParams[1])
    spl_residuals = abs(y-spl_yhat)
    spl_avgRes = np.sum(spl_residuals)/len(x)

    logtext += f"simplified PowerLaw use only first {endFit} ranks to fit, b={spl_estParams[0]}, c={spl_estParams[1]}\n"

    # Exponential using MLE 
    # let’s start with some random coefficient guesses and optimize
    initParams = np.array([1,1,1,1])
    ep_results = minimize(MLERegressionOfExponential, initParams, method = 'Nelder-Mead', 
    options={'disp': True})
    ep_estParams = ep_results.x
    ep_yhat = funcForMLE(ep_estParams[0],ep_estParams[1],ep_estParams[2])
    ep_residuals = abs(y-ep_yhat)
    ep_avgRes = np.sum(ep_residuals)/len(x)

    logtext += f"Exponential use only first {endFit} ranks to fit, a={ep_estParams[0]}, b={ep_estParams[1]}, c={ep_estParams[2]}. avg Residual={ep_avgRes}\n"
    writeIntoLog(logtext, commDir, logFileName)
    
    """
    # curve fitting 
    # for power law
    pl_popt, pl_pcov = curve_fit(powerlawFunc, x, y)
    pl_cf_yhat = powerlawFunc(x, *pl_popt)
    pl_cf_avgRes = np.sum(abs(y - pl_cf_yhat))/len(x)
    # for exponential
    ep_popt, ep_pcov = curve_fit(func, x, y)
    ep_cf_yhat = func(x, *ep_popt)
    ep_cf_avgRes = np.sum(abs(y - ep_cf_yhat))/len(x)
    """
    
    # Plotting
    Fontsize=40
    fig1, axes1 = plt.subplots(ncols=1, nrows=1)
    plt.grid(color='grey', linestyle='--', linewidth=0.2, alpha = 0.5)
    fig1.set_size_inches(20, 10)
    axes1.set_xlabel('rank',fontsize=Fontsize)
    axes1.set_ylabel('Vote Proportion',fontsize=Fontsize)
    # axes1.set_title(f"{commName} {year}",fontsize=Fontsize)
    # only plot the beginning
    x=[i+1 for i in range(len(ranks))][:endRank]
    # y=y[:endRank]
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
    width = 0.8
    bottom = np.zeros(len(x))

    for boolean, weight_count in weight_counts.items():
        p = axes1.bar(x, weight_count, width, label=boolean, bottom=bottom, color=mycolors[boolean],alpha = myalphas[boolean])
        bottom += weight_count
    
    axes1.set_xticks(x)
    
    # # add numbers on each bar
    # for rect in rects:
    #     height = rect.get_height()
    #     axes1.text(rect.get_x() + rect.get_width()/2., 1.05*height,
    #             '%5.3f' % height,\
    #             ha='center', va='bottom')
    # axes1.set_ylim(0, 1)

    axes1.set_ylim(0, 0.5)

    # axes1.plot(x, powerlawFuncForMLE(pl_estParams[0],pl_estParams[1],pl_estParams[2]), 'r-',
    #      label='fit with %d ranks:\nPowerLaw func MLE:\na=%5.5f, b=%5.5f, c=%5.5f\naverage residual=%5.5f' % (endFit,pl_estParams[0],pl_estParams[1],pl_estParams[2],pl_avgRes) )
    # axes1.plot(x, simplifiedpowerlawFuncForMLE(spl_estParams[0],spl_estParams[1]), 'o-',
    #     label='fit with %d ranks:\n Simplified PowerLaw MLE:  b=%5.3f,c=%5.3f\n avgRes=%5.5f' % (endFit,spl_estParams[0],spl_estParams[1],spl_avgRes) )
    x =np.arange(x[0], x[-1]+1, 0.2)
    axes1.plot(x, simplifiedpowerlawFuncForMLE(spl_estParams[0],spl_estParams[1]), '-', color = 'black')
    
    # axes1.plot(x, funcForMLE(ep_estParams[0],ep_estParams[1],ep_estParams[2]), 'g-',
    #      label='Exponential func MLE:\na=%5.5f, b=%5.5f, c=%5.5f\naverage residual=%5.5f' % (ep_estParams[0],ep_estParams[1],ep_estParams[2],ep_avgRes) )

    """
    axes1.plot(x, pl_cf_yhat[:endRank], 'r--',
         label='powelaw curve fit:\n a=%5.3f, b=%5.3f, c=%5.3f,\n avgRes=%5.5f' % (pl_popt[0],pl_popt[1],pl_popt[2],pl_cf_avgRes) )
    axes1.plot(x, ep_cf_yhat[:endRank], 'g--',
         label='exponential curve fit:\n a=%5.3f, b=%5.3f, c=%5.3f,\n avgRes=%5.5f' % (ep_popt[0],ep_popt[1],ep_popt[2],ep_cf_avgRes) )
    """
    
    axes1.legend(fontsize=20, frameon=False)
    plt.setp(axes1.get_xticklabels(),fontsize=20)
    plt.setp(axes1.get_yticklabels(),fontsize=20)

    axes1.spines['top'].set_visible(False)
    axes1.spines['right'].set_visible(False)
    

    label='b=%5.3f' % (spl_estParams[1])
    
    # save plot
    if commName =='MEAN':
        savePlot(fig1, f'MEAN_voteProportionAtEachRank_SimplifiedPowerLaw_Till{year}.pdf')
    else:
        savePlot(fig1, f'voteProportionAtEachRank_SimplifiedPowerLaw_Till{year}.pdf')

    # return pl_estParams, pl_avgRes, ep_estParams, ep_avgRes
    return year, pl_estParams, pl_avgRes, spl_estParams, spl_avgRes, ep_estParams, ep_avgRes
    

def myFun(commName, commDir, return_trendiness_dict, root_dir):
    print(f"comm {commName} running on {mp.current_process().name}")
    logFileName = 'descriptive2_Trendiness_log.txt'

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data files
    current_directory = os.getcwd()
    intermediate_directory = os.path.join(current_directory, r'intermediate_data_folder')
    
    with open(intermediate_directory+'/'+'yearlyVoteCountAtEachRank.dict', 'rb') as inputFile:
        year2voteCountAtEachRank_total = pickle.load( inputFile)

    # process Questions chunk by chunk
    all_outputs = []
    n_proc = mp.cpu_count()-2 # left 2 cores to do others
    with mp.Pool(processes=n_proc) as pool:
        args = []
        for year, rank2voteCount in year2voteCountAtEachRank_total.items():
            args.append((commName,commDir, year,rank2voteCount,logFileName))
        # issue tasks to the process pool and wait for tasks to complete
        results = pool.starmap(myAction, args , chunksize=n_proc)
        # process pool is closed automatically
        for res in results:
            if res != None:
                if isinstance(res, str):
                    print(res)
                else:
                    all_outputs.append(res)
            else:
                print(f"None")

    results.clear()
    
    year2fittingResults= defaultdict()
    for tup in all_outputs:
        year, pl_estParams, pl_avgRes, spl_estParams, spl_avgRes, ep_estParams, ep_avgRes = tup
        year2fittingResults[year] = {'pl_estParams':pl_estParams,
                                     'pl_avgRes':pl_avgRes,
                                     'spl_estParams':spl_estParams,
                                     'spl_avgRes':spl_avgRes,
                                     'ep_estParams':ep_estParams,
                                     'ep_avgRes':ep_avgRes}
    
    return_trendiness_dict[commName] = year2fittingResults
                
    return_trendiness_normalDict = defaultdict()
    for commName, d in return_trendiness_dict.items():
        return_trendiness_normalDict[commName] = d
    
    os.chdir(root_dir) # go back to root directory
    with open('descriptive_TrendinessFittingResults_allComm.dict', 'wb') as outputFile:
        pickle.dump(return_trendiness_normalDict, outputFile)
        print(f"saved return_trendiness_normalDict, {len(return_trendiness_normalDict)} comms.")


def plot4inOne(commDir_sizes_sortedlist, selected_comms):
    commName2rank2voteCount = defaultdict()
    
    for commName in selected_comms: # keep the order of the selected comms
        for tup in commDir_sizes_sortedlist:
            if tup[0] == commName:
                commDir = tup[1]
                break

        # load intermediate_data files
        intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')

        with open(intermediate_directory+'/'+'yearlyVoteCountAtEachRank.dict', 'rb') as inputFile:
            year2voteCountAtEachRank_total = pickle.load( inputFile)
        
        commName2rank2voteCount[commName] = year2voteCountAtEachRank_total[2022]
    
    # plot
    fig, axs = plt.subplots(2, 2)

    for i in range(4):
        commName, rank2voteCount = list(commName2rank2voteCount.items())[i]
        if i == 0:
            cur_axs = axs[0,0]
        elif i == 1:
            cur_axs = axs[0,1]
        elif i == 2:
            cur_axs = axs[1,0]
        else:
            cur_axs = axs[1,1]

        # prepare data
        # fill up rank2voteCount with ranks without vote
        # sort
        rank2voteCount = dict(sorted(rank2voteCount.items()))
        Longest = max(list(rank2voteCount.keys()))

        ranks = []
        for r in range(1,Longest+1):
            if r in rank2voteCount.keys():
                ranks.append(rank2voteCount[r])
            else:
                ranks.append(0)


        endRank = 15
        # endRank = int(0.8 * Longest)
        voteSum = sum(ranks)
        if voteSum==0: # no vote, skip
            return None

        #use partial data to fit
        #endFit = 50
        endFit = getEndFitRank (ranks) # dynamically fitting

        global x,y

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
        
        cur_axs.set_xlabel('rank',fontsize=10)
        cur_axs.set_ylabel('Vote Proportion',fontsize=10)
 
        # only plot the beginning
        x=x[:endRank]
        y=y[:endRank]
        rects = cur_axs.bar(x, y, color ="blue", alpha=0.5)
        cur_axs.set_xticks(x)

        cur_axs.set_ylim(0, 0.5)

        cur_axs.plot(x, simplifiedpowerlawFuncForMLE(spl_estParams[0],spl_estParams[1]), 'b-',
            label='b=%5.4f' % (spl_estParams[0]) )

        cur_axs.legend(fontsize=15,loc="best")
        plt.setp(cur_axs.get_xticklabels(),fontsize=10)
        plt.setp(cur_axs.get_yticklabels(),fontsize=10)
        cur_axs.set_title(f"{commName.replace('.stackexchange','')}", fontsize = 15)
    
    plt.subplots_adjust(wspace=0.3,hspace=0.3)
    fig.set_figheight(10)
    fig.set_figwidth(10)

    picDir = f'voteProportionAtEachRank_4CommsInOneFigure_SimplifiedPowerLaw_Till2022.pdf'
    savePlot(fig, picDir)
    print(f"4 in one figure saved {picDir} for {selected_comms}")




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

    # use shared variable to communicate among all comm's process
    manager = mp.Manager()
    return_trendiness_dict = manager.dict() # to save the trendiness fitting results of each community

    # # test on comm "coffee.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[166][0], commDir_sizes_sortedlist[166][1], return_trendiness_dict, root_dir)
    # # test on comm "datascience.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[301][0], commDir_sizes_sortedlist[301][1], return_trendiness_dict, root_dir)
    # test on comm "webapps.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[305][0], commDir_sizes_sortedlist[305][1], return_trendiness_dict, root_dir)
    # test on comm "writers" to debug
    # myFun(commDir_sizes_sortedlist[286][0], commDir_sizes_sortedlist[286][1], return_trendiness_dict, root_dir)
    # test on comm "mathoverflow" to debug
    # myFun(commDir_sizes_sortedlist[343][0], commDir_sizes_sortedlist[343][1], return_trendiness_dict, root_dir)
    # test on comm "unix.meta" to debug
    # myFun(commDir_sizes_sortedlist[173][0], commDir_sizes_sortedlist[173][1], return_trendiness_dict, root_dir)

    # skip splitted communities
    splitted_comms = ['tex.stackexchange','meta.stackexchange','softwareengineering.stackexchange','superuser','math.stackexchange','worldbuilding.stackexchange','codegolf.stackexchange','stackoverflow']

    selected_comms = ['mathoverflow.net','stackoverflow','unix.meta.stackexchange','politics.stackexchange']
    
    # # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for tup in commDir_sizes_sortedlist:
        commName = tup[0]
        commDir = tup[1]

        try:
            p = mp.Process(target=myFun, args=(commName,commDir, return_trendiness_dict, root_dir))
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
    

    # plot four communities in one figure
    # plot4inOne(commDir_sizes_sortedlist, selected_comms)

    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('descriptive 2 trendiness fitting Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
