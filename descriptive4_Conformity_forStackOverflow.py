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
from statistics import mean

def negExpFunc(x, a, b, c):
    return a * np.exp(- (b * x)) + c

def ExpFunc(x, a, b, c):
    return a * np.exp(b * x) + c

def powerlawFunc(x, a, b, c):
    return a/(x**b+1)+c

def simplifiedpowerlawFunc(x, b,c):
    return 1/(x**b+1) + c

def plotVoteProportionAtEachVoteDiffForCertainRank(commName,commDir,year, voteDiff2voteCounts,logFileName, certainRank, certainAnswer = None):
    # change cwd to commDir
    os.chdir(commDir)
    # plot bars
    print("start to extract y_pos and y_neg ratios...")
    x = list (range( min(voteDiff2voteCounts.keys()), max(voteDiff2voteCounts.keys())+1 ))
    y = [0]*len(x) # vote proportion of sign corresponding to current vote difference's sign
    y_pos = [0] * len(x)
    y_neg = [0] * len(x)

    colors  = []
    for i,vd in enumerate(x):
        if vd in voteDiff2voteCounts.keys():
            vcDict = voteDiff2voteCounts[vd]
            if vd >= 0:
                y[i] = (vcDict['pos']-vcDict['neg'])/(vcDict['pos']+vcDict['neg'])
                y_pos[i] = vcDict['pos']
                y_neg[i] = vcDict['neg']
                colors.append('red')
            else:
                y[i] = (vcDict['neg']-vcDict['pos'])/(vcDict['pos']+vcDict['neg'])
                y_pos[i] = vcDict['pos']
                y_neg[i] = vcDict['neg']
                colors.append('blue')

    # only plot a segment of whole list
    print("start to plot a segment of whole list...")
    start_diff = x[0]
    if start_diff<0:
        start_diff= int(-start_diff)
    else: 
        start_diff = 0

    # find index of zero vd
    for i,vd in enumerate(x):
        if vd == 0: # zero vd
            zeroVd_Index = i
            break
    
    """
    ### bucketing smoothing
    bucket_size = 1
    startIndex = 0
    while ((zeroVd_Index-startIndex) % bucket_size !=0):
        startIndex += 1


    end_index = zeroVd_Index + (zeroVd_Index - startIndex) +1
    x = x[startIndex:end_index]
    y = y[startIndex:end_index]
    y_pos = y_pos[startIndex:end_index]
    y_neg = y_neg[startIndex:end_index]


    x_bucket = []
    y_bucket = []
    colors_bucket = []
    cur_x_bucket = []
    cur_y_pos_bucket = []
    cur_y_neg_bucket = []
    for i,vd in enumerate(x):
        if vd == 0: # zero vd
            # store previous bucket
            x_bucket.append(mean(cur_x_bucket))
            if vd > 0:
                try:
                    y_bucket.append(sum(cur_y_pos_bucket)/(sum(cur_y_pos_bucket)+sum(cur_y_neg_bucket)))
                except: # divided by 0
                    y_bucket.append(0) 
                colors_bucket.append('red')
            else: # vd <0
                try:
                    y_bucket.append(sum(cur_y_neg_bucket)/(sum(cur_y_pos_bucket)+sum(cur_y_neg_bucket)))
                except: # divided by 0
                    y_bucket.append(0)
                colors_bucket.append('blue')
            # clear cur bucket
            cur_x_bucket = []
            cur_y_pos_bucket = []
            cur_y_neg_bucket = []
            # append cur vd
            x_bucket.append(vd)
            y_bucket.append(y_pos[i]/(y_pos[i]+y_neg[i]))
            colors_bucket.append('grey')
            continue

        if len(cur_x_bucket)<bucket_size:
            cur_x_bucket.append(vd)
            cur_y_pos_bucket.append(y_pos[i])
            cur_y_neg_bucket.append(y_neg[i])
        else: # reach bucket size
            x_bucket.append(mean(cur_x_bucket))
            if vd > 0:
                try:
                    y_bucket.append(sum(cur_y_pos_bucket)/(sum(cur_y_pos_bucket)+sum(cur_y_neg_bucket)))
                except:
                    y_bucket.append(0)
                colors_bucket.append('red')
            else: # vd <0
                try:
                    y_bucket.append(sum(cur_y_neg_bucket)/(sum(cur_y_pos_bucket)+sum(cur_y_neg_bucket)))
                except:
                    y_bucket.append(0)
                colors_bucket.append('blue')
            # clear cur bucket
            cur_x_bucket = []
            cur_y_pos_bucket = []
            cur_y_neg_bucket = []
            # append cur vd
            cur_x_bucket.append(vd)
            cur_y_pos_bucket.append(y_pos[i])
            cur_y_neg_bucket.append(y_neg[i])
    """     
    
    
    ### slide-window smoothing
    window_size = 5
    stride = 3
    startIndex = 0
    while ((zeroVd_Index-startIndex) % stride !=0):
        startIndex += 1


    end_index = zeroVd_Index + (zeroVd_Index - startIndex)*6
    x = x[startIndex:end_index]
    y = y[startIndex:end_index]
    y_pos = y_pos[startIndex:end_index]
    y_neg = y_neg[startIndex:end_index]


    x_window = []
    y_window = []
    colors_window = []
    cur_x_window = []
    cur_y_window = []
    for i,vd in enumerate(x):
        if vd == 0: # zero vd
            # store previous window
            x_window.append(mean(cur_x_window))
            # previous vd <0
            y_window.append(mean(cur_y_window))
            colors_window.append('blue')
            # clear cur window
            cur_x_window= []
            cur_y_window = []
            # append cur vd
            x_window.append(vd)
            y_window.append(y[i])
            colors_window.append('grey')
            continue

        if len(cur_x_window)<window_size:
            cur_x_window.append(vd)
            cur_y_window.append(y[i])
        else: # reach window size
            x_window.append(mean(cur_x_window))
            if vd > 0:
                try:
                    y_window.append(mean(cur_y_window))
                except: # divide by zero
                    y_window.append(0)
                colors_window.append('red')
            else: # vd <0
                try:
                    y_window.append(mean(cur_y_window))
                except:
                    y_window.append(0)
                colors_window.append('blue')
            # re-initialized cur window
            cur_x_window= cur_x_window[stride:]
            cur_y_window = cur_y_window[stride:]
            # append cur vd
            cur_x_window.append(vd)
            cur_y_window.append(y[i])

    # add on the last window
    if len(cur_x_window)>0:
        x_window.append(mean(cur_x_window))
        if vd > 0:
            try:
                y_window.append(mean(cur_y_window))
            except: # divide by zero
                y_window.append(0)
            colors_window.append('red')
        else: # vd <0
            try:
                y_window.append(mean(cur_y_window))
            except:
                y_window.append(0)
            colors_window.append('blue')
    """
    ### shrinking slide-window smoothing
    window_size = 20
    window_sizes = [window_size] # keep the shrinked window sizes
    startIndex = 0

    end_index = startIndex + start_diff * 2 +1
    x = x[startIndex:end_index]
    y = y[startIndex:end_index]
    y_pos = y_pos[startIndex:end_index]
    y_neg = y_neg[startIndex:end_index]


    x_window = []
    y_window = []
    colors_window = []
    cur_x_window = []
    cur_y_pos_window = []
    cur_y_neg_window = []
    for i,vd in enumerate(x):
        if vd == 0: # zero vd
            # store previous window
            x_window.append(mean(cur_x_window))
            if vd > 0:
                y_window.append(sum(cur_y_pos_window)/(sum(cur_y_pos_window)+sum(cur_y_neg_window)))
                colors_window.append('red')
            else: # vd <0
                y_window.append(sum(cur_y_neg_window)/(sum(cur_y_pos_window)+sum(cur_y_neg_window)))
                colors_window.append('blue')
            # clear cur window
            cur_x_window= []
            cur_y_pos_window = []
            cur_y_neg_window = []
            # append cur vd
            x_window.append(vd)
            y_window.append(y_pos[i]/(y_pos[i]+y_neg[i]))
            colors_window.append('grey')
            continue

        if len(cur_x_window)<window_size:
            cur_x_window.append(vd)
            cur_y_pos_window.append(y_pos[i])
            cur_y_neg_window.append(y_neg[i])
        else: # reach window size
            x_window.append(mean(cur_x_window))
            if vd > 0:
                y_window.append(sum(cur_y_pos_window)/(sum(cur_y_pos_window)+sum(cur_y_neg_window)))
                colors_window.append('red')
            else: # vd <0
                y_window.append(sum(cur_y_neg_window)/(sum(cur_y_pos_window)+sum(cur_y_neg_window)))
                colors_window.append('blue')
            # clear cur window
            cur_x_window= []
            cur_y_pos_window = []
            cur_y_neg_window = []
            # append cur vd
            cur_x_window.append(vd)
            cur_y_pos_window.append(y_pos[i])
            cur_y_neg_window.append(y_neg[i])
            # update window size
            if vd < 0:
                window_size = window_size*0.5
                if window_size < 1:
                    window_size = 1
                window_sizes.append(window_size)
            else: #vd >0
                window_size = window_sizes.pop()
    """
    """
    # curve fitting 
    # seperately fitting neg and pos sides
    negSide_x = []
    negSide_y = []
    posSide_x = []
    posSide_y = []
    # for x,y in zip(x_bucket,y_bucket):
    for x,y in zip(x_window,y_window):
        if x < 0:
            negSide_x.append(x)
            negSide_y.append(y)
        else:
            posSide_x.append(x)
            posSide_y.append(y)

    negSide_x = np.array(negSide_x)
    negSide_y = np.array(negSide_y)
    posSide_x = np.array(posSide_x)
    posSide_y = np.array(posSide_y)
      
    try:
        # for neg side fitting, using neg Exponetial Func
        npl_popt, npl_pcov = curve_fit(negExpFunc, negSide_x,negSide_y, maxfev=10000)
        npl_cf_yhat = negExpFunc(negSide_x, *npl_popt)
        npl_cf_avgRes = np.sum(abs(negSide_y - npl_cf_yhat))/len(negSide_x)

        # for pos side fitting, using Exponetial Func
        pl_popt, pl_pcov = curve_fit(ExpFunc, posSide_x,posSide_y, maxfev=10000)
        pl_cf_yhat = ExpFunc(posSide_x, *pl_popt)
        pl_cf_avgRes = np.sum(abs(posSide_y - pl_cf_yhat))/len(posSide_x)
    except:
        print(f"{commName} fail to fit!")
        return

    """

    # visualize 
    plt.cla()
    fig, ax = plt.subplots()
    
    # Make the plot
    # plt.bar(x, y, color =colors)
    # plt.bar(x_bucket, y_bucket, color =colors_bucket)
    ax.bar(x_window, y_window, color =colors_window)
    

    """
     # Plotting curves
    plt.plot(negSide_x, npl_cf_yhat, 'b--',
         label='neg Exp Func curve fit:\n a=%5.3f, b=%5.3f, c=%5.3f,\n avgRes=%5.5f' % (npl_popt[0],npl_popt[1],npl_popt[2],npl_cf_avgRes) )
    plt.plot(posSide_x, pl_cf_yhat, 'r--',
         label='Exp Func curve fit:\n a=%5.3f, b=%5.3f, c=%5.3f,\n avgRes=%5.5f' % (pl_popt[0],pl_popt[1],pl_popt[2],pl_cf_avgRes) )
    """
    
    # Adding Xticks
    ax.set_xlabel('vote difference', fontsize = 15)
    ax.set_ylabel('Vote Proportion', fontsize = 15)
    # plt.xticks([r for r in x if r%10==0], [str(i) for i in x if i%10==0])
    # plt.legend(loc="upper right")
    ax.set_title(f' till{year} (at rank {certainRank}) window{window_size} stride{stride}')

    # plt.ylim(ymin=min(y),ymax=max(y)+0.02)
    # plt.xlim(-64,xmax=100)

    if certainAnswer == None:
        picFileName = f'voteProportionAtEachVoteDifference_Till{year}_atRank{certainRank}_window{window_size}stride{stride}.pdf'
    else:
        picFileName = f'voteProportionAtEachVoteDifference_Till{year}_atRank{certainRank}_forQuestion{certainAnswer[0]}Answer{certainAnswer[1]}_window{window_size}stride{stride}.pdf'
    savePlot(fig, picFileName)
    print(f"{picFileName} saved for {commName}.")
    return 

def plotVoteProportionAtEachVoteDiffForCertainRankCertainAnswer(commName,commDir,year, valuesTupleAtCertainRank,logFileName, certainRank):
    # convert valuesTupleAtCertainRank to dict, (qid, ai) as key
    answer2values = defaultdict()
    for tup in valuesTupleAtCertainRank:
        if len(tup)==8:
            cur_vote, cur_n_pos, cur_n_neg,rank, t, cur_year, qid, ans_index = tup
        else:
            cur_vote, cur_n_pos, cur_n_neg,rank, t, qid, ans_index = tup
        if (qid,ans_index) not in answer2values.keys():
            answer2values[(qid,ans_index)] = [(cur_vote, cur_n_pos, cur_n_neg)]
        else: 
            answer2values[(qid,ans_index)].append((cur_vote, cur_n_pos, cur_n_neg))
    valuesTupleAtCertainRank.clear()

    # find the answer with the most votes
    maxVotesIndex = np.argmax([len(tup[1]) for tup in answer2values.items()])
    targetAnswerTuple = list(answer2values.items())[maxVotesIndex]
    targetValuesList = targetAnswerTuple[1]

    # create a voteDiff2voteCounts table with targetValuseList
    voteDiff2voteCounts_atCertainRank_forCertainAnswer = defaultdict()
    for values in targetValuesList:
        cur_vote, cur_n_pos, cur_n_neg = values
        cur_vote_diff = cur_n_pos - cur_n_neg

        if cur_vote_diff in voteDiff2voteCounts_atCertainRank_forCertainAnswer.keys():
                if cur_vote == 1: # add on pos
                    voteDiff2voteCounts_atCertainRank_forCertainAnswer[cur_vote_diff]['pos'] = voteDiff2voteCounts_atCertainRank_forCertainAnswer[cur_vote_diff]['pos'] + 1
                else: # add on neg
                    voteDiff2voteCounts_atCertainRank_forCertainAnswer[cur_vote_diff]['neg'] = voteDiff2voteCounts_atCertainRank_forCertainAnswer[cur_vote_diff]['neg'] + 1
        else:
            if cur_vote == 1: # add on pos
                voteDiff2voteCounts_atCertainRank_forCertainAnswer[cur_vote_diff] = {'pos':1, 'neg':0}
            else: # add on neg
                voteDiff2voteCounts_atCertainRank_forCertainAnswer[cur_vote_diff] = {'pos':0, 'neg':1}
    
    # plot for voteDiff2voteCounts_atCertainRank_forCertainAnswer 
    certainAnswer = targetAnswerTuple[0]
    plotVoteProportionAtEachVoteDiffForCertainRank(commName,commDir,year, voteDiff2voteCounts_atCertainRank_forCertainAnswer,logFileName, certainRank, certainAnswer)
        
          
        


def myAction (commName,commDir, year,valuesForConformityComputation,logFileName):
    bugVoteCount = 0
    cur_logSum = 0
    cur_voteCount =0
    for i, tup in enumerate(valuesForConformityComputation):
        print(f"process year {year} {i+1}th/{cur_voteCount} for {commName}")
        try:
            cur_vote, cur_n_pos, cur_n_neg, cur_voteDiff2voteCounts = tup
            cur_voteCount +=1
        except:
            print(f'tup is {tup} in year {year} {commName}.')
            bugVoteCount +=1
            continue
        a = cur_voteDiff2voteCounts['pos']
        b = cur_voteDiff2voteCounts['neg']
        if cur_n_pos >= cur_n_neg:
            cur_logSum += np.log(a+1) - np.log(b+1)
        else:
            cur_logSum += np.log(b+1) - np.log(a+1)
    
    
    return (year, cur_logSum, cur_voteCount, bugVoteCount)

def myFun(commName, commDir, root_dir):
    print(f"comm {commName} running on {mp.current_process().name}")
    logFileName = 'descriptive4_Conformity_forStackOverflow_log.txt'

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data files
    current_directory = os.getcwd()
    intermediate_directory = os.path.join(current_directory, r'intermediate_data_folder')

    # # debug
    # with open('intermediate_data_folder/descriptive4_voteCountsAtEachVoteDiff_till2022.dict', 'rb') as inputFile:
    #     voteDiff2voteCounts = pickle.load( inputFile)
    # plotVoteProportionAtEachVoteDiffForCertainRank(commName,commDir,2022, voteDiff2voteCounts,logFileName, 6)

    splitted_intermediate_data_folder = os.path.join(intermediate_directory, r'splitted_intermediate_data_folder')
    split_valuesForConformtiyComputation_forEachYear_directory = os.path.join(splitted_intermediate_data_folder, r'totalValuesForConformtiyComputation_forEachYear_parts_folder')
    if not os.path.exists(split_valuesForConformtiyComputation_forEachYear_directory): 
        print("Exception: no split_valuesForConformtiyComputation_forEachYear_directory!")

    yearFiles = [ f.path for f in os.scandir(split_valuesForConformtiyComputation_forEachYear_directory) if f.path.endswith('.dict') ]
    # sort files based on year
    yearFiles.sort(key=lambda p: int(p.strip(".dict").split("_")[-1]))
    yearsCount = len(yearFiles)
                                
    print(f"there are {yearsCount} splitted files in {commName}")

    # compute Conformity
    year2Conformity= defaultdict()
    logSum = 0
    totalVoteCount = 0

    bugVoteCount = 0
    
    for subDir in yearFiles:
        
        year = int(subDir.strip(".dict").split("_")[-1])
        
        # get valuesForConformtiyComputation of each year
        with open(subDir, 'rb') as inputFile:
            valuesForConformtiyComputation = pickle.load( inputFile)
            print(f"year {year} of {commName} is loaded.")

        year, cur_logSum, cur_voteCount, bugVoteCount_curYear = myAction(commName, commDir, year, valuesForConformtiyComputation, logFileName)
        
        if bugVoteCount_curYear >0:
            logtext = f"there are {bugVoteCount_curYear} bug votes in year {year}"
            writeIntoLog(logtext, commDir, logFileName)
            print(logtext)
        bugVoteCount += bugVoteCount_curYear
        
        totalVoteCount += cur_voteCount
        logSum += cur_logSum
        if totalVoteCount !=0:
            Conformity = math.exp(logSum/totalVoteCount)
            logtext = f"{commName} year {year} logSum:{logSum}, Conformity: {Conformity}, size: {totalVoteCount}.\n"
            writeIntoLog(logtext, commDir, logFileName)
            print(logtext)
            year2Conformity[year] = {'conformtiy':Conformity,'size':totalVoteCount}
        
        # # plot Conformity verifying 
        # if year == 2022:  
        #     # load voteDiff2voteCounts
        #     certainRank = 6
        #     try:
        #         with open(f'intermediate_data_folder/voteDiff2voteCounts_comm_atCertainRank{certainRank}.dict', 'rb') as inputFile:
        #             voteDiff2voteCounts = pickle.load( inputFile)
        #             print(f"loaded voteDiff2voteCounts till 2022 for {commName}.") 
        #             plotVoteProportionAtEachVoteDiffForCertainRank(commName,commDir,year, voteDiff2voteCounts,logFileName, certainRank)
        #     except Exception as e:
        #         print(e)
        #         writeIntoLog(f"fail to plot:{e}", commDir, logFileName)
        #     """
        #     try:
        #         with open(f'intermediate_data_folder/valuesTupleAtCertainRank{certainRank}.dict', 'rb') as inputFile:
        #             valuesTupleAtCertainRank = pickle.load( inputFile)
        #             print(f"loaded valuesTupleAtCertainRank till 2022 for {commName}.") 
        #             plotVoteProportionAtEachVoteDiffForCertainRankCertainAnswer(commName,commDir,year, valuesTupleAtCertainRank,logFileName, certainRank)
        #     except Exception as e:
        #         print(e)
        #         writeIntoLog(f"fail to plot:{e}", commDir, logFileName)
        #     """  
    return_conformity_normalDict = defaultdict()
    return_conformity_normalDict[commName] = year2Conformity
    
    os.chdir(root_dir) # go back to root directory
    with open('descriptive_ConformityAndSize_SOF.dict', 'wb') as outputFile:
        pickle.dump(return_conformity_normalDict, outputFile)
        print(f"saved return_conformity_normalDict for SOF.")


def main():

    t0=time.time()
    root_dir = os.getcwd()

    
    # # check whether already done
    # with open('descriptive_TrendinessFittingResults_allComm.dict', 'rb') as inputFile:
    #     return_conformity_normalDict = pickle.load( inputFile)
    # print("return_conformity_normalDict loaded.")

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("other sorted CommDir loaded.")

    # test on comm "stackoverflow" to debug
    myFun(commDir_sizes_sortedlist[359][0], commDir_sizes_sortedlist[359][1], root_dir)

    """
    # skip splitted communities
    splitted_comms = ['tex.stackexchange','meta.stackexchange','softwareengineering.stackexchange','superuser','math.stackexchange','worldbuilding.stackexchange','codegolf.stackexchange','stackoverflow']
    
    # # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for tup in commDir_sizes_sortedlist:
        commName = tup[0]
        commDir = tup[1]

        try:
            p = mp.Process(target=myFun, args=(commName,commDir, return_conformity_dict, root_dir))
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

    return_conformity_normalDict = defaultdict()
    for commName, d in return_conformity_dict.items():
        return_conformity_normalDict[commName] = d
    
    """
    
    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('descriptive 4 conformity computing and plotting Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
