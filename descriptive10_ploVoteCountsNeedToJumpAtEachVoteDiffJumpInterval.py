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

def get_vdJumpIntervals (min_vd, max_vd, vd_jump_step_size):
    n_neg_vd_intervals = 0 # number of negative vote difference intervals
    n_pos_vd_intervals = 0 # number of negative vote difference intervals
    if min_vd < 0:
        n_neg_vd_intervals = int(abs(min_vd) / vd_jump_step_size)
    if max_vd > 0:
        n_pos_vd_intervals = int(max_vd / vd_jump_step_size)
    
    neg_vd_interval_thresholds = [i for i in range(-vd_jump_step_size * n_neg_vd_intervals, 0, vd_jump_step_size)]
    pos_vd_interval_thresholds = [i for i in range(0, vd_jump_step_size * n_pos_vd_intervals, vd_jump_step_size)]

    vd_intervals = [(vd+vd_jump_step_size-1, vd) for vd in neg_vd_interval_thresholds] # save as (start_vd, end_vd) tuples
    vd_intervals +=  [(vd, vd+vd_jump_step_size-1) for vd in pos_vd_interval_thresholds]

    vd_intervals_dict = dict(vd_intervals) # key as start vd of each interval, value as end vd of each interval

    return vd_intervals_dict

def plotAverageVoteCountAtEachVoteDiffJumpInterval(commName,commDir,total_vdInterval2voteAccumulatedCount,logFileName, vd_jump_step_size, first_interval, last_invertal, certainRank):

    x = []
    y = []
    xticks = []

    for tup, vcList in total_vdInterval2voteAccumulatedCount.items():
        start_vd = tup[0]
        end_vd = tup[1]
        avg_vc = mean(vcList)
        
        minTick = min(start_vd,end_vd)
        x.append(minTick)
        y.append(avg_vc)
        xticks.append(f"{minTick}\n~\n{max(start_vd,end_vd)}")

    # only plot a segment of whole list
    print("start to plot a segment of whole list...")

    # find index of zero vd
    for i,vd in enumerate(x):
        if vd == 0: # zero vd
            zeroInterval_Index = i
            break
    
    # if zeroInterval_Index > 0:
    #     endIntervalIndex = i * 4
    # else: # without negative interval
    #     endIntervalIndex = i * 3
    endIntervalIndex = 400
    
    # cut to show
    x = x[:endIntervalIndex]
    y = y[:endIntervalIndex]
    xticks = xticks[:endIntervalIndex]

    # reduce the number of xticks for clearer plot
    totalLength = len(x)
    for i, tick in enumerate(xticks):
        if i == 0:
            continue
        elif  i == totalLength-1:
            continue
        elif i <= zeroInterval_Index:
            continue
        else:
            if i%(int(totalLength/10))==0:
                continue
            else:
                xticks[i]=""


     # visualize 
    plt.cla()
    fig, axs = plt.subplots(2, 1)
    fig.tight_layout(pad=3.0)
    
    # Make the plot for negative side
    axs[0].bar(x[:zeroInterval_Index], y[:zeroInterval_Index]) 

    axs[0].set_xlabel('negative vote difference interval', fontsize = 8)
    axs[0].set_ylabel('avg vote count to jump', fontsize = 8)
    for index, tick in enumerate(xticks[:zeroInterval_Index]):
        axs[0].text(x[index],y[index], str(round(y[index],3)), color = 'black',fontsize=8, horizontalalignment='center')
    axs[0].set_xticks(x[:zeroInterval_Index], xticks[:zeroInterval_Index], fontsize = 8)
    axs[0].tick_params(axis='y', labelsize= 8)
    axs[0].set_ylim(ymin=0,ymax=max(y[:zeroInterval_Index])+vd_jump_step_size)
    

    # Make the plot for positive side
    axs[1].bar(x[zeroInterval_Index:], y[zeroInterval_Index:]) 
    axs[1].set_xlabel('positive vote difference interval', fontsize = 8)
    axs[1].set_ylabel('avg vote count to jump', fontsize = 8)
    for index, tick in enumerate(xticks[zeroInterval_Index:]):
        if tick != "":
            axs[1].text(x[zeroInterval_Index+index],y[zeroInterval_Index+index], str(round(y[zeroInterval_Index+index],3)), color = 'black',fontsize=8, horizontalalignment='center')
    axs[1].set_xticks(x[zeroInterval_Index:], xticks[zeroInterval_Index:], fontsize = 8)
    axs[1].tick_params(axis='y', labelsize= 8)
    axs[1].set_ylim(ymin=min(y[zeroInterval_Index:])-0.1,ymax=max(y[zeroInterval_Index:])+0.5)


    # plt.legend(loc="upper right")
    if certainRank != None:
        fig.suptitle(f'vote difference interval size {vd_jump_step_size} for certain Rank {certainRank}')
    else:
        fig.suptitle(f'vote difference interval size {vd_jump_step_size}')

    # plt.ylim(ymin=min(y),ymax=max(y)+0.02)
    # plt.xlim(-64,xmax=100)

    if certainRank != None:
        picFileName = f'average_voteCount_atEachVoteDiffJumpInterval_jumpSize{vd_jump_step_size}_forCertainRank{certainRank}.pdf'
    else:
        picFileName = f'average_voteCount_atEachVoteDiffJumpInterval_jumpSize{vd_jump_step_size}.pdf'
    savePlot(fig, picFileName)
    print(f"{picFileName} saved for {commName}.")
    return


def myFun(commName, commDir):
    print(f"comm {commName} running on {mp.current_process().name}")
    logFileName = 'descriptive10_Log.txt'

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data files
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')

    
    # get total_valuesForConformityComputation_forEachVote
    print(f"loading total_valuesForConformityComputation_forEachVote_original of {commName}...")
    with open(intermediate_directory+'/'+f'total_valuesForConformityComputation_forEachVote_original.dict', 'rb') as inputFile:
        total_valuesForConformityComputation_forEachVote_original = pickle.load( inputFile)

    certainRank = 4
    
    
    # convert valuesTupleAtCertainRank to dict, (qid, ai) as key
    print("start to conver total_valuesForConformityComputation_forEachVote_original to answer2values...")
    totalVoteCount = len(total_valuesForConformityComputation_forEachVote_original)
    answer2values = defaultdict()
    min_vd_comm = 100000  # min vote difference of comm
    max_vd_comm = -100000 # max vote difference of comm
    for i,tup in enumerate(total_valuesForConformityComputation_forEachVote_original):
        print(f"processing {i+1}th/{totalVoteCount} vote of {commName}")
        if len(tup)==8:
            cur_vote, cur_n_pos, cur_n_neg,rank, t, cur_year, qid, ans_index = tup
        else:
            cur_vote, cur_n_pos, cur_n_neg,rank, t, qid, ans_index = tup
        
        if certainRank != None: 
            if rank != certainRank: # skip not certainRank
                continue
        
        cur_vd = cur_n_pos - cur_n_neg
        if cur_vd < min_vd_comm:
            min_vd_comm = cur_vd
        elif cur_vd > max_vd_comm:
            max_vd_comm = cur_vd

        if (qid,ans_index) not in answer2values.keys():
            answer2values[(qid,ans_index)] = [(cur_vote, cur_vd, rank, t)]
        else: 
            answer2values[(qid,ans_index)].append((cur_vote, cur_vd, rank, t))
    total_valuesForConformityComputation_forEachVote_original.clear()

    # save answer2values
    if certainRank != None:
        with open(intermediate_directory+'/'+f'answer2values_forCertainRank{certainRank}.dict', 'wb') as outputFile:
            pickle.dump((answer2values, min_vd_comm, max_vd_comm), outputFile) 
            logtext = f"saved answer2values_forCertainRank{certainRank} of {commName}. length: {len(answer2values)}.  Min vote diff:{min_vd_comm}, Max vote Diff: {max_vd_comm}.\n"
            writeIntoLog(logtext, commDir, logFileName)
            print(logtext)
    else:
        with open(intermediate_directory+'/'+f'answer2values.dict', 'wb') as outputFile:
            pickle.dump((answer2values, min_vd_comm, max_vd_comm), outputFile) 
            logtext = f"saved answer2values of {commName}. length: {len(answer2values)}.  Min vote diff:{min_vd_comm}, Max vote Diff: {max_vd_comm}.\n"
            writeIntoLog(logtext, commDir, logFileName)
            print(logtext)
    
    
    # load intermediate data
    if certainRank != None:
        with open(intermediate_directory+'/'+f'answer2values_forCertainRank{certainRank}.dict', 'rb') as inputFile:
            answer2values, min_vd_comm, max_vd_comm = pickle.load( inputFile)
    else:
        with open(intermediate_directory+'/'+f'answer2values.dict', 'rb') as inputFile:
            answer2values, min_vd_comm, max_vd_comm = pickle.load( inputFile)
    
    
    vd_jump_step_size = 5  # the minimux jump size should be 3, because the first pos interval can only start from 1, if jump size=2, the first pos interval has to be (1,1) which may cause bug

    vd_intervals_dict_comm = get_vdJumpIntervals (min_vd_comm, max_vd_comm, vd_jump_step_size)
    
    
    intervalCount = len(vd_intervals_dict_comm)
    logtext = f"When vd_jump_step_size = {vd_jump_step_size}, there are {intervalCount} intervals of vote difference for {commName}.\n"
    writeIntoLog(logtext, commDir, logFileName)
    print(logtext)

    
    total_vdInterval2voteAccumulatedCount = defaultdict() # for all answers of comm
    # initialize
    for start_vd, end_vd in vd_intervals_dict_comm.items():
        total_vdInterval2voteAccumulatedCount[(start_vd,end_vd)] = []
    
    print(f"start to process answer2values...")
    totalAnswerCount = len(answer2values.keys())
    for i,tup in enumerate(answer2values.items()):
        print(f"processing {i+1}th/{totalAnswerCount} answer of {commName}...")
        qidAiTup, values = tup
        # get min and max vd for cur answer
        min_vd = 100000  # min vote difference of cur answer
        max_vd = -100000 # max vote difference of cur answer
        
        vdList = [t[1] for t in values]
        for i, vd in enumerate(vdList):
            # if i > 0: # sanity check
            #     if not ((vd == vdList[i-1] -1) or (vd == vdList[i-1] +1)):
            #         print("bug")
            if vd < min_vd:
                min_vd = vd
            elif vd > max_vd:
                max_vd = vd
        
        # get vd_intervals_dict for this answer
        vd_intervals_dict = get_vdJumpIntervals(min_vd, max_vd, vd_jump_step_size)
        if len(vd_intervals_dict) == 0:
            print(f"no enough interval for answer {qidAiTup} of {commName}, skip.")
            continue

        signalsOfStarVdAndEndVd = defaultdict() # for one answer
        vdInterval2voteAccumulatedCount = defaultdict() # for one answer
        firstStartVd = 0
        if 0 not in vd_intervals_dict.keys():
            firstStartVd = list(vd_intervals_dict.keys())[0]
        cur_startVdList =[firstStartVd]
        # initialize
        for start_vd, end_vd in vd_intervals_dict.items():
            if start_vd == 0 : # because the first vote of each answer's cur_vd=0 which was removed, so add back here when initialize for each anwswer
                vdInterval2voteAccumulatedCount[(start_vd,end_vd)]= 1
                if certainRank != None:
                    vdInterval2voteAccumulatedCount[(start_vd,end_vd)]= 0
                signalsOfStarVdAndEndVd[start_vd] =True
                signalsOfStarVdAndEndVd[end_vd] =False
            else:
                vdInterval2voteAccumulatedCount[(start_vd,end_vd)]= 0
                signalsOfStarVdAndEndVd[start_vd] =False
                signalsOfStarVdAndEndVd[end_vd] =False
        
        for v in values:
            cur_vote, cur_vd, rank, t = v
  
            if cur_vd in vd_intervals_dict.keys(): # one of start vd
                startSignal = signalsOfStarVdAndEndVd[cur_vd]
                if not startSignal: # hasn't started yet
                    signalsOfStarVdAndEndVd[cur_vd] = True
                    if cur_vd not in cur_startVdList:
                        cur_startVdList.append(cur_vd)
                    # accumulate cur vote
                    start_vd = cur_vd
                    end_vd = vd_intervals_dict[start_vd]
                    vdInterval2voteAccumulatedCount[(start_vd,end_vd)] +=1
                else: # already started
                    for start_vd in cur_startVdList:
                        end_vd = vd_intervals_dict[start_vd]
                        if cur_vd < end_vd:
                            endSignal = signalsOfStarVdAndEndVd[end_vd]
                            if not endSignal: # hasn't ended yet for cur_startVd interval, accumulated
                                vdInterval2voteAccumulatedCount[(start_vd,end_vd)] +=1
                            else: # already end. ignore
                                continue
            
            elif cur_vd in vd_intervals_dict.values(): # one of end vd
                if len(cur_startVdList) == 0: # hasn't started yet, ignore
                    continue
                else: # already started
                    for start_vd in cur_startVdList:
                        end_vd = vd_intervals_dict[start_vd]
                        if cur_vd < end_vd: # not the target end vd, treat as normal vote, 
                            endSignal = signalsOfStarVdAndEndVd[end_vd]
                            if not endSignal: # hasn't ended yet for cur_startVd interval, accumulated
                                vdInterval2voteAccumulatedCount[(start_vd,end_vd)] +=1
                            else: # already end. ignore
                                continue
                        elif cur_vd == end_vd: # reach the target end vd, and accumulat cur vote
                            signalsOfStarVdAndEndVd[end_vd] = True
                            vdInterval2voteAccumulatedCount[(start_vd,end_vd)] +=1
                            # remove the corresponding started vd from cur_startVdList, becuase this interval is ended
                            cur_startVdList.remove(start_vd)
                        else: # cur_vd > end_vd, not for this interval, try next interval
                            continue
            
            else: # not a start vd, nor an end vd, should be accumulated to cur started intervals
                for start_vd in cur_startVdList:
                    end_vd = vd_intervals_dict[start_vd]
                    if (cur_vd >0 and cur_vd < end_vd) or (cur_vd < 0 and cur_vd > end_vd): # not the target end vd, treat as normal vote, 
                        endSignal = signalsOfStarVdAndEndVd[end_vd]
                        if not endSignal: # hasn't ended yet for cur_startVd interval, accumulated
                            vdInterval2voteAccumulatedCount[(start_vd,end_vd)] +=1
                        else: # already end. ignore
                            continue
                    else: # not for this interval, try next interval
                        continue
        
        # updated accumulated counts of this answer to total
        for tup, count in vdInterval2voteAccumulatedCount.items():
            if count > 0 :
                total_vdInterval2voteAccumulatedCount[tup].append(count)
                if count < vd_jump_step_size:
                    print("debug")
   
    # delete empty tails
    for i in range (len(total_vdInterval2voteAccumulatedCount)-1,0, -1):
        if len(list(total_vdInterval2voteAccumulatedCount.items())[i][1])==0:
            del total_vdInterval2voteAccumulatedCount[list(total_vdInterval2voteAccumulatedCount.items())[i][0]]
        else:
            break

    # save total_vdInterval2voteAccumulatedCount
    if certainRank != None:
        with open(intermediate_directory+'/'+f'total_vdInterval2voteAccumulatedCount_jumpSize{vd_jump_step_size}_forCertainRank{certainRank}.dict', 'wb') as outputFile:
            pickle.dump(total_vdInterval2voteAccumulatedCount, outputFile) 
            logtext = f"saved total_vdInterval2voteAccumulatedCount_jumpSize{vd_jump_step_size}_forCertainRank{certainRank} of {commName}. length: {len(total_vdInterval2voteAccumulatedCount)}\n"
            writeIntoLog(logtext, commDir, logFileName)
            print(logtext)
    else:
        with open(intermediate_directory+'/'+f'total_vdInterval2voteAccumulatedCount_jumpSize{vd_jump_step_size}.dict', 'wb') as outputFile:
            pickle.dump(total_vdInterval2voteAccumulatedCount, outputFile) 
            logtext = f"saved total_vdInterval2voteAccumulatedCount_jumpSize{vd_jump_step_size} of {commName}. length: {len(total_vdInterval2voteAccumulatedCount)}\n"
            writeIntoLog(logtext, commDir, logFileName)
            print(logtext)
        
    
    # load 
    if certainRank != None:
        with open(intermediate_directory+'/'+f'total_vdInterval2voteAccumulatedCount_jumpSize{vd_jump_step_size}_forCertainRank{certainRank}.dict', 'rb') as inputFile:
            total_vdInterval2voteAccumulatedCount = pickle.load( inputFile)
    else:
        with open(intermediate_directory+'/'+f'total_vdInterval2voteAccumulatedCount_jumpSize{vd_jump_step_size}.dict', 'rb') as inputFile:
            total_vdInterval2voteAccumulatedCount = pickle.load( inputFile)
    
    # plot for averageVoteCountAtEachVoteDiffJumpInterval
    plotAverageVoteCountAtEachVoteDiffJumpInterval(commName,commDir,total_vdInterval2voteAccumulatedCount,logFileName, vd_jump_step_size, list(vd_intervals_dict_comm.items())[0], list(vd_intervals_dict_comm.items())[-1], certainRank)
        
        
    

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
    myFun(commDir_sizes_sortedlist[349][0], commDir_sizes_sortedlist[349][1])
    # test on comm "stackoverflow" to debug
    # myFun(commDir_sizes_sortedlist[359][0], commDir_sizes_sortedlist[359][1])

    # skip splitted communities
    splitted_comms = ['tex.stackexchange','meta.stackexchange','softwareengineering.stackexchange','superuser','math.stackexchange','worldbuilding.stackexchange','codegolf.stackexchange','stackoverflow']

    """
    # # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for tup in commDir_sizes_sortedlist:
        commName = tup[0]
        commDir = tup[1]
        try:
            p = mp.Process(target=myFun, args=(commName,commDir))
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
    
    """       
    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('descriptive10 Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
