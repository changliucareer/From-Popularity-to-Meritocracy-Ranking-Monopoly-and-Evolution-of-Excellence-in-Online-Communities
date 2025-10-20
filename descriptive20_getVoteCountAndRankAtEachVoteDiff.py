import os
#First print the current working directory
print("Current Working Directory", os.getcwd())
###############################################################
from toolFunctions import format_time, writeIntoLog, insertEventInUniversalTimeStep
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
"""
def whichYear(universal_timestepIndex, year2UniversalTimeStepIndex):
    for year, lastIndex in year2UniversalTimeStepIndex.items():
        if universal_timestepIndex <= lastIndex:
            return year
    return None


def myAction (question_tuple, commName):
    qid = question_tuple[0]
    content = question_tuple[1]
    eventList = content['eventList']
    lts = content['local_timesteps']

    # extract values for conformity computation for each vote event
    
    # a list of tuple to keep all needed values to compute conformity at any time tick (except rank).rank is used for Herding bias Verifying
    # each tuple includes (cur_vote, cur_n_pos, cur_n_neg, rank)
    valuesForConformityComputation = [] 

    firstVoteRemovedFlagForEachAnswer =  []

    for i,e in enumerate(eventList):
        t = lts[i]
        if commName == 'stackoverflow':
            cur_year = t[2].year
            t = t[1] # which is vid, can be used to sort
        eventType = e['et']
        if eventType != 'v': # skip all event that is not a vote
            continue

        if e['ai'] in firstVoteRemovedFlagForEachAnswer: # current vote's target answer already has the first vote removed
            ans_index = e['ai']
            ranksOfAnswersBeforeT = e['ranks']
            rank = ranksOfAnswersBeforeT[ans_index]
            cur_vote = e['v']
            cur_n_pos = e['n_pos']
            cur_n_neg = e['n_neg']

            if commName == 'stackoverflow':
                valuesForConformityComputation.append((cur_vote, cur_n_pos, cur_n_neg,rank, t, cur_year, qid, ans_index))
            else:
                valuesForConformityComputation.append((cur_vote, cur_n_pos, cur_n_neg,rank, t, qid, ans_index))

        else:# current vote is its target answer's first vote, don't use as sample
            firstVoteRemovedFlagForEachAnswer.append(e['ai'])

    print(f"question {qid} of {commName} on {mp.current_process().name} return")

    # return processed question's qid, content, the number of answers with votes, the number of real votes as samples
    return valuesForConformityComputation
"""

def myFun(commName, commDir):
    print(f"comm {commName} running on {mp.current_process().name}")
    logFileName = 'descriptive20_otherComm_log.txt'

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data files
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')

    # load total_valuesForConformityComputation_forEachVote
    with open(intermediate_directory+'/'+'total_valuesForConformityComputation_forEachVote_original.dict', 'rb') as inputFile:
        total_valuesForConformityComputation_forEachVote = pickle.load( inputFile)
    
    totalVoteCount = len(total_valuesForConformityComputation_forEachVote)
    
    # update values for conformity computation along with real time voteDiff2voteCounts_comm table
    voteDiff2voteCountsAndRank_comm = defaultdict()

    for i, tup in enumerate(total_valuesForConformityComputation_forEachVote):
        print(f"processing {i+1}th/{totalVoteCount} vote of {commName}...")
        cur_vote, cur_n_pos, cur_n_neg,rank, t, qid, ans_index = tup
        cur_vote_diff = cur_n_pos - cur_n_neg
        
        if cur_vote_diff in voteDiff2voteCountsAndRank_comm.keys():
            voteDiff2voteCountsAndRank_comm[cur_vote_diff].append({'vote':cur_vote, 'rank':rank}) 
        else:
            voteDiff2voteCountsAndRank_comm[cur_vote_diff] = [{'vote':cur_vote, 'rank':rank}] 
        
        
    # save voteDiff2voteCounts_comm_atCertainRank
    with open(intermediate_directory+'/'+f'voteDiff2voteCountAndRank_comm.dict', 'wb') as outputFile:
        pickle.dump( voteDiff2voteCountsAndRank_comm, outputFile) 
        logtext = f"saved  voteDiff2voteCountsAndRank_comm of {commName}.\n"
        writeIntoLog(logtext, commDir, logFileName)
        print(logtext)
    
def main():

    t0=time.time()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("other sorted CommDir loaded.")

    # # test on comm "coffee.stackexchange" to debug
    myFun(commDir_sizes_sortedlist[166][0], commDir_sizes_sortedlist[166][1])
    # # test on comm "datascience.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[301][0], commDir_sizes_sortedlist[301][1])
    # test on comm "webapps.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[305][0], commDir_sizes_sortedlist[305][1])
    # test on comm "stackoverflow" to debug
    # myFun_SOF(commDir_sizes_sortedlist[359][0], commDir_sizes_sortedlist[359][1])

    # skip splitted communities
    splitted_comms = ['tex.stackexchange','meta.stackexchange','softwareengineering.stackexchange','superuser','math.stackexchange','worldbuilding.stackexchange','codegolf.stackexchange','stackoverflow']

    
    # # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for tup in commDir_sizes_sortedlist[:300]:
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
    
    
            
    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('descriptive20 get vote count and rank at each vote diff Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
