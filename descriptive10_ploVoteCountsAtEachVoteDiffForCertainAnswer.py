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

import descriptive4_Conformity_forStackOverflow


def myFun(commName, commDir):
    print(f"comm {commName} running on {mp.current_process().name}")
    logFileName = 'descriptive10_Log.txt'

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data files
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')
    
    splitted_intermediate_data_folder = os.path.join(intermediate_directory, r'splitted_intermediate_data_folder')
    split_valuesForConformtiyComputation_forEachYear_directory = os.path.join(splitted_intermediate_data_folder, r'totalValuesForConformtiyComputation_forEachYear_parts_folder')
    if not os.path.exists(split_valuesForConformtiyComputation_forEachYear_directory): 
        print("Exception: no split_valuesForConformtiyComputation_forEachYear_directory!")

    yearFiles = [ f.path for f in os.scandir(split_valuesForConformtiyComputation_forEachYear_directory) if f.path.endswith('.dict') ]
    # sort files based on year
    yearFiles.sort(key=lambda p: int(p.strip(".dict").split("_")[-1]))
    yearsCount = len(yearFiles)
                                
    print(f"there are {yearsCount} splitted files in {commName}")

    # get total_valuesForConformityComputation_forEachVote
    with open(intermediate_directory+'/'+f'total_valuesForConformityComputation_forEachVote_original.dict', 'rb') as inputFile:
        total_valuesForConformityComputation_forEachVote_original = pickle.load( inputFile)
        print(f"total_valuesForConformityComputation_forEachVote_original of {commName} is loaded.")

    
    # convert valuesTupleAtCertainRank to dict, (qid, ai) as key
    answer2values = defaultdict()
    for tup in total_valuesForConformityComputation_forEachVote_original:
        if len(tup)==8:
            cur_vote, cur_n_pos, cur_n_neg,rank, t, cur_year, qid, ans_index = tup
        else:
            cur_vote, cur_n_pos, cur_n_neg,rank, t, qid, ans_index = tup
        if (qid,ans_index) not in answer2values.keys():
            answer2values[(qid,ans_index)] = [(cur_vote, cur_n_pos, cur_n_neg)]
        else: 
            answer2values[(qid,ans_index)].append((cur_vote, cur_n_pos, cur_n_neg))
    total_valuesForConformityComputation_forEachVote_original.clear()

    # find the answer with the most votes
    maxVotesIndex = np.argmax([len(tup[1]) for tup in answer2values.items()])
    targetAnswerTuple = list(answer2values.items())[maxVotesIndex]
    targetValuesList = targetAnswerTuple[1]

    # create a voteDiff2voteCounts table with targetValuseList
    voteDiff2voteCounts_forCertainAnswer = defaultdict()
    for values in targetValuesList:
        cur_vote, cur_n_pos, cur_n_neg = values
        cur_vote_diff = cur_n_pos - cur_n_neg

        if cur_vote_diff in voteDiff2voteCounts_forCertainAnswer.keys():
                if cur_vote == 1: # add on pos
                    voteDiff2voteCounts_forCertainAnswer[cur_vote_diff]['pos'] = voteDiff2voteCounts_forCertainAnswer[cur_vote_diff]['pos'] + 1
                else: # add on neg
                    voteDiff2voteCounts_forCertainAnswer[cur_vote_diff]['neg'] = voteDiff2voteCounts_forCertainAnswer[cur_vote_diff]['neg'] + 1
        else:
            if cur_vote == 1: # add on pos
                voteDiff2voteCounts_forCertainAnswer[cur_vote_diff] = {'pos':1, 'neg':0}
            else: # add on neg
                voteDiff2voteCounts_forCertainAnswer[cur_vote_diff] = {'pos':0, 'neg':1}
    
    # plot for voteDiff2voteCounts_atCertainRank_forCertainAnswer 
    certainAnswer = targetAnswerTuple[0]
    descriptive4_Conformity_forStackOverflow.plotVoteProportionAtEachVoteDiffForCertainRank(commName,commDir,year, voteDiff2voteCounts_forCertainAnswer,logFileName, 'AllRank', certainAnswer)
        
        
    

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
    # myFun_SOF(commDir_sizes_sortedlist[359][0], commDir_sizes_sortedlist[359][1])

    # skip splitted communities
    splitted_comms = ['tex.stackexchange','meta.stackexchange','softwareengineering.stackexchange','superuser','math.stackexchange','worldbuilding.stackexchange','codegolf.stackexchange','stackoverflow']

    
    # # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for tup in commDir_sizes_sortedlist[359:]:
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
    
            
    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('descriptive10 Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
