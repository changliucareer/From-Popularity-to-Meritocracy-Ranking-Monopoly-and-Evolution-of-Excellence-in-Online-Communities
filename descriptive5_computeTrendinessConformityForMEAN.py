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

import descriptive1_computeVoteCountAtEachRank
import descriptive2_Trendiness
import descriptive4_Conformity_forStackOverflow

def whichYear(universal_timestepIndex, year2UniversalTimeStepIndex):
    for year, lastIndex in year2UniversalTimeStepIndex.items():
        if universal_timestepIndex <= lastIndex:
            return year
    return None

def conformity_myAction (question_tuple, commName, year2UniversalTimeStepIndex, universal_timestepsDir):
    qid = question_tuple[0]
    content = question_tuple[1]
    eventList = content['eventList']
    lts = content['local_timesteps']

    with open(universal_timestepsDir, 'rb') as inputFile:
        universal_timesteps = pickle.load( inputFile)

    # extract values for conformity computation for each vote event
    
    # a list of tuple to keep all needed values to compute conformity at any time tick (except rank).rank is used for Herding bias Verifying
    # each tuple includes (cur_vote, cur_n_pos, cur_n_neg, rank)
    valuesForConformityComputation = [] 

    firstVoteRemovedFlagForEachAnswer =  []

    for i,e in enumerate(eventList):
        if commName == 'stackoverflow':
            cur_dt = lts[i][2]
        else:
            cur_dt = universal_timesteps[lts[i]][2]

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

            valuesForConformityComputation.append((cur_vote, cur_n_pos, cur_n_neg,rank, cur_dt))

        else:# current vote is its target answer's first vote, don't use as sample
            firstVoteRemovedFlagForEachAnswer.append(e['ai'])

    print(f"question {qid} of {commName} on {mp.current_process().name} return")
    universal_timesteps.clear()
    # return processed question's qid, content, the number of answers with votes, the number of real votes as samples
    return valuesForConformityComputation

def computeVoteCountAtEachRank_myFunc(MEANcommName, MEANcommDir, commName2commDir):
    print(f"comm {MEANcommName} running on {mp.current_process().name}")

    with open('MEAN0_samples_10each_fromEventList.dict', 'rb') as inputFile:
        return_sample_normalDict = pickle.load( inputFile)
        print("return_sample_normalDict loaded.")

    # prepare args
    args = []
    commName2year2UniversalTimeStepIndex = defaultdict()
    for commName, QuestionsList in return_sample_normalDict.items():
        if commName not in commName2year2UniversalTimeStepIndex.keys(): # need to load corresponding year2UniversalTimeStepIndex
            intermediate_directory = os.path.join(commName2commDir[commName], r'intermediate_data_folder')
            with open(intermediate_directory+'/'+'year2UniversalTimeStepIndex.dict', 'rb') as inputFile:
                year2UniversalTimeStepIndex = pickle.load( inputFile)
                commName2year2UniversalTimeStepIndex[commName] = year2UniversalTimeStepIndex
        
        for question_tuple in QuestionsList: 
            args.append((question_tuple, commName, commName2year2UniversalTimeStepIndex[commName]))    
    
    # process Questions chunk by chunk
    n_proc = mp.cpu_count()-2 # left 2 cores to do others
    all_outputs = []
    with mp.Pool(processes=n_proc) as pool:
        # issue tasks to the process pool and wait for tasks to complete
        results = pool.starmap(descriptive1_computeVoteCountAtEachRank.myAction, args , chunksize=n_proc)
        # process pool is closed automatically
        for res in results:
            if res != None:
                if isinstance(res, str):
                    print(res)
                else:
                    all_outputs.append(res)
            else:
                print(f"None")
    
    return_sample_normalDict.clear()
    results.clear()
    # combine results
    print("start to combine year2voteCountAtEachYear to each question...")
    year2voteCountAtEachRank_total = defaultdict()
    for tup in all_outputs: 
        qid = tup[0]
        print(f"combining question {qid} results to total for {MEANcommName}")
        year2voteCountAtEachRank = tup[1]

        for year, rank2voteCount in year2voteCountAtEachRank.items():
            if year not in year2voteCountAtEachRank_total.keys():
                year2voteCountAtEachRank_total[year] = copy.deepcopy(rank2voteCount)
                # sort
                year2voteCountAtEachRank_total = dict(sorted(year2voteCountAtEachRank_total.items()))
            else: # need to combine two rank2voteCount
                combined_rank2voteCount = copy.deepcopy(rank2voteCount)
                previous_rank2voteCount = year2voteCountAtEachRank_total[year]
                for previous_rank, previous_vc in previous_rank2voteCount.items():
                    if previous_rank in combined_rank2voteCount.keys():
                        combined_rank2voteCount[previous_rank] = combined_rank2voteCount[previous_rank] + previous_vc
                    else:
                        combined_rank2voteCount[previous_rank] = previous_vc
                # sort
                combined_rank2voteCount = dict(sorted(combined_rank2voteCount.items()))
                # update year2voteCountAtEachRank_total
                year2voteCountAtEachRank_total[year] = combined_rank2voteCount
    # sort
    year2voteCountAtEachRank_total = dict(sorted(year2voteCountAtEachRank_total.items()))
                
    # save updated Questions
    with open('MEAN_samples_10each_yearlyVoteCountAtEachRank.dict', 'wb') as outputFile:
        pickle.dump(year2voteCountAtEachRank_total, outputFile) 
        print(f"saved MEAN_yearlyVoteCountAtEachRank.dict for {MEANcommName}.")  

def Trendiness_myFun(MEANcommName, MEANcommDir):
    print(f"comm {MEANcommName} running on {mp.current_process().name}")
    logFileName = 'descriptive5_computeTrendinessConformityForMEAN_log.txt'
    
    with open('MEAN_samples_10each_yearlyVoteCountAtEachRank.dict', 'rb') as inputFile:
        year2voteCountAtEachRank_total = pickle.load( inputFile)
        print("year2voteCountAtEachRank_total loaded.")

    # process Questions chunk by chunk
    all_outputs = []
    n_proc = mp.cpu_count()-2 # left 2 cores to do others
    with mp.Pool(processes=n_proc) as pool:
        args = []
        for year, rank2voteCount in year2voteCountAtEachRank_total.items():
            args.append((MEANcommName,MEANcommDir, year,rank2voteCount,logFileName))
        # issue tasks to the process pool and wait for tasks to complete
        results = pool.starmap(descriptive2_Trendiness.myAction, args , chunksize=n_proc)
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
                
    return_trendiness_normalDict = defaultdict()
    return_trendiness_normalDict[MEANcommName] = year2fittingResults

    writeIntoLog(str(return_trendiness_normalDict), MEANcommDir, logFileName)
    
    with open('MEAN_samples_10each_descriptive_TrendinessFittingResults.dict', 'wb') as outputFile:
        pickle.dump(return_trendiness_normalDict, outputFile)
        print(f"saved return_trendiness_normalDict for MEAN")

def getValuesForConformityComputation_myFun(MEANcommName, MEANcommDir, commName2commDir):
    print(f"comm {MEANcommName} running on {mp.current_process().name}")
    logFileName = 'descriptive5_computeTrendinessConformityForMEAN_log.txt'

    # load intermediate_data files
    with open('MEAN0_samples_10each_fromEventList.dict', 'rb') as inputFile:
        return_sample_normalDict = pickle.load( inputFile)
        print("return_sample_normalDict loaded.")

    # prepare args
    args = []
    commName2year2UniversalTimeStepIndex = defaultdict()
    commName2universal_timestepsDir = defaultdict()
    for commName, QuestionsList in return_sample_normalDict.items():
        if commName not in commName2year2UniversalTimeStepIndex.keys(): # need to load corresponding year2UniversalTimeStepIndex
            intermediate_directory = os.path.join(commName2commDir[commName], r'intermediate_data_folder')
            with open(intermediate_directory+'/'+'year2UniversalTimeStepIndex.dict', 'rb') as inputFile:
                year2UniversalTimeStepIndex = pickle.load( inputFile)
                commName2year2UniversalTimeStepIndex[commName] = year2UniversalTimeStepIndex

            commName2universal_timestepsDir[commName] = os.path.join(intermediate_directory, r'universal_timesteps_afterCombineQAVH.dict')
        
        for question_tuple in QuestionsList: 
            args.append((question_tuple, commName, commName2year2UniversalTimeStepIndex[commName], commName2universal_timestepsDir[commName]))
        
        QuestionsList.clear()
    
    # process Questions chunk by chunk
    n_proc = mp.cpu_count()-2 # left 2 cores to do others
    with mp.Pool(processes=n_proc) as pool:
        # issue tasks to the process pool and wait for tasks to complete
        all_outputs = pool.starmap(conformity_myAction, args , chunksize=n_proc)
    
    return_sample_normalDict.clear()
   
    # combine all_outputs into a whole list
    total_valuesForConformityComputation_forEachVote = []
    for ll in all_outputs:
        if len(ll)>0:
            total_valuesForConformityComputation_forEachVote.extend(ll)
    
    all_outputs.clear()

    # sort each vote tuple by time step index
    total_valuesForConformityComputation_forEachVote.sort(key=lambda t: t[4])
    totalVoteCount = len(total_valuesForConformityComputation_forEachVote)

    # update values for conformity computation along with real time voteDiff2voteCounts_comm table
    voteDiff2voteCounts_comm = defaultdict()
    certainRank = 6
    voteDiff2voteCounts_comm_atCertainRank = defaultdict()

    year = total_valuesForConformityComputation_forEachVote[0][4].year

    year2valuesForConformityComputation_forEachVote_onlyForCurYear = defaultdict()

    startIndex = 0
    for i, tup in enumerate(total_valuesForConformityComputation_forEachVote):
        print(f"processing {i+1}th/{totalVoteCount} vote of {commName}...")
        cur_vote, cur_n_pos, cur_n_neg,rank, cur_dt = tup
        cur_vote_diff = cur_n_pos - cur_n_neg
        cur_year = cur_dt.year
        
        # update valuesForConformityComputation year by year
        if cur_year == year:
            # update valuesForConformityComputation
            if cur_vote_diff in voteDiff2voteCounts_comm.keys():
                cur_voteDiff2voteCounts = copy.deepcopy(voteDiff2voteCounts_comm[cur_vote_diff]) # must be real time table
            else:
                cur_voteDiff2voteCounts = {'pos':0, 'neg':0} # cur_vote_diff has never been voted, assume the possibility of pos and neg is the same

            total_valuesForConformityComputation_forEachVote[i] =(cur_vote,cur_n_pos,cur_n_neg,cur_voteDiff2voteCounts)
        
        elif cur_year < year:
            print(f"Exception: cur_year < year for {i+1}th time step")
        
        else: # cur_year > year
            # save previous year's all valuesForConformityComputation
            endIndex = i
            valuesForConformityComputation_forEachVote_onlyForCurYear = total_valuesForConformityComputation_forEachVote[startIndex:endIndex]
            year2valuesForConformityComputation_forEachVote_onlyForCurYear [year] = copy.deepcopy(valuesForConformityComputation_forEachVote_onlyForCurYear)
            print(f"get valuesForConformityComputation_forCurYear for year {year} of {commName}, length: {len(valuesForConformityComputation_forEachVote_onlyForCurYear)}.")

            # start a new year
            year = cur_year
            startIndex = endIndex
            
            # aggregate the current event to total_valuesForConformityComputation_forEachVote
            if cur_vote_diff in voteDiff2voteCounts_comm.keys():
                cur_voteDiff2voteCounts = copy.deepcopy(voteDiff2voteCounts_comm[cur_vote_diff]) # must be real time table
            else:
                cur_voteDiff2voteCounts = {'pos':0, 'neg':0} # cur_vote_diff has never been voted, assume the possibility of pos and neg is the same

            total_valuesForConformityComputation_forEachVote[i] =(cur_vote,cur_n_pos,cur_n_neg,cur_voteDiff2voteCounts)

        # update voteDiff2voteCounts table with cur_vote
        if cur_vote_diff in voteDiff2voteCounts_comm.keys():
            if cur_vote == 1: # add on pos
                voteDiff2voteCounts_comm[cur_vote_diff]['pos'] = voteDiff2voteCounts_comm[cur_vote_diff]['pos'] + 1
            else: # add on neg
                voteDiff2voteCounts_comm[cur_vote_diff]['neg'] = voteDiff2voteCounts_comm[cur_vote_diff]['neg'] + 1
        else:
            if cur_vote == 1: # add on pos
                voteDiff2voteCounts_comm[cur_vote_diff] = {'pos':1, 'neg':0}
            else: # add on neg
                voteDiff2voteCounts_comm[cur_vote_diff] = {'pos':0, 'neg':1}
        
        if rank == certainRank: # updated table for certain Rank
            if cur_vote_diff in voteDiff2voteCounts_comm_atCertainRank.keys():
                if cur_vote == 1: # add on pos
                    voteDiff2voteCounts_comm_atCertainRank[cur_vote_diff]['pos'] = voteDiff2voteCounts_comm_atCertainRank[cur_vote_diff]['pos'] + 1
                else: # add on neg
                    voteDiff2voteCounts_comm_atCertainRank[cur_vote_diff]['neg'] = voteDiff2voteCounts_comm_atCertainRank[cur_vote_diff]['neg'] + 1
            else:
                if cur_vote == 1: # add on pos
                    voteDiff2voteCounts_comm_atCertainRank[cur_vote_diff] = {'pos':1, 'neg':0}
                else: # add on neg
                    voteDiff2voteCounts_comm_atCertainRank[cur_vote_diff] = {'pos':0, 'neg':1}
        
    # save for the last year
    valuesForConformityComputation_forEachVote_onlyForCurYear = total_valuesForConformityComputation_forEachVote[startIndex:]
    year2valuesForConformityComputation_forEachVote_onlyForCurYear [year] = copy.deepcopy(valuesForConformityComputation_forEachVote_onlyForCurYear)
    print(f"get valuesForConformityComputation_forCurYear for year {year} of {commName}, length: {len(valuesForConformityComputation_forEachVote_onlyForCurYear)}.")

    
    # save voteDiff2voteCounts_comm_atCertainRank
    with open(f'MEAN_voteDiff2voteCounts_comm_atCertainRank{certainRank}.dict', 'wb') as outputFile:
        pickle.dump(voteDiff2voteCounts_comm_atCertainRank, outputFile) 
        logtext = f"saved voteDiff2voteCounts_comm_atCertainRank of {commName}.\n"
        writeIntoLog(logtext, MEANcommDir, logFileName)
        print(logtext)

                
    # save updated Questions
    with open('MEAN_samples_10each_yearlyValuesForConformityComputation.dict', 'wb') as outputFile:
        pickle.dump(year2valuesForConformityComputation_forEachVote_onlyForCurYear, outputFile) 
        print(f"saved yearlyvaluesForConformityComputation.dict for {MEANcommName}.")

def myAction (commName,commDir, year,valuesForConformityComputation,logFileName):
    cur_logSum = 0
    cur_voteCount = len(valuesForConformityComputation) 
    for i, tup in enumerate(valuesForConformityComputation):
        print(f"process year {year} {i+1}th/{cur_voteCount} for {commName}")
        cur_vote, cur_n_pos, cur_n_neg, cur_voteDiff2voteCounts = tup
        a = cur_voteDiff2voteCounts['pos']
        b = cur_voteDiff2voteCounts['neg']
        if cur_n_pos >= cur_n_neg:
            cur_logSum += np.log(a+1) - np.log(b+1)
        else:
            cur_logSum += np.log(b+1) - np.log(a+1)
     
    return (year, cur_logSum, cur_voteCount)

def Conformity_myFun(MEANcommName, MEANcommDir):
    print(f"comm {MEANcommName} running on {mp.current_process().name}")
    logFileName = 'descriptive5_computeTrendinessConformityForMEAN_log.txt'
    
    with open('MEAN_samples_10each_yearlyValuesForConformityComputation.dict', 'rb') as inputFile:
        year2valuesForConformityComputation = pickle.load( inputFile)

    print(f"start to compute Conformity for MEAN...")
    # compute Conformity
    year2Conformity= defaultdict()
    logSum = 0
    totalVoteCount = 0
    certainRank = 6 # an arbitrarily choosen rank for conformity verifying
    # create the table for certain rank
    voteDiff2voteCounts = defaultdict() # a dict to keep real time vote difference to vote counts of pos and neg (as #vote+_{vd^t} in conformity equation)
    
    for year, valuesForConformtiyComputation in year2valuesForConformityComputation.items():
        year, cur_logSum, cur_voteCount = myAction(MEANcommName, MEANcommDir, year, valuesForConformtiyComputation, logFileName)
        
        totalVoteCount += cur_voteCount
        logSum += cur_logSum
        if totalVoteCount !=0:
            Conformity = math.exp(logSum/totalVoteCount)
            logtext = f" year {year} logSum:{logSum}, Conformity: {Conformity}, size: {totalVoteCount}.\n"
            writeIntoLog(logtext, MEANcommDir, logFileName)
            print(logtext)
        year2Conformity[year] = {'conformtiy':Conformity,'size':totalVoteCount}
            
        # plot Conformity verifying 
        if year == 2022:  
            # load voteDiff2voteCounts
            certainRank = 6
            try:
                with open(f'MEAN_voteDiff2voteCounts_comm_atCertainRank{certainRank}.dict', 'rb') as inputFile:
                    voteDiff2voteCounts = pickle.load( inputFile)
                    print(f"loaded voteDiff2voteCounts till 2022 for {MEANcommName}.")  
                    descriptive4_Conformity_forStackOverflow.plotVoteProportionAtEachVoteDiffForCertainRank(MEANcommName,MEANcommDir,year, voteDiff2voteCounts,logFileName, certainRank)
            except Exception as e:
                print(e)
                writeIntoLog(f"fail to plot:{e}", MEANcommDir, logFileName)
            
    return_conformity_normalDict = defaultdict()
    return_conformity_normalDict[MEANcommName] = year2Conformity
    writeIntoLog(str(return_conformity_normalDict), MEANcommDir, logFileName)

    with open('MEAN_samples_10each_descriptive_ConformityAndSize.dict', 'wb') as outputFile:
        pickle.dump(return_conformity_normalDict, outputFile)
        print(f"saved return_conformity_normalDict, {len(return_conformity_normalDict)} comms.")

def main():

    t0=time.time()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("other sorted CommDir loaded.")

    # convert commDir_sizes_sortedlist to dict
    commName2commDir = defaultdict()
    for tup in commDir_sizes_sortedlist:
        commName = tup[0]
        commDir = tup[1]
        commName2commDir[commName] = commDir
    
    # Trendiness for MEAN
    computeVoteCountAtEachRank_myFunc('MEAN', os.getcwd(), commName2commDir)
    
    Trendiness_myFun('MEAN', os.getcwd())
    
    # Conformity for MEAN
    getValuesForConformityComputation_myFun('MEAN', os.getcwd(), commName2commDir)
    
    Conformity_myFun('MEAN', os.getcwd())
            
    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('compute Trendiness and Conformity for MEAN Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
