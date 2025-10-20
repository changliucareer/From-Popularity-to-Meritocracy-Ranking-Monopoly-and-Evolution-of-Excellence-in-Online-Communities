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
import descriptive3_getValuesForConformityComputation
import descriptive4_Conformity
import descriptive4_Conformity_forStackOverflow

def whichYear(universal_timestepIndex, year2UniversalTimeStepIndex):
    for year, lastIndex in year2UniversalTimeStepIndex.items():
        if universal_timestepIndex <= lastIndex:
            return year
    return None

def computeVoteCountAtEachRank_myFunc(parentCommName, parentCommDir, subCommName, year2UniversalTimeStepIndex, subComm_QuestionsWithEventList_directory, subComm_yearlyVoteCountAtEachRank_directory, logFileName):
    # check whether already done this step, skip
    resultFileDir = subComm_yearlyVoteCountAtEachRank_directory
    if os.path.exists(resultFileDir):
        print(f"{subCommName} has already done this step.")
        return
    
    
    print(f"sub comm {subCommName} of {parentCommName} computeVoteCountAtEachRank running on {mp.current_process().name}")
    try:
        with open(subComm_QuestionsWithEventList_directory, 'rb') as inputFile:
            sub_Questions = pickle.load( inputFile)
            qCount = len(sub_Questions)
            print(f"subComm {subCommName} QuestionsWithEventList loaded, length {qCount}.")
    except:
        exceptionLog = f"{subCommName} hasn't been created yet, no EventList data!"
        return exceptionLog

    
    # prepare args
    args = zip(list(sub_Questions.items()), [parentCommName]*qCount, [year2UniversalTimeStepIndex]*qCount)
    
    # process Questions chunk by chunk
    n_proc = mp.cpu_count()
    all_outputs = []
    with mp.Pool(processes=n_proc) as pool:
        # issue tasks to the process pool and wait for tasks to complete
        results = pool.starmap(descriptive1_computeVoteCountAtEachRank.myAction, args , chunksize=1)
        # process pool is closed automatically
        for res in results:
            if res != None:
                all_outputs.append(res)
    
    sub_Questions.clear()
    results.clear()
    # combine results
    print("start to combine year2voteCountAtEachYear to each question...")
    year2voteCountAtEachRank_total = defaultdict()
    for tup in all_outputs: 
        qid = tup[0]
        print(f"combining question {qid} results to total for {subCommName}")
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
                
    # save updated results
    with open(subComm_yearlyVoteCountAtEachRank_directory, 'wb') as outputFile:
        pickle.dump(year2voteCountAtEachRank_total, outputFile) 
        logtext = f"saved subComm_yearlyVoteCountAtEachRank for {subCommName}.\n"
        writeIntoLog(logtext, parentCommDir, logFileName)
        print(logtext)

def Trendiness_myFun(parentCommName, parentCommDir, subCommName,subComm_yearlyVoteCountAtEachRank_directory, logFileName):
    print(f"sub comm {subCommName} of {parentCommName} Trendiness_myFun running on {mp.current_process().name}")
    
    with open(subComm_yearlyVoteCountAtEachRank_directory, 'rb') as inputFile:
        year2voteCountAtEachRank_total = pickle.load( inputFile)
        print("year2voteCountAtEachRank_total loaded.")

    # process Questions chunk by chunk
    all_outputs = []
    n_proc = mp.cpu_count()
    with mp.Pool(processes=n_proc) as pool:
        args = []
        for year, rank2voteCount in year2voteCountAtEachRank_total.items():
            args.append((subCommName,parentCommDir, year,rank2voteCount,logFileName))
        # issue tasks to the process pool and wait for tasks to complete
        results = pool.starmap(descriptive2_Trendiness.myAction, args , chunksize=1)
        # process pool is closed automatically
        for res in results:
            if res != None:
                all_outputs.append(res)

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
                
    return year2fittingResults

def getValuesForConformityComputation_myFun(parentCommName, parentCommDir, subCommName, subCommDir, subComm_QuestionsWithEventList_directory,subComm_yearlyValuesForConformityComputation_folder, logFileName):
    certainRank = 6
    # check whether already done this step, skip
    resultFileDir = subCommDir+'/'+f'voteDiff2voteCounts_comm_atCertainRank{certainRank}.dict'
    if os.path.exists(resultFileDir):
        print(f"{subCommName} has already done this step.")
        return
    
    print(f"sub comm {subCommName} of {parentCommName} getValuesForConformityComputation running on {mp.current_process().name}")

    # load intermediate_data files
    try:
        with open(subComm_QuestionsWithEventList_directory, 'rb') as inputFile:
            sub_Questions = pickle.load( inputFile)
            qCount = len(sub_Questions)
            print(f"subComm {subCommName} QuestionsWithEventList loaded, length {qCount}.")
    except:
        exceptionLog = f"{subCommName} hasn't been created yet, no EventList data!"
        return exceptionLog


    # prepare args
    args = zip(list(sub_Questions.items()), [parentCommName]*qCount)
    
    # process Questions chunk by chunk
    # process Questions chunk by chunk
    n_proc = mp.cpu_count()
    with mp.Pool(processes=n_proc) as pool:
        # issue tasks to the process pool and wait for tasks to complete
        all_outputs = pool.starmap(descriptive3_getValuesForConformityComputation.myAction, args , chunksize=1)
    
    sub_Questions.clear()

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
    
    voteDiff2voteCounts_comm_atCertainRank = defaultdict()

    year = total_valuesForConformityComputation_forEachVote[0][5]

    startIndex = 0
    for i, tup in enumerate(total_valuesForConformityComputation_forEachVote):
        print(f"processing {i+1}th/{totalVoteCount} vote of {subCommName}...")
        cur_vote, cur_n_pos, cur_n_neg,rank, t, cur_year, qid, ans_index= tup
        cur_vote_diff = cur_n_pos - cur_n_neg
        
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
            with open(subComm_yearlyValuesForConformityComputation_folder+'/'+f'valuesForConformityComputationOnlyForYear_{year}.dict', 'wb') as outputFile:
                pickle.dump(valuesForConformityComputation_forEachVote_onlyForCurYear, outputFile) 
                logtext = f"saved valuesForConformityComputation_forCurYear for year {year} of {subCommName}, length: {len(valuesForConformityComputation_forEachVote_onlyForCurYear)}.\n"
                writeIntoLog(logtext, parentCommDir, logFileName)
                print(logtext) 

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
    with open(subComm_yearlyValuesForConformityComputation_folder+'/'+f'valuesForConformityComputationOnlyForYear_{year}.dict', 'wb') as outputFile:
        pickle.dump(valuesForConformityComputation_forEachVote_onlyForCurYear, outputFile) 
        logtext = f"saved valuesForConformityComputation_forCurYear for year {year} of {subCommName}, length: {len(valuesForConformityComputation_forEachVote_onlyForCurYear)}.\n"
        writeIntoLog(logtext, parentCommDir, logFileName)
        print(logtext) 
    
    # save voteDiff2voteCounts_comm_atCertainRank
    with open(subCommDir+'/'+f'voteDiff2voteCounts_comm_atCertainRank{certainRank}.dict', 'wb') as outputFile:
        pickle.dump(voteDiff2voteCounts_comm_atCertainRank, outputFile) 
        logtext = f"saved voteDiff2voteCounts_comm_atCertainRank of {subCommName}.\n"
        writeIntoLog(logtext, parentCommDir, logFileName)
        print(logtext)

def myAction (commName, year,valuesForConformityComputation):
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

def Conformity_myFun(parentCommName, parentCommDir, subCommName, subCommDir,subComm_yearlyValuesForConformityComputation_folder, logFileName):
    print(f"sub comm {subCommName} of {parentCommName} Conformity_myFun running on {mp.current_process().name}")
    
    # load earlyValuesForConformityComputation
    yearFiles = [ f.path for f in os.scandir(subComm_yearlyValuesForConformityComputation_folder) if f.path.endswith('.dict') ]
    # sort files based on year
    yearFiles.sort(key=lambda p: int(p.strip(".dict").split("_")[-1]))
    yearsCount = len(yearFiles)
                                
    print(f"there are {yearsCount} splitted year files in {subCommName}")

    print(f"start to compute Conformity for sub comm...")
    # compute Conformity
    year2ConformitySize= defaultdict()
    logSum = 0
    totalVoteCount = 0
    certainRank = 6 # an arbitrarily choosen rank for conformity verifying
    # create the table for certain rank
    voteDiff2voteCounts = defaultdict() # a dict to keep real time vote difference to vote counts of pos and neg (as #vote+_{vd^t} in conformity equation)
    
    bugVoteCount = 0

    for subDir in yearFiles:
        year = int(subDir.strip(".dict").split("_")[-1])
        # get valuesForConformtiyComputation of each year
        with open(subDir, 'rb') as inputFile:
            valuesForConformtiyComputation = pickle.load( inputFile)
            print(f"year {year} of {subCommName} is loaded.")

        year, cur_logSum, cur_voteCount, bugVoteCount_curYear = myAction(subCommName, year,valuesForConformtiyComputation)
        
        if bugVoteCount_curYear >0:
            logtext = f"there are {bugVoteCount_curYear} bug votes in year {year} for sub comm {subCommName} of {parentCommName}.\n"
            writeIntoLog(logtext, parentCommDir, logFileName)
            print(logtext)
        bugVoteCount += bugVoteCount_curYear

        totalVoteCount += cur_voteCount
        logSum += cur_logSum
        if totalVoteCount !=0:
            Conformity = math.exp(logSum/totalVoteCount)
            
            logtext = f" year {year} logSum:{logSum}, Conformity: {Conformity}, size: {totalVoteCount}.\n"
            writeIntoLog(logtext, parentCommDir, logFileName)
            print(logtext)
            
            year2ConformitySize[year] = {'conformtiy':Conformity,'size':totalVoteCount}
            
        # plot Conformity verifying 
        if year == 2022:  
            # load voteDiff2voteCounts
            certainRank = 6
            try:
                with open(subCommDir+'/'+f'voteDiff2voteCounts_comm_atCertainRank{certainRank}.dict', 'rb') as inputFile:
                    voteDiff2voteCounts = pickle.load( inputFile)
                    print(f"loaded voteDiff2voteCounts till 2022 for {subCommName}.")  
                    descriptive4_Conformity_forStackOverflow.plotVoteProportionAtEachVoteDiffForCertainRank(subCommName,subCommDir,year, voteDiff2voteCounts,logFileName, certainRank)
            except Exception as e:
                print(e)
                writeIntoLog(f"fail to plot:{e}", parentCommDir, logFileName)
            
    return year2ConformitySize

def mySteps (parentCommName, parentCommDir, subComms_data_folder, subCommName, subCommDir, year2UniversalTimeStepIndex, subComm_QuestionsWithEventList_directory, subComm_yearlyVoteCountAtEachRank_directory,subComm_yearlyValuesForConformityComputation_folder, logFileName, return_dict):
    
    # Trendiness for sub Comm
    exceptionLog = computeVoteCountAtEachRank_myFunc(parentCommName, parentCommDir, subCommName, year2UniversalTimeStepIndex, subComm_QuestionsWithEventList_directory, subComm_yearlyVoteCountAtEachRank_directory, logFileName)
    if exceptionLog != None:
        return
    
    year2fittingResults = Trendiness_myFun(parentCommName, parentCommDir, subCommName,subComm_yearlyVoteCountAtEachRank_directory, logFileName)
    
    # Conformity for sub Comm
    getValuesForConformityComputation_myFun(parentCommName, parentCommDir, subCommName, subCommDir, subComm_QuestionsWithEventList_directory,subComm_yearlyValuesForConformityComputation_folder, logFileName)
    
    year2ConformitySize = Conformity_myFun(parentCommName, parentCommDir, subCommName, subCommDir, subComm_yearlyValuesForConformityComputation_folder, logFileName)

    return_dict[subCommName] = {'TrendinessFittingResults': year2fittingResults,
                                'ConformitySizeResults': year2ConformitySize}

    # save results so far
    return_normalDict = defaultdict()
    for subCommName, d in return_dict.items():
        return_normalDict[subCommName] = d

    with open(subComms_data_folder+'/'+f'subCommsTrendinessConformitySize.dict', 'wb') as outputFile:
        pickle.dump(return_normalDict, outputFile)

def main():

    t0=time.time()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
        print("other sorted CommDir loaded.")
    logFileName = 'descriptive8_computeTrendinessConformityForSubCommsOfStackOverflow_log.txt'

    # go to StackOverflow data directory
    parentCommName = commDir_sizes_sortedlist[359][0]
    parentCommDir = commDir_sizes_sortedlist[359][1]
    os.chdir(parentCommDir)
    print(os.getcwd())

    # go to the target splitted files folder
    intermediate_directory = os.path.join(parentCommDir, r'intermediate_data_folder')

    with open(intermediate_directory+'/'+'year2UniversalTimeStepIndex.dict', 'rb') as inputFile:
        year2UniversalTimeStepIndex = pickle.load( inputFile)

    subComms_data_folder = os.path.join(parentCommDir, f'subCommunities_folder')
    if not os.path.exists( subComms_data_folder):
        print("Exception: no subComms_data_folder!")
    
    ## Load all sub community direcotries 
    with open(subComms_data_folder+'/'+f'subCommName2subCommDir.dict', 'rb') as inputFile:
        subCommName2commDir = pickle.load( inputFile)

    # multiprocessing
    # use shared variable to communicate among all sub comm's process
    manager = mp.Manager()
    return_dict = manager.dict() # to save final results of each sub community
    
    # prepare args 
    argsList = []
    for subCommName, subCommDir in subCommName2commDir.items():

        # subComm_QuestionsWithEventList
        subComm_QuestionsWithEventList_directory = subCommDir+'/'+f'QuestionsWithEventList_tag_{subCommName}.dict'

        # under splitted_intermediate_data_folder, create a folder to store intermediate trendiness results of sub communities 
        subComm_yearlyVoteCountAtEachRank_directory = subCommDir + f'/subComm_yearlyVoteCountAtEachRank_tag_{subCommName}.dict'
        
        # create a folder to store intermediate conformity results of sub communities 
        subComm_yearlyValuesForConformityComputation_folder = os.path.join(subCommDir, r'yearlyValuesForConformityComputation')
        if not os.path.exists(subComm_yearlyValuesForConformityComputation_folder):
            print("no subComm_yearlyValuesForConformityComputation_folder, create one")
            os.makedirs(subComm_yearlyValuesForConformityComputation_folder)
    
        # prepare args
        argsList.append((parentCommName, parentCommDir, subComms_data_folder, subCommName, subCommDir, year2UniversalTimeStepIndex, subComm_QuestionsWithEventList_directory, subComm_yearlyVoteCountAtEachRank_directory,subComm_yearlyValuesForConformityComputation_folder, logFileName, return_dict))
    
    
    # run on all sub communities of stackoverflow
    finishedCount = 0
    processes = []
    for args in argsList:
        try:
            p = mp.Process(target=mySteps, args=args)
            p.start()
        except Exception as e:
            print(e)
            pscount = sum(1 for proc in psutil.process_iter() if proc.name() == 'python3')
            print(f"current python3 processes count {pscount}.")
            sys.exit()
            return

        processes.append(p)
        if len(processes)==5:
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
    
    # save all
    return_normalDict = defaultdict()
    for subCommName, d in return_dict.items():
        return_normalDict[subCommName] = d

    with open(subComms_data_folder+'/'+f'subCommsTrendinessConformitySize.dict', 'wb') as outputFile:
        pickle.dump(return_normalDict, outputFile)
        logtext = f"saved commName2TrendinessConformitySize length:{len(return_normalDict)}.\n"
        writeIntoLog(logtext, parentCommDir, logFileName)
        print(logtext)
            
    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('compute Trendiness and Conformity for sub comms Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
