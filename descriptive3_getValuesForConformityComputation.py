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

def myFun(commName, commDir):
    print(f"comm {commName} running on {mp.current_process().name}")
    logFileName = 'descriptive3_otherComm_log.txt'

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data files
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')
    
    with open(intermediate_directory+'/'+'QuestionsWithAnswersWithVotesWithPostHistoryWithMatrixWithEventList.dict', 'rb') as inputFile:
        Questions = pickle.load( inputFile)

    with open(intermediate_directory+'/'+'year2UniversalTimeStepIndex.dict', 'rb') as inputFile:
        year2UniversalTimeStepIndex = pickle.load( inputFile)
    minYear = min(list(year2UniversalTimeStepIndex.keys()))

    # mkdir to keep splitted_intermediate_data
    splitted_intermediate_data_folder = os.path.join(intermediate_directory, r'splitted_intermediate_data_folder')
    if not os.path.exists(splitted_intermediate_data_folder): # don't have dataset splitted intermediate data folder, skip
        print(f"{commName} has no  splitted_intermediate_data_folder, create one.")
        os.makedirs(splitted_intermediate_data_folder)
    # mkdir to keep total valuesForConformityComputation list for each year
    split_valuesForConformtiyComputation_forEachYear_directory = os.path.join(splitted_intermediate_data_folder, r'totalValuesForConformtiyComputation_forEachYear_parts_folder')
    if not os.path.exists(split_valuesForConformtiyComputation_forEachYear_directory): # don't have dataset splitted intermediate data folder, skip
        print(f"{commName} has no  split_valuesForConformtiyComputation_forEachYear_directory, create one.")
        os.makedirs(split_valuesForConformtiyComputation_forEachYear_directory)

    # extract only useful info from each Question
    # process Questions chunk by chunk
    n_proc = mp.cpu_count() # left 2 cores to do others
    with mp.Pool(processes=n_proc) as pool:
        args = zip(list(Questions.items()), len(Questions)*[commName])
        # issue tasks to the process pool and wait for tasks to complete
        all_outputs = pool.starmap(myAction, args , chunksize=10)
    
    Questions.clear()
    
    # combine all_outputs into a whole list
    total_valuesForConformityComputation_forEachVote = []
    for ll in all_outputs:
        if len(ll)>0:
            total_valuesForConformityComputation_forEachVote.extend(ll)
    
    all_outputs.clear()

    # sort each vote tuple by time step index
    total_valuesForConformityComputation_forEachVote.sort(key=lambda t: t[4])
    totalVoteCount = len(total_valuesForConformityComputation_forEachVote)

    # save original total list
    with open(intermediate_directory+'/'+f'total_valuesForConformityComputation_forEachVote_original.dict', 'wb') as outputFile:
        pickle.dump(total_valuesForConformityComputation_forEachVote, outputFile) 
        logtext = f"saved total_valuesForConformityComputation_forEachVote of SOF, length {totalVoteCount}.\n"
        writeIntoLog(logtext, commDir, logFileName)
        print(logtext)

    """
    # update values for conformity computation along with real time voteDiff2voteCounts_comm table
    voteDiff2voteCounts_comm = defaultdict()
    certainRank = 6
    voteDiff2voteCounts_comm_atCertainRank = defaultdict()
    valuesTupleAtCertainRank = []

    year = whichYear(total_valuesForConformityComputation_forEachVote[0][4], year2UniversalTimeStepIndex)

    startIndex = 0
    for i, tup in enumerate(total_valuesForConformityComputation_forEachVote):
        print(f"processing {i+1}th/{totalVoteCount} vote of {commName}...")
        cur_vote, cur_n_pos, cur_n_neg,rank, t, qid, ans_index = tup
        cur_vote_diff = cur_n_pos - cur_n_neg

        cur_year = whichYear(t, year2UniversalTimeStepIndex)
        
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
            with open(split_valuesForConformtiyComputation_forEachYear_directory+'/'+f'valuesForConformityComputationOnlyForYear_{year}.dict', 'wb') as outputFile:
                pickle.dump(valuesForConformityComputation_forEachVote_onlyForCurYear, outputFile) 
                logtext = f"saved valuesForConformityComputation_forCurYear for year {year} of {commName}, length: {len(valuesForConformityComputation_forEachVote_onlyForCurYear)}.\n"
                writeIntoLog(logtext, commDir, logFileName)
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
            
            valuesTupleAtCertainRank.append(tup)
        
    # save for the last year
    valuesForConformityComputation_forEachVote_onlyForCurYear = total_valuesForConformityComputation_forEachVote[startIndex:]
    with open(split_valuesForConformtiyComputation_forEachYear_directory+'/'+f'valuesForConformityComputationOnlyForYear_{year}.dict', 'wb') as outputFile:
        pickle.dump(valuesForConformityComputation_forEachVote_onlyForCurYear, outputFile) 
        logtext = f"saved valuesForConformityComputation_forCurYear for year {year} of {commName}, length: {len(valuesForConformityComputation_forEachVote_onlyForCurYear)}.\n"
        writeIntoLog(logtext, commDir, logFileName)
        print(logtext)
    
    # save voteDiff2voteCounts_comm_atCertainRank
    with open(intermediate_directory+'/'+f'voteDiff2voteCounts_comm_atCertainRank{certainRank}.dict', 'wb') as outputFile:
        pickle.dump(voteDiff2voteCounts_comm_atCertainRank, outputFile) 
        logtext = f"saved voteDiff2voteCounts_comm_atCertainRank of {commName}.\n"
        writeIntoLog(logtext, commDir, logFileName)
        print(logtext)
    
    # save valuesTupleAtCertainRank
    with open(intermediate_directory+'/'+f'valuesTupleAtCertainRank{certainRank}.dict', 'wb') as outputFile:
        pickle.dump(valuesTupleAtCertainRank, outputFile) 
        logtext = f"saved vvaluesTupleAtCertainRank of SOF, length {len(valuesTupleAtCertainRank)}.\n"
        writeIntoLog(logtext, commDir, logFileName)
        print(logtext)
    """

def combine_all_outputs_SOF(commName, commDir, split_valuesForConformtiyComputation_forEachQuestion_directory, split_valuesForConformtiyComputation_forEachYear_directory, logFileName, minYear, year2UniversalTimeStepIndex,intermediate_directory):

    partFiles = [ f.path for f in os.scandir(split_valuesForConformtiyComputation_forEachQuestion_directory) if f.path.endswith('.dict') ]
    # sort csvFiles paths based on part number
    partFiles.sort(key=lambda p: int(p.strip(".dict").split("_")[-1]))
    partsCount = len(partFiles)

    assert partsCount == int(partFiles[-1].strip(".dict").split("_")[-1]) # last part file's part number should equal to the parts count
                                
    print(f"there are {partsCount} splitted values for conformity compution all_outputs files in SOF")

    # combine all_outputs into a whole list
    total_valuesForConformityComputation_forEachVote = []

    for i, subDir in enumerate(partFiles):
        part = i+1
        partDir = subDir
        # get question count of each part
        with open(partDir, 'rb') as inputFile:
            all_outputs = pickle.load( inputFile)
            print(f"part {part} of SOF is loaded.")

            qCount = len(all_outputs)
            for i,ll in enumerate(all_outputs): 
                if len(ll)>0:
                    total_valuesForConformityComputation_forEachVote.extend(ll)
                    print(f"combined part {part}, {i+1}th/{qCount} quetion's list to total list.")
    
    all_outputs.clear()

    # sort each vote tuple by time step index
    total_valuesForConformityComputation_forEachVote.sort(key=lambda t: t[4])
    totalVoteCount = len(total_valuesForConformityComputation_forEachVote)

    
    # save original total list
    with open(intermediate_directory+'/'+f'total_valuesForConformityComputation_forEachVote_original.dict', 'wb') as outputFile:
        pickle.dump(total_valuesForConformityComputation_forEachVote, outputFile) 
        logtext = f"saved total_valuesForConformityComputation_forEachVote of SOF, length {totalVoteCount}.\n"
        writeIntoLog(logtext, commDir, logFileName)
        print(logtext)
    """
    # update values for conformity computation along with real time voteDiff2voteCounts_comm table
    voteDiff2voteCounts_comm = defaultdict()
    certainRank = 6
    voteDiff2voteCounts_comm_atCertainRank = defaultdict()
    valuesTupleAtCertainRank = []

    year = total_valuesForConformityComputation_forEachVote[0][5]

    startIndex = 0
    for i, tup in enumerate(total_valuesForConformityComputation_forEachVote):
        print(f"processing {i+1}th/{totalVoteCount} vote of {commName}...")
        cur_vote, cur_n_pos, cur_n_neg,rank, t, cur_year, qid, ans_index = tup
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
            with open(split_valuesForConformtiyComputation_forEachYear_directory+'/'+f'valuesForConformityComputationOnlyForYear_{year}.dict', 'wb') as outputFile:
                pickle.dump(valuesForConformityComputation_forEachVote_onlyForCurYear, outputFile) 
                logtext = f"saved valuesForConformityComputation_forCurYear for year {year} of {commName}, length: {len(valuesForConformityComputation_forEachVote_onlyForCurYear)}.\n"
                writeIntoLog(logtext, commDir, logFileName)
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
            
            valuesTupleAtCertainRank.append(tup)
                
            
    # save valuesForConformityComputation list for last year
    valuesForConformityComputation_forEachVote_onlyForCurYear = total_valuesForConformityComputation_forEachVote[startIndex:]
    with open(split_valuesForConformtiyComputation_forEachYear_directory+'/'+f'valuesForConformityComputationOnlyForYear_{year}.dict', 'wb') as outputFile:
        pickle.dump(valuesForConformityComputation_forEachVote_onlyForCurYear, outputFile) 
        logtext = f"saved valuesForConformityComputation_forEachVote_onlyForCurYear for year {year} of SOF, length: {len(valuesForConformityComputation_forEachVote_onlyForCurYear)}.\n"
        writeIntoLog(logtext, commDir, logFileName)
        print(logtext)

    # save voteDiff2voteCounts_comm_atCertainRank
    with open(intermediate_directory+'/'+f'voteDiff2voteCounts_comm_atCertainRank{certainRank}.dict', 'wb') as outputFile:
        pickle.dump(voteDiff2voteCounts_comm_atCertainRank, outputFile) 
        logtext = f"saved voteDiff2voteCounts_comm_atCertainRank of SOF.\n"
        writeIntoLog(logtext, commDir, logFileName)
        print(logtext)
    
    # save valuesTupleAtCertainRank
    with open(intermediate_directory+'/'+f'valuesTupleAtCertainRank{certainRank}.dict', 'wb') as outputFile:
        pickle.dump(valuesTupleAtCertainRank, outputFile) 
        logtext = f"saved valuesTupleAtCertainRank of SOF, length {len(valuesTupleAtCertainRank)}.\n"
        writeIntoLog(logtext, commDir, logFileName)
        print(logtext)
    """


def myFun_SOF(commName, commDir):
    print(f"comm {commName} running on {mp.current_process().name}")
    logFileName = 'descriptive3_SOF_log.txt'

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # go to the target splitted files folder
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')

    with open(intermediate_directory+'/'+'year2UniversalTimeStepIndex.dict', 'rb') as inputFile:
        year2UniversalTimeStepIndex = pickle.load( inputFile)

    minYear = min(list(year2UniversalTimeStepIndex.keys()))

    splitted_intermediate_data_folder = os.path.join(intermediate_directory, r'splitted_intermediate_data_folder')
    split_QuestionsWithEventList_files_directory = os.path.join(splitted_intermediate_data_folder, r'QuestionsPartsWithEventList')
    if not os.path.exists(split_QuestionsWithEventList_files_directory): # didn't find the parts files
        print("Exception: no split_QuestionsWithEventList_files_directory!")

    partFiles = [ f.path for f in os.scandir(split_QuestionsWithEventList_files_directory) if f.path.endswith('.dict') ]
    # sort csvFiles paths based on part number
    partFiles.sort(key=lambda p: int(p.strip(".dict").split("_")[-1]))
    partsCount = len(partFiles)

    assert partsCount == int(partFiles[-1].strip(".dict").split("_")[-1]) # last part file's part number should equal to the parts count
                                
    print(f"there are {partsCount} splitted event list files in {commName}")

    # mkdir to keep all_outputs for each part
    split_valuesForConformtiyComputation_forEachQuestion_directory = os.path.join(splitted_intermediate_data_folder, r'valuesForConformtiyComputation_forEachQuestion_parts_folder')
    if not os.path.exists(split_valuesForConformtiyComputation_forEachQuestion_directory): # don't have dataset splitted intermediate data folder, skip
        print(f"{commName} has no  split_valuesForConformtiyComputation_forEachQuestion_directory folder, create one.")
        os.makedirs(split_valuesForConformtiyComputation_forEachQuestion_directory)

    # mkdir to keep total valuesForConformityComputation list for each year
    split_valuesForConformtiyComputation_forEachYear_directory = os.path.join(splitted_intermediate_data_folder, r'totalValuesForConformtiyComputation_forEachYear_parts_folder')
    if not os.path.exists(split_valuesForConformtiyComputation_forEachYear_directory): # don't have dataset splitted intermediate data folder, skip
        print(f"{commName} has no  split_valuesForConformtiyComputation_forEachYear_directory, create one.")
        os.makedirs(split_valuesForConformtiyComputation_forEachYear_directory)

    """
    for i, subDir in enumerate(partFiles):
        part = i+1
        partDir = subDir
        # get question count of each part
        with open(partDir, 'rb') as inputFile:
            Questions_part = pickle.load( inputFile)
            print(f"part {part} of {commName} is loaded.")
        
        # process Questions chunk by chunk
        n_proc = mp.cpu_count() # left 2 cores to do others
        with mp.Pool(processes=n_proc) as pool:
            args = zip(list(Questions_part.items()), len(Questions_part)*[commName])
            # issue tasks to the process pool and wait for tasks to complete
            all_outputs = pool.starmap(myAction, args , chunksize=10)
        
        print(f"part {part} of stackoverflow result all_outputs, length {len(all_outputs)}.")

        # save all_outputs for current part
        with open(split_valuesForConformtiyComputation_forEachQuestion_directory+'/'+f'descriptive3_SOF_all_outputs_part_{part}.dict', 'wb') as outputFile:
            pickle.dump(all_outputs, outputFile) 
            print(f"saved descriptive3_SOF_all_outputs_part_{part} for {commName}.")
            writeIntoLog(f"saved SOF all_outputs part {part}, length :{len(all_outputs)}.\n", commDir, logFileName)
    all_outputs.clear()
    """
    
    # combine results
    print(f"start to combine yearly valuesForConformityComputation list from each question...for {commName}")
    combine_all_outputs_SOF(commName, commDir, split_valuesForConformtiyComputation_forEachQuestion_directory, split_valuesForConformtiyComputation_forEachYear_directory, logFileName, minYear, year2UniversalTimeStepIndex, intermediate_directory)

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
    for tup in commDir_sizes_sortedlist[300:]:
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
    myFun_SOF('stackoverflow', stackoverflow_dir)
    
    
            
    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('descriptive3_get values fro conformity computation Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
