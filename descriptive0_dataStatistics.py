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


def myAction (question_tuple, commName, year2UniversalTimeStepIndex):
    qid = question_tuple[0]
    content = question_tuple[1]
    print(f"processing question {qid} of {commName} on {mp.current_process().name}...")
    eventList = content['eventList']
    lts = content['local_timesteps']

    # compute vote count at each rank for each year
    year2answerCountAndVoteCount = defaultdict()

    if commName == 'stackoverflow':
        year = lts[0][2].year
    else:
        year = whichYear(lts[0], year2UniversalTimeStepIndex)

    firstVoteRemovedFlagForEachAnswer =  []

    voteCount = 0
    answerCount = 0
    for i,e in enumerate(eventList):
        t = lts[i]
        eventType = e['et']
        if eventType not in ['v','w']: # skip all event that is not a vote
            continue

        if eventType == 'w': # write a new answer
            answerCount += 1
            continue
        
        # vote event
        vote = e['v']
        
        if e['ai'] in firstVoteRemovedFlagForEachAnswer: # current vote's target answer already has the first vote removed
            voteCount +=1

            if commName == 'stackoverflow':
                cur_year = t[2].year
            else:
                cur_year = whichYear(t, year2UniversalTimeStepIndex)
            
            if cur_year == year:
                continue
            elif cur_year < year:
                print(f"Exception: cur_year < year for {i+1}th time step")
            else: # cur_year > year
                # save previous year's last universal time steps index
                year2answerCountAndVoteCount[year] = {'answerCount':answerCount, 'voteCount':voteCount}
                # start a new year
                year = cur_year
                
        else:# current vote is its target answer's first vote, don't use as sample
            firstVoteRemovedFlagForEachAnswer.append(e['ai'])

    # save the last year's 
    year2answerCountAndVoteCount[year] = {'answerCount':answerCount, 'voteCount':voteCount}

    # sort year2answerCountAndVoteCount
    year2answerCountAndVoteCount = dict(sorted(year2answerCountAndVoteCount.items()))

    # fill up unvoted year
    minYear = min(list(year2answerCountAndVoteCount.keys()))
    for year in range(minYear, 2023):
        if year not in year2answerCountAndVoteCount.keys():
            if year-1 not in year2answerCountAndVoteCount.keys():
                year2answerCountAndVoteCount[year] = {'answerCount':0, 'voteCount':0}
            else:
                year2answerCountAndVoteCount[year] = copy.deepcopy(year2answerCountAndVoteCount[year-1])
    
    # sort year2answerCountAndVoteCount
    year2answerCountAndVoteCount = dict(sorted(year2answerCountAndVoteCount.items()))

    # return processed question's qid, content, the number of answers with votes, the number of real votes as samples
    return (qid, year2answerCountAndVoteCount)
    

def myFun(commName, commDir, rootDir):
    print(f"comm {commName} running on {mp.current_process().name}")

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data files
    current_directory = os.getcwd()
    intermediate_directory = os.path.join(current_directory, r'intermediate_data_folder')
    
    with open(intermediate_directory+'/'+'QuestionsWithAnswersWithVotesWithPostHistoryWithMatrixWithEventList.dict', 'rb') as inputFile:
        Questions = pickle.load( inputFile)

    with open(intermediate_directory+'/'+'year2UniversalTimeStepIndex.dict', 'rb') as inputFile:
        year2UniversalTimeStepIndex = pickle.load( inputFile)
    
    # process Questions chunk by chunk
    n_proc = mp.cpu_count()-2 # left 2 cores to do others
    all_outputs = []
    with mp.Pool(processes=n_proc) as pool:
        args = zip(list(Questions.items()), len(Questions)*[commName], len(Questions)*[year2UniversalTimeStepIndex])
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
    
    Questions.clear()
    results.clear()
    # combine results
    print(f"start to combine year2voteCountAtEachYear to year2totalCount...for {commName}")
    if len(all_outputs) == 0:
        print(f"no output for {commName}")
        return
    year2totalCount = defaultdict()
    for tup in all_outputs: 
        qid = tup[0]
        print(f"combining question {qid} results to total for {commName}")
        year2answerCountAndVoteCount = tup[1]

        minYear = min(list(year2answerCountAndVoteCount.keys()))

        if minYear not in year2totalCount.keys():
            year2totalCount[minYear] = {'questionCount':1, 'answerCount':year2answerCountAndVoteCount[minYear]['answerCount'], 'voteCount':year2answerCountAndVoteCount[minYear]['voteCount']}

        else:
            year2totalCount[minYear]['questionCount'] += 1
            year2totalCount[minYear]['answerCount'] += year2answerCountAndVoteCount[minYear]['answerCount']
            year2totalCount[minYear]['voteCount'] += year2answerCountAndVoteCount[minYear]['voteCount']

        for year, d in year2answerCountAndVoteCount.items():
            if year == minYear:
                continue
            if year not in year2totalCount.keys():
                year2totalCount[year] = {'questionCount':1, 'answerCount':d['answerCount'], 'voteCount':d['voteCount']}
            else:
                year2totalCount[year]['questionCount'] += 1
                year2totalCount[year]['answerCount'] += d['answerCount']
                year2totalCount[year]['voteCount'] += d['voteCount']
    # sort
    year2totalCount = dict(sorted(year2totalCount.items()))
                
    # save total statistics
    with open(intermediate_directory+'/'+'yearlyStatistics.dict', 'wb') as outputFile:
        pickle.dump(year2totalCount, outputFile) 
        print(f"saved year2totalCount for {commName}.")  

    # write into descriptive_allComm_statistics.csv
    with open(rootDir+'/'+f'descriptive_allComm_statistics.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for year, d in year2totalCount.items():
            writer.writerow( [commName, year, d['questionCount'], d['answerCount'], d['voteCount']])
        print(f"saved year2totalCount for {commName} in descriptive_allComm_statistics.csv.")

def myFun_SOF(commName, commDir, rootDir):
    print(f"comm {commName} running on {mp.current_process().name}")

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # go to the target splitted files folder
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')

    with open(intermediate_directory+'/'+'year2UniversalTimeStepIndex.dict', 'rb') as inputFile:
        year2UniversalTimeStepIndex = pickle.load( inputFile)

    splitFolder_directory = os.path.join(intermediate_directory, r'splitted_intermediate_data_folder')
    split_QuestionsWithEventList_files_directory = os.path.join(splitFolder_directory, r'QuestionsPartsWithEventList')
    if not os.path.exists(split_QuestionsWithEventList_files_directory): # didn't find the parts files
        print("Exception: no split_QuestionsWithEventList_files_directory!")

    partFiles = [ f.path for f in os.scandir(split_QuestionsWithEventList_files_directory) if f.path.endswith('.dict') ]
    # sort csvFiles paths based on part number
    partFiles.sort(key=lambda p: int(p.strip(".dict").split("_")[-1]))
    partsCount = len(partFiles)

    assert partsCount == int(partFiles[-1].strip(".dict").split("_")[-1]) # last part file's part number should equal to the parts count
                                
    print(f"there are {partsCount} splitted event list files in {commName}")

    all_outputs = []
    for i, subDir in enumerate(partFiles):
        part = i+1
        partDir = subDir
        # get question count of each part
        with open(partDir, 'rb') as inputFile:
            Questions_part = pickle.load( inputFile)
            print(f"part {part} of {commName} is loaded.")
        
        # process Questions chunk by chunk
        n_proc = mp.cpu_count()-2 # left 2 cores to do others
        with mp.Pool(processes=n_proc) as pool:
            args = zip(list(Questions_part.items()), len(Questions_part)*[commName], len(Questions_part)*[year2UniversalTimeStepIndex])
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
        
        print(f"part {part} of stackoverflow result added to all_outputs, current total length {len(all_outputs)}.")
        Questions_part.clear()
        results.clear()
    
    # combine results
    print(f"start to combine year2voteCountAtEachYear to year2totalCount...for {commName}")
    if len(all_outputs) == 0:
        print(f"no output for {commName}")
        return
    year2totalCount = defaultdict()
    for tup in all_outputs: 
        qid = tup[0]
        print(f"combining question {qid} results to total for {commName}")
        year2answerCountAndVoteCount = tup[1]

        minYear = min(list(year2answerCountAndVoteCount.keys()))

        if minYear not in year2totalCount.keys():
            year2totalCount[minYear] = {'questionCount':1, 'answerCount':year2answerCountAndVoteCount[minYear]['answerCount'], 'voteCount':year2answerCountAndVoteCount[minYear]['voteCount']}

        else:
            year2totalCount[minYear]['questionCount'] += 1
            year2totalCount[minYear]['answerCount'] += year2answerCountAndVoteCount[minYear]['answerCount']
            year2totalCount[minYear]['voteCount'] += year2answerCountAndVoteCount[minYear]['voteCount']

        for year, d in year2answerCountAndVoteCount.items():
            if year == minYear:
                continue
            if year not in year2totalCount.keys():
                year2totalCount[year] = {'questionCount':1, 'answerCount':d['answerCount'], 'voteCount':d['voteCount']}
            else:
                year2totalCount[year]['questionCount'] += 1
                year2totalCount[year]['answerCount'] += d['answerCount']
                year2totalCount[year]['voteCount'] += d['voteCount']
    # sort
    year2totalCount = dict(sorted(year2totalCount.items()))
                
    # save total statistics
    with open(intermediate_directory+'/'+'yearlyStatistics.dict', 'wb') as outputFile:
        pickle.dump(year2totalCount, outputFile) 
        print(f"saved year2totalCount for {commName}.")  

    # write into descriptive_allComm_statistics.csv
    with open(rootDir+'/'+f'descriptive_allComm_statistics.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for year, d in year2totalCount.items():
            writer.writerow( [commName, year, d['questionCount'], d['answerCount'], d['voteCount']])
        print(f"saved year2totalCount for {commName} in descriptive_allComm_statistics.csv.")

def main():

    t0=time.time()
    rootDir = os.getcwd()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("other sorted CommDir loaded.")

    with open(rootDir+'/'+f'descriptive_allComm_statistics.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow( ["commName","till Year","question Count", "answer count", "vote count"])
    

    # # test on comm "coffee.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[166][0], commDir_sizes_sortedlist[166][1])
    # # test on comm "datascience.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[301][0], commDir_sizes_sortedlist[301][1])
    # test on comm "webapps.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[305][0], commDir_sizes_sortedlist[305][1])

    # test on comm "stackoverflow" to debug
    # myFun_SOF(commDir_sizes_sortedlist[359][0], commDir_sizes_sortedlist[359][1])

    
    
    # # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for tup in commDir_sizes_sortedlist:
        commName = tup[0]
        commDir = tup[1]
        if commName != 'stackoverflow': # skip stackoverflow 
            try:
                p = mp.Process(target=myFun, args=(commName,commDir, rootDir))
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
    myFun_SOF('stackoverflow', stackoverflow_dir, rootDir)
    
            
    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('descriptive1 compute vote count at each rank Done completely.    Elapsed: {:}.\n'.format(elapsed))

def myAction_subComm (question_tuple, commName, year2UniversalTimeStepIndex):
    qid = question_tuple[0]
    content = question_tuple[1]
    print(f"processing question {qid} of {commName} on {mp.current_process().name}...")
    eventList = content['eventList']
    lts = content['local_timesteps']

    # compute vote count at each rank for each year
    year2answerCountAndVoteCount = defaultdict()

    year = lts[0][2].year

    firstVoteRemovedFlagForEachAnswer =  []

    voteCount = 0
    answerCount = 0
    for i,e in enumerate(eventList):
        t = lts[i]
        eventType = e['et']
        if eventType not in ['v','w']: # skip all event that is not a vote
            continue

        if eventType == 'w': # write a new answer
            answerCount += 1
            continue
        
        # vote event
        vote = e['v']
        
        if e['ai'] in firstVoteRemovedFlagForEachAnswer: # current vote's target answer already has the first vote removed
            voteCount +=1

            cur_year = t[2].year
            
            if cur_year == year:
                continue
            elif cur_year < year:
                print(f"Exception: cur_year < year for {i+1}th time step")
            else: # cur_year > year
                # save previous year's last universal time steps index
                year2answerCountAndVoteCount[year] = {'answerCount':answerCount, 'voteCount':voteCount}
                # start a new year
                year = cur_year
                
        else:# current vote is its target answer's first vote, don't use as sample
            firstVoteRemovedFlagForEachAnswer.append(e['ai'])

    # save the last year's 
    year2answerCountAndVoteCount[year] = {'answerCount':answerCount, 'voteCount':voteCount}

    # sort year2answerCountAndVoteCount
    year2answerCountAndVoteCount = dict(sorted(year2answerCountAndVoteCount.items()))

    # fill up unvoted year
    minYear = min(list(year2answerCountAndVoteCount.keys()))
    for year in range(minYear, 2023):
        if year not in year2answerCountAndVoteCount.keys():
            if year-1 not in year2answerCountAndVoteCount.keys():
                year2answerCountAndVoteCount[year] = {'answerCount':0, 'voteCount':0}
            else:
                year2answerCountAndVoteCount[year] = copy.deepcopy(year2answerCountAndVoteCount[year-1])
    
    # sort year2answerCountAndVoteCount
    year2answerCountAndVoteCount = dict(sorted(year2answerCountAndVoteCount.items()))

    # return processed question's qid, content, the number of answers with votes, the number of real votes as samples
    return (qid, year2answerCountAndVoteCount)

def myFun_subComm(commName, commDir, rootDir,year2UniversalTimeStepIndex):
    print(f"comm {commName} running on {mp.current_process().name}")

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data files
    current_directory = os.getcwd()
    intermediate_directory = os.path.join(current_directory, r'intermediate_data_folder')
    
    subComm_QuestionsWithEventList_directory = commDir+'/'+f'QuestionsWithEventList_tag_{commName}.dict'
    with open(subComm_QuestionsWithEventList_directory, 'rb') as inputFile:
        Questions = pickle.load( inputFile)
    
    # process Questions chunk by chunk
    n_proc = mp.cpu_count()-2 # left 2 cores to do others
    all_outputs = []
    with mp.Pool(processes=n_proc) as pool:
        args = zip(list(Questions.items()), len(Questions)*[commName], len(Questions)*[year2UniversalTimeStepIndex])
        # issue tasks to the process pool and wait for tasks to complete
        results = pool.starmap(myAction_subComm, args , chunksize=n_proc)
        # process pool is closed automatically
        for res in results:
            if res != None:
                if isinstance(res, str):
                    print(res)
                else:
                    all_outputs.append(res)
            else:
                print(f"None")
    
    Questions.clear()
    results.clear()
    # combine results
    print(f"start to combine year2voteCountAtEachYear to year2totalCount...for {commName}")
    if len(all_outputs) == 0:
        print(f"no output for {commName}")
        return
    year2totalCount = defaultdict()
    for tup in all_outputs: 
        qid = tup[0]
        print(f"combining question {qid} results to total for {commName}")
        year2answerCountAndVoteCount = tup[1]

        minYear = min(list(year2answerCountAndVoteCount.keys()))

        if minYear not in year2totalCount.keys():
            year2totalCount[minYear] = {'questionCount':1, 'answerCount':year2answerCountAndVoteCount[minYear]['answerCount'], 'voteCount':year2answerCountAndVoteCount[minYear]['voteCount']}

        else:
            year2totalCount[minYear]['questionCount'] += 1
            year2totalCount[minYear]['answerCount'] += year2answerCountAndVoteCount[minYear]['answerCount']
            year2totalCount[minYear]['voteCount'] += year2answerCountAndVoteCount[minYear]['voteCount']

        for year, d in year2answerCountAndVoteCount.items():
            if year == minYear:
                continue
            if year not in year2totalCount.keys():
                year2totalCount[year] = {'questionCount':1, 'answerCount':d['answerCount'], 'voteCount':d['voteCount']}
            else:
                year2totalCount[year]['questionCount'] += 1
                year2totalCount[year]['answerCount'] += d['answerCount']
                year2totalCount[year]['voteCount'] += d['voteCount']
    # sort
    year2totalCount = dict(sorted(year2totalCount.items()))
                
    # save total statistics
    with open(intermediate_directory+'/'+'SOF_subComms_yearlyStatistics.dict', 'wb') as outputFile:
        pickle.dump(year2totalCount, outputFile) 
        print(f"saved year2totalCount for {commName}.")  

    # write into descriptive_allComm_statistics.csv
    with open(rootDir+'/'+f'descriptive_SOF_subComm_statistics.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for year, d in year2totalCount.items():
            writer.writerow( [commName, year, d['questionCount'], d['answerCount'], d['voteCount']])
        print(f"saved year2totalCount for {commName} in descriptive_allComm_statistics.csv.")

def main_subComm():

    t0=time.time()
    rootDir = os.getcwd()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("other sorted CommDir loaded.")

    with open(rootDir+'/'+f'descriptive_SOF_subComm_statistics.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow( ["commName","till Year","question Count", "answer count", "vote count"])
    
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

    # # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for tup in subCommName2commDir.items():
        commName = tup[0]
        commDir = tup[1]

        try:
            p = mp.Process(target=myFun_subComm, args=(commName,commDir, rootDir, year2UniversalTimeStepIndex))
            p.start()
        except Exception as e:
            print(e)
            pscount = sum(1 for proc in psutil.process_iter() if proc.name() == 'python3')
            print(f"current python3 processes count {pscount}.")
            return

        processes.append(p)
        if len(processes)==1:
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
    print('get statistics about sub comms    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    # main()

    main_subComm()
