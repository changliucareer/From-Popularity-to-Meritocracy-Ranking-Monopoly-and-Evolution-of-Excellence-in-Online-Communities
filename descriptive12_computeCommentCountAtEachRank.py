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

def datetime2UniversalTimeStepIndex(universal_timesteps, post2commentIdAndCreationTime):
    post2commentIdAndUTSindex = defaultdict()
    
    commentId2creationTime = defaultdict()

    for pid, content in post2commentIdAndCreationTime.items():
        creationTimes = content['creationTimes']
        commentIds = content['commentIds']
        for i, commentId in enumerate(commentIds):
            commentId2creationTime[commentId] = creationTimes[i]
    
    # sort commentId2creationTime by commentId
    sorted_commentId2creationTime = dict(sorted(commentId2creationTime.items(), key=lambda t:t[0]))

    commentId2UTSindex = defaultdict()
    startIndex = 0
    for commentId, dt in sorted_commentId2creationTime.items():
        findFlag = False
        for i, ut in enumerate(universal_timesteps):
            if i< startIndex: # skip before startIndex
                continue
            if dt < ut[2]: 
                commentId2UTSindex[commentId] = i - 0.5
                startIndex = i 
                findFlag = True
                break
            elif dt == ut[2]: 
                commentId2UTSindex[commentId] = i 
                startIndex = i 
                findFlag = True
                break
        if not findFlag: # haven't found cur comment dt yet, might be later than the last vote event
            commentId2UTSindex[commentId] = len(universal_timesteps)-1


    # sanity check
    assert len(commentId2UTSindex) == len(commentId2creationTime)


    # update post2commentIdAndUTSindex
    for pid, content in post2commentIdAndCreationTime.items():
        commentIds = content['commentIds']
        UTSindexList = []
        for commentId in commentIds:
            UTSindexList.append(commentId2UTSindex[commentId])
        
        post2commentIdAndUTSindex[pid] = {'commentIds':commentIds,'UTSindexList':UTSindexList}

    return post2commentIdAndUTSindex, commentId2UTSindex

def whichYear(universal_timestepIndex, year2UniversalTimeStepIndex):
    for year, lastIndex in year2UniversalTimeStepIndex.items():
        if universal_timestepIndex <= lastIndex:
            return year
    return None

def myAction (question_tuple, commName, year2UniversalTimeStepIndex, commentId2time):
    qid = question_tuple[0]
    content = question_tuple[1]
    print(f"processing question {qid} of {commName} on {mp.current_process().name}...")
    eventList = content['eventList']
    lts = content['local_timesteps']

    # compute comment count at each rank for each year
    year2commentCountAtEachRank = defaultdict()
    rank2commentCount = defaultdict()
    if commName == 'stackoverflow':
        year = lts[0][2].year
    else:
        year = whichYear(lts[0], year2UniversalTimeStepIndex)

    commentCount = 0
    startIndex = 0
    for commentId, time in commentId2time.items():
        for i,e in enumerate(eventList):
            if i < startIndex: # skip before startIndex
                continue

            eventType = e['et']
            if eventType != 'v': # skip all event that is not a vote
                continue

            t = lts[i]
            if time <= t:
                commentCount +=1
                ans_index = e['ai']
                ranksOfAnswersBeforeT = e['ranks']
                rank = ranksOfAnswersBeforeT[ans_index]

                if commName == 'stackoverflow':
                    cur_year = t[2].year
                else:
                    cur_year = whichYear(t, year2UniversalTimeStepIndex)
                if cur_year == year:
                    if rank in rank2commentCount.keys():
                        rank2commentCount[rank] += 1
                    else:
                        rank2commentCount[rank] = 1
                elif cur_year < year:
                    print(f"Exception: cur_year < year for {i+1}th time step")
                else: # cur_year > year
                    # save previous year's last universal time steps index
                    # sort rank2commentCount
                    rank2commentCount = dict(sorted(rank2commentCount.items()))
                    year2commentCountAtEachRank[year] = copy.deepcopy(rank2commentCount)
                    # start a new year
                    year = cur_year
                    # aggregate the current event
                    if rank in rank2commentCount.keys():
                        rank2commentCount[rank] += 1
                    else:
                        rank2commentCount[rank] = 1
                
                startIndex = i
                break
        

    # save the last year's rank2commentCount
    # sort rank2commentCount
    rank2commentCount = dict(sorted(rank2commentCount.items()))
    year2commentCountAtEachRank[year] = copy.deepcopy(rank2commentCount)

    # fill up unvoted year
    latestYear = min(list(year2UniversalTimeStepIndex.keys()))
    for year in range(latestYear, 2023):
        if year not in year2commentCountAtEachRank.keys():
            if latestYear not in year2commentCountAtEachRank.keys():
                year2commentCountAtEachRank[year] = {1:0} # empty dict as the vc of rank 1 is 0
            else:
                year2commentCountAtEachRank[year] = copy.deepcopy(year2commentCountAtEachRank[latestYear])
        else:
            latestYear = year
    
    # sort year2commentCountAtEachRank
    year2commentCountAtEachRank = dict(sorted(year2commentCountAtEachRank.items()))

    assert commentCount == sum([vc for r,vc in year2commentCountAtEachRank[2022].items()]) # vote count sum at last year must be equal to vote count

    print(f"{mp.current_process().name} return")

    # return processed question's qid, content, the number of answers with votes, the number of real votes as samples
    return (qid, year2commentCountAtEachRank)
    

def myFun(commName, commDir):
    print(f"comm {commName} running on {mp.current_process().name}")

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data files
    current_directory = os.getcwd()
    intermediate_directory = os.path.join(current_directory, r'intermediate_data_folder')

    # check whether already done this step, skip
    resultFiles = ['commentId2UTSindex.dict']
    resultFiles = [intermediate_directory+'/'+f for f in resultFiles]
    # if os.path.exists(resultFiles[0]) and os.path.exists(resultFiles[1]):
    if os.path.exists(resultFiles[0]):
        print(f"{commName} has already done this step.")
        return
    
    with open(intermediate_directory+'/'+'QuestionsWithAnswersWithVotesWithPostHistoryWithMatrixWithEventList.dict', 'rb') as inputFile:
        Questions = pickle.load( inputFile)
    
    with open(intermediate_directory+'/'+'postId2commentIdAndCreationTime.dict', 'rb') as inputFile:
        post2commentIdAndCreationTime = pickle.load( inputFile)

    # if not stackoverflow, convert data_time_obj to universal time step index
    if commName != 'stackoverflow':
        print(f"generating post2commentIdAndCreationTime and commentId2UTSindex for {commName}...")
        with open(intermediate_directory+'/'+'universal_timesteps_afterCombineQAVH.dict', 'rb') as inputFile:
            universal_timesteps = pickle.load( inputFile)

        post2commentIdAndCreationTime, commentId2UTSindex = datetime2UniversalTimeStepIndex(universal_timesteps, post2commentIdAndCreationTime)
        universal_timesteps.clear()

        # save updated commentId2UTSindex
        with open(intermediate_directory+'/'+'commentId2UTSindex.dict', 'wb') as outputFile:
            pickle.dump(commentId2UTSindex, outputFile) 
            print(f"saved commentId2UTSindex  for {commName}.") 
    
    else: # for stackoverflow
        commentId2time = defaultdict()
        for pid, content in post2commentIdAndCreationTime.items():
            commentIds = content['commentIds']
            timeList = content['creationTimes']
            for i,commentId in enumerate(commentIds):
                commentId2time[commentId]=timeList[i]
        
        # save updated commentId2time
        with open(intermediate_directory+'/'+'commentId2time.dict', 'wb') as outputFile:
            pickle.dump(commentId2time, outputFile) 
            print(f"saved commentId2time  for {commName}.") 
                
    """
    with open(intermediate_directory+'/'+'year2UniversalTimeStepIndex.dict', 'rb') as inputFile:
        year2UniversalTimeStepIndex = pickle.load( inputFile)
    
    # process Questions chunk by chunk
    n_proc = mp.cpu_count()
    all_outputs = []
    with mp.Pool(processes=n_proc) as pool:
        # prepare args
        args = []
        for qid,content in Questions.items():
            commentId2time_forCurQuestion = defaultdict()
            answerIds = content['filtered_answerList']
            commentIdAndUTSindexTups = [post2commentIdAndCreationTime[aid] for aid in answerIds if aid in post2commentIdAndCreationTime.keys()]
            for tup in commentIdAndUTSindexTups:
                commentIds = tup['commentIds']
                if 'UTSindexList' in tup.keys(): # for all comms except SOF
                    times = tup['UTSindexList']
                else: # for SOF
                    times = tup['creationTimes']
                for i, commentId in enumerate(commentIds):
                    commentId2time_forCurQuestion[commentId] = times[i]
            
            args.append(((qid,content), commName, year2UniversalTimeStepIndex, commentId2time_forCurQuestion))
        # issue tasks to the process pool and wait for tasks to complete
        results = pool.starmap(myAction, args , chunksize=n_proc)
        # process pool is closed automatically
        for res in results:
            if res != None:
                all_outputs.append(res)
            else:
                print(f"None")
    
    Questions.clear()
    results.clear()
    # combine results
    print("start to combine year2commentCountAtEachYear to each question...")
    year2commentCountAtEachRank_total = defaultdict()
    for tup in all_outputs: 
        qid = tup[0]
        print(f"combining question {qid} results to total for {commName}")
        year2commentCountAtEachRank = tup[1]

        for year, rank2commentCount in year2commentCountAtEachRank.items():
            if year not in year2commentCountAtEachRank_total.keys():
                year2commentCountAtEachRank_total[year] = copy.deepcopy(rank2commentCount)
                # sort
                year2commentCountAtEachRank_total = dict(sorted(year2commentCountAtEachRank_total.items()))
            else: # need to combine two rank2commentCount
                combined_rank2commentCount = copy.deepcopy(rank2commentCount)
                previous_rank2commentCount = year2commentCountAtEachRank_total[year]
                for previous_rank, previous_vc in previous_rank2commentCount.items():
                    if previous_rank in combined_rank2commentCount.keys():
                        combined_rank2commentCount[previous_rank] = combined_rank2commentCount[previous_rank] + previous_vc
                    else:
                        combined_rank2commentCount[previous_rank] = previous_vc
                # sort
                combined_rank2commentCount = dict(sorted(combined_rank2commentCount.items()))
                # update year2commentCountAtEachRank_total
                year2commentCountAtEachRank_total[year] = combined_rank2commentCount
    # sort
    year2commentCountAtEachRank_total = dict(sorted(year2commentCountAtEachRank_total.items()))
                
    # save updated Questions
    with open(intermediate_directory+'/'+'yearlycommentCountAtEachRank.dict', 'wb') as outputFile:
        pickle.dump(year2commentCountAtEachRank_total, outputFile) 
        print(f"saved yearlycommentCountAtEachRank.dict for {commName}.")  

def myFun_SOF(commName, commDir):
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
    print("start to combine year2commentCountAtEachYear to each question...")
    year2commentCountAtEachRank_total = defaultdict()
    totalQcount = len(all_outputs)
    for i,tup in enumerate(all_outputs): 
        qid = tup[0]
        print(f"combining the {i+1}th of {totalQcount} question results to total for {commName}")
        year2commentCountAtEachRank = tup[1]

        for year, rank2commentCount in year2commentCountAtEachRank.items():
            if year not in year2commentCountAtEachRank_total.keys():
                year2commentCountAtEachRank_total[year] = copy.deepcopy(rank2commentCount)
                # sort
                year2commentCountAtEachRank_total = dict(sorted(year2commentCountAtEachRank_total.items()))
            else: # need to combine two rank2commentCount
                combined_rank2commentCount = copy.deepcopy(rank2commentCount)
                previous_rank2commentCount = year2commentCountAtEachRank_total[year]
                for previous_rank, previous_vc in previous_rank2commentCount.items():
                    if previous_rank in combined_rank2commentCount.keys():
                        combined_rank2commentCount[previous_rank] = combined_rank2commentCount[previous_rank] + previous_vc
                    else:
                        combined_rank2commentCount[previous_rank] = previous_vc
                # sort
                combined_rank2commentCount = dict(sorted(combined_rank2commentCount.items()))
                # update year2commentCountAtEachRank_total
                year2commentCountAtEachRank_total[year] = combined_rank2commentCount
    # sort
    year2commentCountAtEachRank_total = dict(sorted(year2commentCountAtEachRank_total.items()))
                
    # save updated Questions
    with open(intermediate_directory+'/'+'yearlycommentCountAtEachRank.dict', 'wb') as outputFile:
        pickle.dump(year2commentCountAtEachRank_total, outputFile) 
        print(f"saved yearlycommentCountAtEachRank.dict for {commName}.")  
    """

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

    
    
    # # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for tup in commDir_sizes_sortedlist:
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

    
    # # # run stackoverflow at the last separately
    # print(f"start to process stackoverflow alone...")
    # myFun_SOF('stackoverflow', stackoverflow_dir)
    
            
    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('descriptive12 compute comment count at each rank Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
