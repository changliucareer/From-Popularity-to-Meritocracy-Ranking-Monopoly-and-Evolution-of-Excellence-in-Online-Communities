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


def myAction (question_tuple, commName):
    qid = question_tuple[0]
    content = question_tuple[1]
    print(f"processing question {qid} of {commName} on {mp.current_process().name}...")
    eventList = content['eventList']

    # compute vote count at each rank for each existing answer count
    answerCount2voteCountAtEachRank = defaultdict()

    existingAnswerIndexList =  []

    voteCount = 0
    for i,e in enumerate(eventList):
        eventType = e['et']

        if eventType not in ['w','v']: # ignore other eventType than writing and voting
            continue
        
        if eventType == 'w': # a new answer created
            if len(existingAnswerIndexList)== 0:
                existingAnswerIndexList.extend(list(range(e['ai']+1)))
            else:
                if e['ai'] not in existingAnswerIndexList:
                    curMaxAi = max(existingAnswerIndexList)
                    existingAnswerIndexList.extend(list(range(curMaxAi+1, e['ai']+1)))
            continue

        # for voting event
        vote = e['v']
        voteCount +=1
        ans_index = e['ai']
        ranksOfAnswersBeforeT = e['ranks']
        rank = ranksOfAnswersBeforeT[ans_index]

        answerCount = len(existingAnswerIndexList)

        if answerCount not in answerCount2voteCountAtEachRank.keys(): # a new answer Count, need a new empty rank2voteCount dict
            rank2voteCount = defaultdict()
        else: # load the previous rank2voteCount of current answerCount
            rank2voteCount = answerCount2voteCountAtEachRank[answerCount]

        if rank in rank2voteCount.keys():
            if vote == 1: # positive vote
                rank2voteCount[rank]['pos'] += 1
            else:
                rank2voteCount[rank]['neg'] += 1
        else:
            if vote == 1: # positive vote
                rank2voteCount[rank] = {'pos':1,'neg':0}
            else:
                rank2voteCount[rank] = {'pos':0,'neg':1}

        # update rank2voteCount for current answerCount
        assert len(rank2voteCount) <= answerCount
        answerCount2voteCountAtEachRank[answerCount] = rank2voteCount

    recheck = 0
    for r2v in answerCount2voteCountAtEachRank.values():
        recheck  += sum([vc['pos']+vc['neg'] for vc in r2v.values()])
    assert recheck == voteCount

    print(f"{mp.current_process().name} return")

    # return processed question's qid, content, the number of answers with votes, the number of real votes as samples
    return (qid, answerCount2voteCountAtEachRank)
    

def myFun(commName, commDir):
    print(f"comm {commName} running on {mp.current_process().name}")

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data files
    current_directory = os.getcwd()
    intermediate_directory = os.path.join(current_directory, r'intermediate_data_folder')
    
    with open(intermediate_directory+'/'+'QuestionsWithAnswersWithVotesWithPostHistoryWithMatrixWithEventList.dict', 'rb') as inputFile:
        Questions = pickle.load( inputFile)

    
    # process Questions chunk by chunk
    n_proc = mp.cpu_count()-2 # left 2 cores to do others
    all_outputs = []
    with mp.Pool(processes=n_proc) as pool:
        args = zip(list(Questions.items()), len(Questions)*[commName])
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
    print("start to combine answerCount2voteCountAtEachRank to each question...")
    if len(all_outputs) == 0:
        print(f"no output for {commName}.")
        return

    answerCount2voteCountAtEachRank_total = defaultdict()

    qid2answerCount2voteCountAtEachRank = defaultdict()
    
    for tup in all_outputs: 
        qid = tup[0]
        print(f"combining question {qid} results to total for {commName}")
        answerCount2voteCountAtEachRank = tup[1]
        qid2answerCount2voteCountAtEachRank[qid] = answerCount2voteCountAtEachRank

    #     for answerCount, rank2voteCount in answerCount2voteCountAtEachRank.items():
    #         rank2voteCount = dict(sorted(rank2voteCount.items()))
    #         if answerCount not in answerCount2voteCountAtEachRank_total.keys():
    #             answerCount2voteCountAtEachRank_total[answerCount] = copy.deepcopy(rank2voteCount)
    #             # sort
    #             answerCount2voteCountAtEachRank_total = dict(sorted(answerCount2voteCountAtEachRank_total.items()))
    #         else: # need to combine two rank2voteCount
    #             combined_rank2voteCount = copy.deepcopy(rank2voteCount)
    #             previous_rank2voteCount = answerCount2voteCountAtEachRank_total[answerCount]
    #             for previous_rank, previous_vc in previous_rank2voteCount.items():
    #                 if previous_rank in combined_rank2voteCount.keys():
    #                     combined_rank2voteCount[previous_rank]['pos'] += previous_vc['pos']
    #                     combined_rank2voteCount[previous_rank]['neg'] += previous_vc['neg']
    #                 else:
    #                     combined_rank2voteCount[previous_rank] = previous_vc
    #             # sort
    #             combined_rank2voteCount = dict(sorted(combined_rank2voteCount.items()))
    #             assert len(combined_rank2voteCount) <= answerCount
    #             # update answer2voteCountAtEachRank_total
    #             answerCount2voteCountAtEachRank_total[answerCount] = combined_rank2voteCount
    # # sort
    # answerCount2voteCountAtEachRank_total = dict(sorted(answerCount2voteCountAtEachRank_total.items()))
                
    # # save updated Questions
    # with open(intermediate_directory+'/'+'answerCount2VoteCountAtEachRank.dict', 'wb') as outputFile:
    #     pickle.dump(answerCount2voteCountAtEachRank_total, outputFile) 
    #     print(f"saved answerCount2VoteCountAtEachRank.dict for {commName}.")  
    
    
    # save qid2answerCount2voteCountAtEachRank
    with open(intermediate_directory+'/'+'qid2answerCount2voteCountAtEachRank.dict', 'wb') as outputFile:
        pickle.dump(qid2answerCount2voteCountAtEachRank, outputFile) 
        print(f"saved qid2answerCount2voteCountAtEachRankdict for {commName}, length:{len(qid2answerCount2voteCountAtEachRank)}.")  

def myFun_SOF(commName, commDir):
    print(f"comm {commName} running on {mp.current_process().name}")

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # go to the target splitted files folder
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')

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
            args = zip(list(Questions_part.items()), len(Questions_part)*[commName])
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
    print("start to combine answerCount2voteCountAtEachRank to each question...")
    if len(all_outputs) == 0:
        print(f"no output for {commName}.")
        return
    
    answerCount2voteCountAtEachRank_total = defaultdict()

    qid2answerCount2voteCountAtEachRank = defaultdict()
    
    for tup in all_outputs: 
        qid = tup[0]
        print(f"combining question {qid} results to total for {commName}")
        answerCount2voteCountAtEachRank = tup[1]
        qid2answerCount2voteCountAtEachRank[qid] = answerCount2voteCountAtEachRank

    #     for answerCount, rank2voteCount in answerCount2voteCountAtEachRank.items():
    #         rank2voteCount = dict(sorted(rank2voteCount.items()))
    #         if answerCount not in answerCount2voteCountAtEachRank_total.keys():
    #             answerCount2voteCountAtEachRank_total[answerCount] = copy.deepcopy(rank2voteCount)
    #             # sort
    #             answerCount2voteCountAtEachRank_total = dict(sorted(answerCount2voteCountAtEachRank_total.items()))
    #         else: # need to combine two rank2voteCount
    #             combined_rank2voteCount = copy.deepcopy(rank2voteCount)
    #             previous_rank2voteCount = answerCount2voteCountAtEachRank_total[answerCount]
    #             for previous_rank, previous_vc in previous_rank2voteCount.items():
    #                 if previous_rank in combined_rank2voteCount.keys():
    #                     combined_rank2voteCount[previous_rank]['pos'] += previous_vc['pos']
    #                     combined_rank2voteCount[previous_rank]['neg'] += previous_vc['neg']
    #                 else:
    #                     combined_rank2voteCount[previous_rank] = previous_vc
    #             # sort
    #             combined_rank2voteCount = dict(sorted(combined_rank2voteCount.items()))
    #             assert len(combined_rank2voteCount) <= answerCount
    #             # update answer2voteCountAtEachRank_total
    #             answerCount2voteCountAtEachRank_total[answerCount] = combined_rank2voteCount
    # # sort
    # answerCount2voteCountAtEachRank_total = dict(sorted(answerCount2voteCountAtEachRank_total.items()))
                
    # # save updated Questions
    # with open(intermediate_directory+'/'+'answerCount2VoteCountAtEachRank.dict', 'wb') as outputFile:
    #     pickle.dump(answerCount2voteCountAtEachRank_total, outputFile) 
    #     print(f"saved answerCount2VoteCountAtEachRank.dict for {commName}.")  
        
    # save qid2answerCount2voteCountAtEachRank
    with open(intermediate_directory+'/'+'qid2answerCount2voteCountAtEachRank.dict', 'wb') as outputFile:
        pickle.dump(qid2answerCount2voteCountAtEachRank, outputFile) 
        print(f"saved qid2answerCount2voteCountAtEachRankdict for {commName}, length:{len(qid2answerCount2voteCountAtEachRank)}.")  


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

    
    # # run stackoverflow at the last separately
    print(f"start to process stackoverflow alone...")
    myFun_SOF('stackoverflow', stackoverflow_dir)
    
            
    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('descriptive21 compute vote count at each rank groupbed by existing answer count Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
