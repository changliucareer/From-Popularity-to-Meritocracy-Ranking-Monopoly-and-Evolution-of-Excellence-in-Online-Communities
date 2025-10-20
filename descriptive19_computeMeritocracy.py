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
from statistics import mean

def whichYear(universal_timestepIndex, year2UniversalTimeStepIndex):
    for year, lastIndex in year2UniversalTimeStepIndex.items():
        if universal_timestepIndex <= lastIndex:
            return year
    return None


def myAction (question_tuple, commName, k_options):
    try:
        qid = question_tuple[0]
        content = question_tuple[1]
        print(f"processing question {qid} of {commName} on {mp.current_process().name}...")
        eventList = content['eventList']

        # get vote and write events count for current question
        lifetime = 0 # The total numbers of answer/vote events given to current question
        lastAnswerTime = 0 # The time tick when the last answer is generated
        answerCountAtEachTime = []
        topAnswerAiAtEachTime = []
        answerRankingsAtEachTime = []
        eventListIndexAtEachTime = []

        ai2voteDiffsoFar = defaultdict()
        ai2creationTime = defaultdict()

        for i,e in enumerate(eventList):
            eventType = e['et']
            if eventType not in ['w', 'v']:
                continue

            lifetime += 1
            eventListIndexAtEachTime.append(i)

            if eventType == 'w': # write an new answer
                ans_index = e['ai']
                if len(answerCountAtEachTime) == 0: # writing the first answer of this quesiton
                    answerCountAtEachTime.append(1)
                else:
                    answerCountAtEachTime.append(answerCountAtEachTime[-1]+1)
                
                lastAnswerTime = lifetime

                ai2creationTime[ans_index] = lifetime

                #initial ize vote diff of this new answer
                ai2voteDiffsoFar[ans_index] = 0

            else : # vote event
                answerCountAtEachTime.append(answerCountAtEachTime[-1]) # same as previous time
                ans_index = e['ai']
                vote = e['v']
                if vote == 1: # positive vote
                    ai2voteDiffsoFar[ans_index] += 1
                else: # negative vote
                    ai2voteDiffsoFar[ans_index] -= 1
            
            # find the answer rankings after current time
            if i == len(eventList)-1: # the last event
                aiWithVoteScore = list(ai2voteDiffsoFar.items())
                aiWithVoteScore.sort(reverse=True, key=lambda x:(x[1],x[0])) # sort by the vote score and then by the answer index
                topAnswerAi = aiWithVoteScore[0][0]
                rankAnswerAiTupList = [(r+1, tup[0]) for r, tup in enumerate(aiWithVoteScore)]
                rankAnswerAiTupList.sort(key=lambda x:x[1]) # sort by answer index
                ranksOfAnswersAfterT = [tup[0] for tup in rankAnswerAiTupList] # only keep the ranks in order of answer index

            else: # not the last event
                ranksOfAnswersAfterT = eventList[i+1]['ranks']
                if ranksOfAnswersAfterT == None: # no previous vote casted, only sort by answer index
                    curAnswerCount = answerCountAtEachTime[-1]
                    ranksOfAnswersAfterT = list(range(curAnswerCount, 0, -1))
                    topAnswerAi = curAnswerCount-1
                else:
                    topAnswerAi = ranksOfAnswersAfterT.index(1)
                
            topAnswerAiAtEachTime.append(topAnswerAi)
            answerRankingsAtEachTime.append(ranksOfAnswersAfterT)

        assert len(answerRankingsAtEachTime) == lifetime
        assert len(answerCountAtEachTime) == lifetime
        assert len(topAnswerAiAtEachTime) == lifetime

        if lifetime == 0:
            return f"lifetime=0 for question {qid}\n"

        # compute inactive period
        inactivePeriod = lifetime - lastAnswerTime

        if inactivePeriod == 0:
            return f"inactivePeriod=0 for question {qid}\n"

        # answers that at the top rank during the inactive period
        topAnswerAi2durationInInactivePeriod = defaultdict()
        for i, topAnswerAi in enumerate(topAnswerAiAtEachTime[lastAnswerTime:]): # lastAnswerTime is already +1 than the index, so this equavalent to Time L+1 to Ti
            if topAnswerAi not in topAnswerAi2durationInInactivePeriod.keys():
                topAnswerAi2durationInInactivePeriod[topAnswerAi] = 1
            else:
                topAnswerAi2durationInInactivePeriod[topAnswerAi] += 1

        sorted_topAnswerAi2durationInInactivePeriod = dict(sorted(topAnswerAi2durationInInactivePeriod.items(), key=lambda item: (item[1], item[0]), reverse=True))

        # for given k
        k2NTRandPNR = defaultdict()

        for k in k_options:
            topK_AnswerAiList = list(sorted_topAnswerAi2durationInInactivePeriod.keys())[:k]

            # through inactivePeriod
            posVoteCount = 0
            negVoteCount = 0
            posVoteCountForNonTopKAnswers = 0
            negVoteCountForTopkAnswers = 0

            for t in range(lastAnswerTime+1, lifetime+1):
                listIndex = t-1
                eventIndex = eventListIndexAtEachTime[listIndex]
                e = eventList[eventIndex]
                if e['et'] =='v': # voting event
                    vote = e['v']
                    ans_index = e['ai']
                    if vote == 1:
                        posVoteCount += 1
                        if ans_index not in topK_AnswerAiList:
                            posVoteCountForNonTopKAnswers +=1
                    else: # negative vote
                        negVoteCount += 1
                        if ans_index in topK_AnswerAiList:
                            negVoteCountForTopkAnswers += 1

            if negVoteCount == 0:
                NTRk = 0
            else:
                NTRk = negVoteCountForTopkAnswers / negVoteCount
            
            if posVoteCount == 0:
                PNRk = 0
            else:
                PNRk = posVoteCountForNonTopKAnswers / posVoteCount
            
            k2NTRandPNR[k]= {'NTR':NTRk, 'PNR':PNRk}

        # return processed question's qid, content, the number of answers with votes, the number of real votes as samples
        return (qid, k2NTRandPNR)
    except Exception as e:
        return str(e)    

def myFun(commName, commDir, rootDir, logFileName):
    print(f"comm {commName} running on {mp.current_process().name}")

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())


    k_options = [1, 2, 3]

    # load intermediate_data files
    current_directory = os.getcwd()
    intermediate_directory = os.path.join(current_directory, r'intermediate_data_folder')

     # # check whether already done this step, skip
    resultFiles = [f'rankingMonopolyOutputs.dict']
    resultFiles = [intermediate_directory+'/'+f for f in resultFiles]
    if os.path.exists(resultFiles[0]):
        # target date
        target_date = datetime.datetime(2023, 10, 10)
        # file last modification time
        timestamp = os.path.getmtime(resultFiles[0])
        # convert timestamp into DateTime object
        datestamp = datetime.datetime.fromtimestamp(timestamp)
        print(f'{commName} Modified Date/Time:{datestamp}')
        if datestamp >= target_date:
            print(f"{commName} has already done this step.")
            return
    
    with open(intermediate_directory+'/'+'QuestionsWithAnswersWithVotesWithPostHistoryWithMatrixWithEventList.dict', 'rb') as inputFile:
        Questions = pickle.load( inputFile)

    ori_QuestionCount = len(Questions)
    if ori_QuestionCount == 0:
        return

    # process Questions chunk by chunk
    n_proc = mp.cpu_count()-2 # left 2 cores to do others
    all_outputs = []
    with mp.Pool(processes=n_proc) as pool:
        args = zip(list(Questions.items()), len(Questions)*[commName], len(Questions)*[k_options])
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
    print(f"start to combine outputs of all questions for {commName}...")
    qid2outputs = defaultdict()
    k2NTRandPNR_comm = defaultdict()

    questionCountForMonopolyComputation = len(all_outputs)

    for tup in all_outputs: 
        qid, k2NTRandPNR = tup
        qid2outputs[qid] = k2NTRandPNR
        
        for k,d in k2NTRandPNR.items():
            NTRk = d['NTR']
            PNRk = d['PNR']

            if k not in k2NTRandPNR_comm.keys():
                k2NTRandPNR_comm[k] = {'NTR_List':[NTRk], 'PNR_List':[PNRk]}
            else: # update 
                k2NTRandPNR_comm[k]['NTR_List'].append(NTRk)
                k2NTRandPNR_comm[k]['PNR_List'].append(PNRk)
    
    # averaging aggregation
    for k in k_options:
        k2NTRandPNR_comm[k]['avg_NTR'] = mean(k2NTRandPNR_comm[k]['NTR_List'])
        k2NTRandPNR_comm[k]['avg_PNR'] = mean(k2NTRandPNR_comm[k]['PNR_List'])

    # save updated Questions
    with open(intermediate_directory+'/'+'meritocracyViolationOutputs.dict', 'wb') as outputFile:
        pickle.dump((qid2outputs,k2NTRandPNR_comm), outputFile) 
        print(f"saved meritocracyViolationOutputs for {commName}.")  
    
    with open(rootDir+'/'+f'allComm_meritocracyViolation_statistics.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow( [commName,ori_QuestionCount, questionCountForMonopolyComputation, 
                          k2NTRandPNR_comm[1]['avg_NTR'], k2NTRandPNR_comm[1]['avg_PNR'],
                          k2NTRandPNR_comm[2]['avg_NTR'], k2NTRandPNR_comm[2]['avg_PNR'],
                          k2NTRandPNR_comm[3]['avg_NTR'], k2NTRandPNR_comm[3]['avg_PNR'],])

def myFun_SOF(commName, commDir, rootDir, logFileName):
    print(f"comm {commName} running on {mp.current_process().name}")

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    k_options = [1, 2, 3]

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

    ori_QuestionCount = 0

    all_outputs = []
    for i, subDir in enumerate(partFiles):
        part = i+1
        partDir = subDir
        # get question count of each part
        with open(partDir, 'rb') as inputFile:
            Questions_part = pickle.load( inputFile)
            ori_QuestionCount += len(Questions_part)
            print(f"part {part} of {commName} is loaded.")
        
        # process Questions chunk by chunk
        n_proc = mp.cpu_count()-2 # left 2 cores to do others
        with mp.Pool(processes=n_proc) as pool:
            args = zip(list(Questions_part.items()), len(Questions_part)*[commName], len(Questions_part)*[k_options])
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
    print(f"start to combine outputs of all questions for {commName}...")
    qid2outputs = defaultdict()
    k2NTRandPNR_comm = defaultdict()
    questionCountForMonopolyComputation = len(all_outputs)
    lifetimeList = []
    inactivePeriodList = []
    for tup in all_outputs: 
        qid, lifetime, lastAnswerTime, inactivePeriod, answerCountAtEachTime, topAnswerAiAtEachTime, answerRankingsAtEachTime, k2NTRandPNR = tup
        qid2outputs[qid] = {'lifetime':lifetime, 'lastAnswerTime':lastAnswerTime, 'inactivePeriod':inactivePeriod,
                            'answerCountAtEachTime': answerCountAtEachTime,
                            'topAnswerAiAtEachTime':topAnswerAiAtEachTime,
                            'answerRanksAtEachTime':answerRankingsAtEachTime,
                            'k2NTRandPNR':k2NTRandPNR}
        lifetimeList.append(lifetime)
        inactivePeriodList.append(inactivePeriod)

        for k,d in k2NTRandPNR.items():
            RORk = d['ROR']
            RPIk = d['RPI']
            topKAnswerCount = d['topKAnswerCount']

            if k not in k2NTRandPNR_comm.keys():
                k2NTRandPNR_comm[k] = {'ROR_List':[RORk], 'RPI_List':[RPIk], 'topKAnswerCount_List':[topKAnswerCount]}
            else: # update 
                k2NTRandPNR_comm[k]['ROR_List'].append(RORk)
                k2NTRandPNR_comm[k]['RPI_List'].append(RPIk)
                k2NTRandPNR_comm[k]['topKAnswerCount_List'].append(topKAnswerCount)
    
    # averaging aggregation
    for k in k_options:
        k2NTRandPNR_comm[k]['avg_ROR'] = mean(k2NTRandPNR_comm[k]['ROR_List'])
        k2NTRandPNR_comm[k]['avg_RPI'] = mean(k2NTRandPNR_comm[k]['RPI_List'])
        k2NTRandPNR_comm[k]['avg_topKAnswerCount'] = mean(k2NTRandPNR_comm[k]['topKAnswerCount_List'])
                
    avg_lifetime = mean(lifetimeList)
    avg_inactivePeriod = mean(inactivePeriodList)

    # save updated Questions
    with open(intermediate_directory+'/'+'rankingMonopolyOutputs.dict', 'wb') as outputFile:
        pickle.dump((qid2outputs,k2NTRandPNR_comm), outputFile) 
        print(f"saved rankingMonopolyOutputs for {commName}.")  
    
    with open(rootDir+'/'+f'allComm_rankingMonopoly_statistics.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow( [commName,ori_QuestionCount, questionCountForMonopolyComputation, 
                          k2NTRandPNR_comm[1]['avg_ROR'], k2NTRandPNR_comm[1]['avg_RPI'],
                          k2NTRandPNR_comm[2]['avg_ROR'], k2NTRandPNR_comm[2]['avg_RPI'],
                          k2NTRandPNR_comm[3]['avg_ROR'], k2NTRandPNR_comm[3]['avg_RPI'],])
    


def main():

    t0=time.time()
    rootDir = os.getcwd()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("other sorted CommDir loaded.")

    logFileName = 'descriptive18_computeRankingMonopoly_Log.txt'

    with open(rootDir+'/'+f'allComm_meritocracyViolation_statistics.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow( ["commName","original question Count", "queston count for monopoly computation", 
                          "k=1 (NTR)", "k=1 (PNR)",
                         "k=2 (NTR)", "k=2 (PNR)",
                          "k=3 (NTR)", "k=3 (PNR)"])
    
    
    # # test on comm "coffee.stackexchange" to debug
    myFun(commDir_sizes_sortedlist[166][0], commDir_sizes_sortedlist[166][1], rootDir, logFileName)
    # # test on comm "datascience.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[301][0], commDir_sizes_sortedlist[301][1], rootDir, logFileName)
    # test on comm "webapps.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[305][0], commDir_sizes_sortedlist[305][1], rootDir, logFileName)

    # test on comm "stackoverflow" to debug
    # myFun_SOF(commDir_sizes_sortedlist[359][0], commDir_sizes_sortedlist[359][1], rootDir, logFileName)

    """
    
    # # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for tup in commDir_sizes_sortedlist:
        commName = tup[0]
        commDir = tup[1]
        if commName != 'stackoverflow': # skip stackoverflow 
            try:
                p = mp.Process(target=myFun, args=(commName,commDir, rootDir,logFileName))
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
    
    """
    
    # # run stackoverflow at the last separately
    # print(f"start to process stackoverflow alone...")
    # myFun_SOF('stackoverflow', stackoverflow_dir, rootDir, logFileName)
    
            
    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('descriptive18 compute ranking monopoly Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
