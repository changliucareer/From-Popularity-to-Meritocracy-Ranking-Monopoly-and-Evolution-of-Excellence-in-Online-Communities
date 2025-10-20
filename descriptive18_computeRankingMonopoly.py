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

        ai2voteDiffsoFar = defaultdict()
        ai2creationTime = defaultdict()

        for i,e in enumerate(eventList):
            eventType = e['et']
            if eventType not in ['w', 'v']:
                continue

            lifetime += 1

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
        k2RORandRPI = defaultdict()

        for k in k_options:
            topK_AnswerAiList = list(sorted_topAnswerAi2durationInInactivePeriod.keys())[:k]

            RORk = 0

            # compute Rank Occupancy Rate (ROR)
            for t in range(1, lifetime+1):
                listIndex = t-1
                topKpositionsAvailableCount = 0 # Denominator
                topPositionsOccupiedByTopKAnswersCount = 0 # Numerator

                if answerCountAtEachTime[listIndex] < k:
                    topKpositionsAvailableCount = answerCountAtEachTime[listIndex]
                else: 
                    topKpositionsAvailableCount = k

                for ai, r in enumerate(answerRankingsAtEachTime[listIndex]):
                    if (r <= k) and (ai in topK_AnswerAiList):
                        topPositionsOccupiedByTopKAnswersCount += 1
                
                curRatio = topPositionsOccupiedByTopKAnswersCount / topKpositionsAvailableCount
            
                RORk += curRatio

            # averaging RORk
            RORk = RORk / lifetime

            # compute Rank Persistence Index (RPI)
            Isum = 0
            for topAnswerAi in topAnswerAiAtEachTime:
                if topAnswerAi in topK_AnswerAiList:
                    Isum += 1
            
            minCreationTime = min([ai2creationTime[topKAnswerAi] for topKAnswerAi in topK_AnswerAiList])
            RPIk = Isum / (lifetime-minCreationTime)

            k2RORandRPI[k]= {'ROR':RORk, 'RPI':RPIk, 'topKAnswerCount':len(topK_AnswerAiList)}
        

        # return processed question's qid, content, the number of answers with votes, the number of real votes as samples
        return (qid, lifetime, lastAnswerTime, inactivePeriod, answerCountAtEachTime, topAnswerAiAtEachTime, answerRankingsAtEachTime, k2RORandRPI)
    except Exception as e:
        return str(e)    


def myAction_new (question_tuple, commName, k_options, answerCount2voteCountAtEachRank):
    try:
        if answerCount2voteCountAtEachRank == None:
            return
        
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

        ai2voteDiffsoFar = defaultdict()
        ai2creationTime = defaultdict()

        # for Understanding 1 (pos vote and neg vote as same, both as 1)
        # for Understanding 2 (pos vote and neg vote as 1 and -1)
        # for Understanding 3 (only pos vote as 1)

        understanding2ai2W = defaultdict()  # ai2understanding2WeightedVoteCount  for MCR (W_ij)

        voteTupleList = [] # a list of vote related info (vote, answer index, rank when cast, timeTick, understanding2w_jt, understanding2p_r)

        for i,e in enumerate(eventList):
            eventType = e['et']
            if eventType not in ['w', 'v']:
                continue

            lifetime += 1

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

                # compute w_jt for MCR
                understanding2w_jt = defaultdict()
                # understanding 1: pos vote and neg vote as same, both as 1
                understanding2w_jt[1] = 1
                
                # understanding 2: pos vote and neg vote as 1 and -1
                if vote == 1:
                    understanding2w_jt[2] = 1
                else:
                    understanding2w_jt[2] = -1
                
                # understanding 2:  only pos vote as 1
                if vote == 1:
                    understanding2w_jt[3] = 1
                else:
                    understanding2w_jt[3] = 0
                
                # update ai2W for MCR
                cur_rank2voteCount = answerCount2voteCountAtEachRank[answerCountAtEachTime[-1]]
                rank2totalVoteCount = defaultdict()
                rank2posVoteCount = defaultdict()
                rank2negVoteCount = defaultdict()
                for rank, d in cur_rank2voteCount.items():
                    pos_vc = d['pos']
                    neg_vc = d['neg']
                    rank2totalVoteCount[rank] = pos_vc + neg_vc
                    rank2posVoteCount[rank] = pos_vc
                    rank2negVoteCount[rank] = neg_vc

                ranksOfAnswersBeforeT = e['ranks']
                cur_rank = ranksOfAnswersBeforeT[ans_index]

                understanding2p_r = defaultdict()
                # for understanding 1
                if sum(rank2totalVoteCount.values()) == 0:
                    understanding2p_r[1] = 0
                else:
                    understanding2p_r[1] = rank2totalVoteCount[cur_rank] / sum(rank2totalVoteCount.values())
                # for understanding 2
                if sum(rank2posVoteCount.values()) == 0:
                    understanding2p_r[2] = 0
                else:
                    if vote == 1:
                        understanding2p_r[2] = rank2posVoteCount[cur_rank] / sum(rank2posVoteCount.values())
                    else:
                        understanding2p_r[2] = rank2negVoteCount[cur_rank] / sum(rank2negVoteCount.values())
                # for understanding 3
                if sum(rank2posVoteCount.values()) == 0:
                    understanding2p_r[3] = 0
                else:
                    understanding2p_r[3] = rank2posVoteCount[cur_rank] / sum(rank2posVoteCount.values())

                for understanding, w_jt in understanding2w_jt.items():
                    p_r = understanding2p_r[understanding] # when using different distribution
                    # p_r = understanding2p_r[1] # when using the same distribution
                    if p_r == 0: # avoid division by zero
                        continue
                    if understanding not in understanding2ai2W.keys():
                        understanding2ai2W[understanding] = {ans_index:w_jt / p_r}
                    else:
                        if ans_index not in understanding2ai2W[understanding].keys():
                            understanding2ai2W[understanding][ans_index] = w_jt / p_r
                        else:
                            understanding2ai2W[understanding][ans_index] += w_jt / p_r

                voteTupleList.append((vote, ans_index, cur_rank, lifetime, understanding2w_jt, understanding2p_r))

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
            

        # for given k
        understanding2k2MCRandHHI = defaultdict() # monopoly
        understanding2k2RORandRPI = defaultdict() # monopoly
        understanding2k2NTRandPNR = defaultdict() # Meritocracy Violation Index (MVI)

        understanding2ai2MS = defaultdict() # for MCR
        for understanding, ai2W in understanding2ai2W.items():    
            Wsum = sum(ai2W.values())  # W_i could be zero for understanding 2
            for ai, W in ai2W.items():
                if Wsum == 0:
                    MS = 0
                else:
                    MS = W / Wsum
                if understanding not in understanding2ai2MS.keys():
                    understanding2ai2MS[understanding] = {ai: MS}
                else:
                    understanding2ai2MS[understanding][ai]= MS

        for k in k_options:
            for understanding in [1,2,3]:
                ai2MS = understanding2ai2MS[understanding]
                # sort ai2MS by MS
                sorted_MStuples = sorted(ai2MS.items(), key=lambda x:x[1], reverse=True)
                topK_AnswerAiList = [tup[0] for tup in sorted_MStuples[:k]]
                MCR = sum([tup[1] for tup in sorted_MStuples[:k]])
                HHI = sum([tup[1]**2 for tup in sorted_MStuples[:k]])

                if understanding not in understanding2k2MCRandHHI.keys():
                    understanding2k2MCRandHHI[understanding] = {k:{'MCR':MCR, 'HHI':HHI}}
                else:
                    understanding2k2MCRandHHI[understanding][k] = {'MCR':MCR, 'HHI':HHI}

                # compute ROR and RPI
                RORk = 0
                # compute Rank Occupancy Rate (ROR)
                for t in range(1, lifetime+1):
                    listIndex = t-1
                    topKpositionsAvailableCount = 0 # Denominator
                    topPositionsOccupiedByTopKAnswersCount = 0 # Numerator

                    if answerCountAtEachTime[listIndex] < k:
                        topKpositionsAvailableCount = answerCountAtEachTime[listIndex]
                    else: 
                        topKpositionsAvailableCount = k

                    for ai, r in enumerate(answerRankingsAtEachTime[listIndex]):
                        if (r <= k) and (ai in topK_AnswerAiList):
                            topPositionsOccupiedByTopKAnswersCount += 1
                    
                    curRatio = topPositionsOccupiedByTopKAnswersCount / topKpositionsAvailableCount
                
                    RORk += curRatio

                # averaging RORk
                RORk = RORk / lifetime
                assert RORk <= 1

                # compute Rank Persistence Index (RPI)
                Isum = 0
                for i, topAnswerAi in enumerate(topAnswerAiAtEachTime):
                    timeTick = i+1
                    creationTimeTick = ai2creationTime[topAnswerAi]
                    if (topAnswerAi in topK_AnswerAiList) and (timeTick > creationTimeTick):
                        Isum += 1
                
                minCreationTime = min([ai2creationTime[topKAnswerAi] for topKAnswerAi in topK_AnswerAiList])
                RPIk = Isum / (lifetime-minCreationTime)
                assert RPIk <= 1

                if understanding not in understanding2k2RORandRPI.keys():
                    understanding2k2RORandRPI[understanding] = {k:{'ROR':RORk, 'RPI':RPIk, 'topKAnswerCount':len(topK_AnswerAiList)}}
                else:
                    understanding2k2RORandRPI[understanding][k]= {'ROR':RORk, 'RPI':RPIk, 'topKAnswerCount':len(topK_AnswerAiList)}

                # for MVI
                # through inactivePeriod
                posVoteCount = 0
                negVoteCount = 0
                posVoteCountForNonTopKAnswers = 0
                negVoteCountForTopkAnswers = 0

                for voteTuple in voteTupleList:
                    vote, ans_index, rank, timeTick, understanding2w_jt, understanding2p_r = voteTuple
                    w_jt = understanding2w_jt[understanding]
                    p_r = understanding2p_r[understanding]
                    if p_r == 0: # avoid division by zero
                        continue

                    if timeTick > lastAnswerTime: # only consider the inactive period
                        if vote == 1:
                            posVoteCount += w_jt / p_r # condition on rank IPW
                            if ans_index not in topK_AnswerAiList:
                                posVoteCountForNonTopKAnswers += w_jt / p_r
                        else: # negative vote
                            negVoteCount += w_jt / p_r
                            if ans_index in topK_AnswerAiList:
                                negVoteCountForTopkAnswers += w_jt / p_r

                if negVoteCount == 0:
                    NTRk = 0
                else:
                    NTRk = negVoteCountForTopkAnswers / negVoteCount
                
                if posVoteCount == 0:
                    PNRk = 0
                else:
                    PNRk = posVoteCountForNonTopKAnswers / posVoteCount
                
                if understanding not in understanding2k2NTRandPNR.keys():
                    understanding2k2NTRandPNR[understanding] = {k:{'NTR':NTRk, 'PNR':PNRk}}
                else:
                    understanding2k2NTRandPNR[understanding][k] = {'NTR':NTRk, 'PNR':PNRk}

        # return processed question's qid, content, the number of answers with votes, the number of real votes as samples
        return (qid, lifetime, lastAnswerTime, inactivePeriod, answerCountAtEachTime, topAnswerAiAtEachTime, answerRankingsAtEachTime, voteTupleList, understanding2ai2MS, understanding2k2MCRandHHI, understanding2k2RORandRPI, understanding2k2NTRandPNR)
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

    #  # # check whether already done this step, skip
    # resultFiles = [f'rankingMonopolyOutputs.dict']
    # resultFiles = [intermediate_directory+'/'+f for f in resultFiles]
    # if os.path.exists(resultFiles[0]):
    #     # target date
    #     target_date = datetime.datetime(2023, 10, 10)
    #     # file last modification time
    #     timestamp = os.path.getmtime(resultFiles[0])
    #     # convert timestamp into DateTime object
    #     datestamp = datetime.datetime.fromtimestamp(timestamp)
    #     print(f'{commName} Modified Date/Time:{datestamp}')
    #     if datestamp >= target_date:
    #         print(f"{commName} has already done this step.")
    #         return
    
    with open(intermediate_directory+'/'+'QuestionsWithAnswersWithVotesWithPostHistoryWithMatrixWithEventList.dict', 'rb') as inputFile:
        Questions = pickle.load( inputFile)

    ori_QuestionCount = len(Questions)
    if ori_QuestionCount == 0:
        return
    
    # load qid2answerCount2voteCountAtEachRank
    with open(intermediate_directory+'/'+'qid2answerCount2voteCountAtEachRank.dict', 'rb') as inputFile:
        qid2answerCount2voteCountAtEachRank = pickle.load( inputFile)

    answerCount2voteCountAtEachRankList = [qid2answerCount2voteCountAtEachRank[qid] if qid in qid2answerCount2voteCountAtEachRank.keys() else None for qid in Questions.keys()]

    # process Questions chunk by chunk
    n_proc = mp.cpu_count()-2 # left 2 cores to do others
    all_outputs = []
    with mp.Pool(processes=n_proc) as pool:
        args = zip(list(Questions.items()), len(Questions)*[commName], len(Questions)*[k_options], answerCount2voteCountAtEachRankList)
        # issue tasks to the process pool and wait for tasks to complete
        results = pool.starmap(myAction_new, args , chunksize=n_proc)
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
    questionCountForMonopolyComputation = len(all_outputs)
    lifetimeList = []
    inactivePeriodList = []

    # for MCR and HHI
    understanding2k2MCRandHHI_comm = defaultdict()
    understanding2k2RORandRPI_comm = defaultdict()
    understanding2k2NTRandPNR_comm = defaultdict()

    for tup in all_outputs: 
        qid, lifetime, lastAnswerTime, inactivePeriod, answerCountAtEachTime, topAnswerAiAtEachTime, answerRankingsAtEachTime, voteTupleList, understanding2ai2MS, understanding2k2MCRandHHI, understanding2k2RORandRPI, understanding2k2NTRandPNR = tup
        qid2outputs[qid] = {'lifetime':lifetime, 'lastAnswerTime':lastAnswerTime, 'inactivePeriod':inactivePeriod,
                            'answerCountAtEachTime': answerCountAtEachTime,
                            'topAnswerAiAtEachTime':topAnswerAiAtEachTime,
                            'answerRanksAtEachTime':answerRankingsAtEachTime,
                            'voteTupleList':voteTupleList,
                            'understanding2ai2MS':understanding2ai2MS,
                            'understanding2k2MCRandHHI':understanding2k2MCRandHHI,
                            'understanding2k2RORandRPI':understanding2k2RORandRPI,
                            'understanding2k2NTRandPNR':understanding2k2NTRandPNR}
        
        lifetimeList.append(lifetime)
        inactivePeriodList.append(inactivePeriod)

        for understanding,k2MCRandHHI in understanding2k2MCRandHHI.items():
            for k,d in k2MCRandHHI.items():
                MCRk = d['MCR']
                HHIk = d['HHI']
                
                if understanding not in understanding2k2MCRandHHI_comm.keys():
                    understanding2k2MCRandHHI_comm[understanding] = {k:{'MCR_List':[MCRk], 'HHI_List':[HHIk]}}
                else: # update 
                    if k not in understanding2k2MCRandHHI_comm[understanding].keys():
                        understanding2k2MCRandHHI_comm[understanding][k] = {'MCR_List':[MCRk], 'HHI_List':[HHIk]}
                    else:
                        understanding2k2MCRandHHI_comm[understanding][k]['MCR_List'].append(MCRk)
                        understanding2k2MCRandHHI_comm[understanding][k]['HHI_List'].append(HHIk)
            
            k2RORandRPI = understanding2k2RORandRPI[understanding]
            for k,d in k2RORandRPI.items():
                RORk = d['ROR']
                RPIk = d['RPI']
                topKAnswerCount = d['topKAnswerCount']

                if understanding not in understanding2k2RORandRPI_comm.keys():
                    understanding2k2RORandRPI_comm[understanding] = {k:{'ROR_List':[RORk], 'RPI_List':[RPIk], 'topKAnswerCount_List':[topKAnswerCount]}}
                else: # update
                    if k not in understanding2k2RORandRPI_comm[understanding].keys():
                        understanding2k2RORandRPI_comm[understanding][k] = {'ROR_List':[RORk], 'RPI_List':[RPIk], 'topKAnswerCount_List':[topKAnswerCount]}
                    else: # update 
                        understanding2k2RORandRPI_comm[understanding][k]['ROR_List'].append(RORk)
                        understanding2k2RORandRPI_comm[understanding][k]['RPI_List'].append(RPIk)
                        understanding2k2RORandRPI_comm[understanding][k]['topKAnswerCount_List'].append(topKAnswerCount)
            
            k2NTRandPNR = understanding2k2NTRandPNR[understanding]
            for k,d in k2NTRandPNR.items():
                NTRk = d['NTR']
                PNRk = d['PNR']

                if understanding not in understanding2k2NTRandPNR_comm.keys():
                    understanding2k2NTRandPNR_comm[understanding] = {k:{'NTR_List':[NTRk], 'PNR_List':[PNRk]}}
                else:
                    if k not in understanding2k2NTRandPNR_comm[understanding].keys():
                        understanding2k2NTRandPNR_comm[understanding][k] = {'NTR_List':[NTRk], 'PNR_List':[PNRk]}
                    else:
                        understanding2k2NTRandPNR_comm[understanding][k]['NTR_List'].append(NTRk)
                        understanding2k2NTRandPNR_comm[understanding][k]['PNR_List'].append(PNRk)
    
    # averaging aggregation
    for understanding in [1,2,3]:
        for k in k_options:
            understanding2k2MCRandHHI_comm[understanding][k]['avg_MCR'] = mean(understanding2k2MCRandHHI_comm[understanding][k]['MCR_List'])
            understanding2k2MCRandHHI_comm[understanding][k]['avg_HHI'] = mean(understanding2k2MCRandHHI_comm[understanding][k]['HHI_List'])
            
            understanding2k2RORandRPI_comm[understanding][k]['avg_ROR'] = mean(understanding2k2RORandRPI_comm[understanding][k]['ROR_List'])
            understanding2k2RORandRPI_comm[understanding][k]['avg_RPI'] = mean(understanding2k2RORandRPI_comm[understanding][k]['RPI_List'])
            understanding2k2RORandRPI_comm[understanding][k]['avg_topKAnswerCount'] = mean(understanding2k2RORandRPI_comm[understanding][k]['topKAnswerCount_List'])

            understanding2k2NTRandPNR_comm[understanding][k]['avg_NTR'] = mean(understanding2k2NTRandPNR_comm[understanding][k]['NTR_List'])
            understanding2k2NTRandPNR_comm[understanding][k]['avg_PNR'] = mean(understanding2k2NTRandPNR_comm[understanding][k]['PNR_List'])

    avg_lifetime = mean(lifetimeList)
    avg_inactivePeriod = mean(inactivePeriodList)

    # save updated Questions
    with open(intermediate_directory+'/'+'rankingMonopolyOutputs_new.dict', 'wb') as outputFile:
        pickle.dump((qid2outputs,understanding2k2MCRandHHI_comm, understanding2k2RORandRPI_comm, understanding2k2NTRandPNR_comm), outputFile) 
        print(f"saved rankingMonopolyOutputs for {commName}.")  
    
    with open(rootDir+'/'+f'allComm_rankingMonopoly_statistics_new.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow( [commName,ori_QuestionCount, questionCountForMonopolyComputation, 
                          avg_lifetime, avg_inactivePeriod, 
                          understanding2k2MCRandHHI_comm[1][1]['avg_MCR'], understanding2k2MCRandHHI_comm[1][1]['avg_HHI'],
                          understanding2k2RORandRPI_comm[1][1]['avg_ROR'], understanding2k2RORandRPI_comm[1][1]['avg_RPI'],
                          understanding2k2NTRandPNR_comm[1][1]['avg_NTR'], understanding2k2NTRandPNR_comm[1][1]['avg_PNR'],
                          understanding2k2MCRandHHI_comm[1][2]['avg_MCR'], understanding2k2MCRandHHI_comm[1][2]['avg_HHI'],
                          understanding2k2RORandRPI_comm[1][2]['avg_ROR'], understanding2k2RORandRPI_comm[1][2]['avg_RPI'],
                            understanding2k2NTRandPNR_comm[1][2]['avg_NTR'], understanding2k2NTRandPNR_comm[1][2]['avg_PNR'],
                          understanding2k2MCRandHHI_comm[1][3]['avg_MCR'], understanding2k2MCRandHHI_comm[1][3]['avg_HHI'],
                            understanding2k2RORandRPI_comm[1][3]['avg_ROR'], understanding2k2RORandRPI_comm[1][3]['avg_RPI'],
                            understanding2k2NTRandPNR_comm[1][3]['avg_NTR'], understanding2k2NTRandPNR_comm[1][3]['avg_PNR'],
                          understanding2k2MCRandHHI_comm[2][1]['avg_MCR'], understanding2k2MCRandHHI_comm[2][1]['avg_HHI'],
                            understanding2k2RORandRPI_comm[2][1]['avg_ROR'], understanding2k2RORandRPI_comm[2][1]['avg_RPI'],
                            understanding2k2NTRandPNR_comm[2][1]['avg_NTR'], understanding2k2NTRandPNR_comm[2][1]['avg_PNR'],
                          understanding2k2MCRandHHI_comm[2][2]['avg_MCR'], understanding2k2MCRandHHI_comm[2][2]['avg_HHI'],
                            understanding2k2RORandRPI_comm[2][2]['avg_ROR'], understanding2k2RORandRPI_comm[2][2]['avg_RPI'],
                            understanding2k2NTRandPNR_comm[2][2]['avg_NTR'], understanding2k2NTRandPNR_comm[2][2]['avg_PNR'],
                          understanding2k2MCRandHHI_comm[2][3]['avg_MCR'], understanding2k2MCRandHHI_comm[2][3]['avg_HHI'],
                            understanding2k2RORandRPI_comm[2][3]['avg_ROR'], understanding2k2RORandRPI_comm[2][3]['avg_RPI'],
                            understanding2k2NTRandPNR_comm[2][3]['avg_NTR'], understanding2k2NTRandPNR_comm[2][3]['avg_PNR'],
                            understanding2k2MCRandHHI_comm[3][1]['avg_MCR'], understanding2k2MCRandHHI_comm[3][1]['avg_HHI'],
                            understanding2k2RORandRPI_comm[3][1]['avg_ROR'], understanding2k2RORandRPI_comm[3][1]['avg_RPI'],
                            understanding2k2NTRandPNR_comm[3][1]['avg_NTR'], understanding2k2NTRandPNR_comm[3][1]['avg_PNR'],
                            understanding2k2MCRandHHI_comm[3][2]['avg_MCR'], understanding2k2MCRandHHI_comm[3][2]['avg_HHI'],
                            understanding2k2RORandRPI_comm[3][2]['avg_ROR'], understanding2k2RORandRPI_comm[3][2]['avg_RPI'],
                            understanding2k2NTRandPNR_comm[3][2]['avg_NTR'], understanding2k2NTRandPNR_comm[3][2]['avg_PNR'],
                            understanding2k2MCRandHHI_comm[3][3]['avg_MCR'], understanding2k2MCRandHHI_comm[3][3]['avg_HHI'],
                            understanding2k2RORandRPI_comm[3][3]['avg_ROR'], understanding2k2RORandRPI_comm[3][3]['avg_RPI'],
                            understanding2k2NTRandPNR_comm[3][3]['avg_NTR'], understanding2k2NTRandPNR_comm[3][3]['avg_PNR'],
                        ])

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

    # load qid2answerCount2voteCountAtEachRank
    with open(intermediate_directory+'/'+'qid2answerCount2voteCountAtEachRank.dict', 'rb') as inputFile:
        qid2answerCount2voteCountAtEachRank = pickle.load( inputFile)


    all_outputs = []
    for i, subDir in enumerate(partFiles):
        part = i+1
        partDir = subDir
        # get question count of each part
        with open(partDir, 'rb') as inputFile:
            Questions_part = pickle.load( inputFile)
            ori_QuestionCount += len(Questions_part)
            print(f"part {part} of {commName} is loaded.")

            answerCount2voteCountAtEachRankList = [qid2answerCount2voteCountAtEachRank[qid] if qid in qid2answerCount2voteCountAtEachRank.keys() else None for qid in Questions_part.keys()]
        
        # process Questions chunk by chunk
        n_proc = mp.cpu_count()-2 # left 2 cores to do others
        with mp.Pool(processes=n_proc) as pool:
            args = zip(list(Questions_part.items()), len(Questions_part)*[commName], len(Questions_part)*[k_options], answerCount2voteCountAtEachRankList)
            # issue tasks to the process pool and wait for tasks to complete
            results = pool.starmap(myAction_new, args , chunksize=n_proc)
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
    questionCountForMonopolyComputation = len(all_outputs)
    lifetimeList = []
    inactivePeriodList = []

    # for MCR and HHI
    understanding2k2MCRandHHI_comm = defaultdict()
    understanding2k2RORandRPI_comm = defaultdict()
    understanding2k2NTRandPNR_comm = defaultdict()

    for tup in all_outputs: 
        qid, lifetime, lastAnswerTime, inactivePeriod, answerCountAtEachTime, topAnswerAiAtEachTime, answerRankingsAtEachTime, voteTupleList, understanding2ai2MS, understanding2k2MCRandHHI, understanding2k2RORandRPI, understanding2k2NTRandPNR = tup
        qid2outputs[qid] = {'lifetime':lifetime, 'lastAnswerTime':lastAnswerTime, 'inactivePeriod':inactivePeriod,
                            'answerCountAtEachTime': answerCountAtEachTime,
                            'topAnswerAiAtEachTime':topAnswerAiAtEachTime,
                            'answerRanksAtEachTime':answerRankingsAtEachTime,
                            'voteTupleList':voteTupleList,
                            'understanding2ai2MS':understanding2ai2MS,
                            'understanding2k2MCRandHHI':understanding2k2MCRandHHI,
                            'understanding2k2RORandRPI':understanding2k2RORandRPI,
                            'understanding2k2NTRandPNR':understanding2k2NTRandPNR}
        
        lifetimeList.append(lifetime)
        inactivePeriodList.append(inactivePeriod)

        for understanding,k2MCRandHHI in understanding2k2MCRandHHI.items():
            for k,d in k2MCRandHHI.items():
                MCRk = d['MCR']
                HHIk = d['HHI']
                
                if understanding not in understanding2k2MCRandHHI_comm.keys():
                    understanding2k2MCRandHHI_comm[understanding] = {k:{'MCR_List':[MCRk], 'HHI_List':[HHIk]}}
                else: # update 
                    if k not in understanding2k2MCRandHHI_comm[understanding].keys():
                        understanding2k2MCRandHHI_comm[understanding][k] = {'MCR_List':[MCRk], 'HHI_List':[HHIk]}
                    else:
                        understanding2k2MCRandHHI_comm[understanding][k]['MCR_List'].append(MCRk)
                        understanding2k2MCRandHHI_comm[understanding][k]['HHI_List'].append(HHIk)
            
            k2RORandRPI = understanding2k2RORandRPI[understanding]
            for k,d in k2RORandRPI.items():
                RORk = d['ROR']
                RPIk = d['RPI']
                topKAnswerCount = d['topKAnswerCount']

                if understanding not in understanding2k2RORandRPI_comm.keys():
                    understanding2k2RORandRPI_comm[understanding] = {k:{'ROR_List':[RORk], 'RPI_List':[RPIk], 'topKAnswerCount_List':[topKAnswerCount]}}
                else: # update
                    if k not in understanding2k2RORandRPI_comm[understanding].keys():
                        understanding2k2RORandRPI_comm[understanding][k] = {'ROR_List':[RORk], 'RPI_List':[RPIk], 'topKAnswerCount_List':[topKAnswerCount]}
                    else: # update 
                        understanding2k2RORandRPI_comm[understanding][k]['ROR_List'].append(RORk)
                        understanding2k2RORandRPI_comm[understanding][k]['RPI_List'].append(RPIk)
                        understanding2k2RORandRPI_comm[understanding][k]['topKAnswerCount_List'].append(topKAnswerCount)
            
            k2NTRandPNR = understanding2k2NTRandPNR[understanding]
            for k,d in k2NTRandPNR.items():
                NTRk = d['NTR']
                PNRk = d['PNR']

                if understanding not in understanding2k2NTRandPNR_comm.keys():
                    understanding2k2NTRandPNR_comm[understanding] = {k:{'NTR_List':[NTRk], 'PNR_List':[PNRk]}}
                else:
                    if k not in understanding2k2NTRandPNR_comm[understanding].keys():
                        understanding2k2NTRandPNR_comm[understanding][k] = {'NTR_List':[NTRk], 'PNR_List':[PNRk]}
                    else:
                        understanding2k2NTRandPNR_comm[understanding][k]['NTR_List'].append(NTRk)
                        understanding2k2NTRandPNR_comm[understanding][k]['PNR_List'].append(PNRk)
    
    # averaging aggregation
    for understanding in [1,2,3]:
        for k in k_options:
            understanding2k2MCRandHHI_comm[understanding][k]['avg_MCR'] = mean(understanding2k2MCRandHHI_comm[understanding][k]['MCR_List'])
            understanding2k2MCRandHHI_comm[understanding][k]['avg_HHI'] = mean(understanding2k2MCRandHHI_comm[understanding][k]['HHI_List'])
            
            understanding2k2RORandRPI_comm[understanding][k]['avg_ROR'] = mean(understanding2k2RORandRPI_comm[understanding][k]['ROR_List'])
            understanding2k2RORandRPI_comm[understanding][k]['avg_RPI'] = mean(understanding2k2RORandRPI_comm[understanding][k]['RPI_List'])
            understanding2k2RORandRPI_comm[understanding][k]['avg_topKAnswerCount'] = mean(understanding2k2RORandRPI_comm[understanding][k]['topKAnswerCount_List'])

            understanding2k2NTRandPNR_comm[understanding][k]['avg_NTR'] = mean(understanding2k2NTRandPNR_comm[understanding][k]['NTR_List'])
            understanding2k2NTRandPNR_comm[understanding][k]['avg_PNR'] = mean(understanding2k2NTRandPNR_comm[understanding][k]['PNR_List'])

    avg_lifetime = mean(lifetimeList)
    avg_inactivePeriod = mean(inactivePeriodList)

    # save updated Questions
    with open(intermediate_directory+'/'+'rankingMonopolyOutputs_new.dict', 'wb') as outputFile:
        pickle.dump((qid2outputs,understanding2k2MCRandHHI_comm, understanding2k2RORandRPI_comm, understanding2k2NTRandPNR_comm), outputFile) 
        print(f"saved rankingMonopolyOutputs for {commName}.")  
    
    with open(rootDir+'/'+f'allComm_rankingMonopoly_statistics_new.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow( [commName,ori_QuestionCount, questionCountForMonopolyComputation, 
                          avg_lifetime, avg_inactivePeriod, 
                          understanding2k2MCRandHHI_comm[1][1]['avg_MCR'], understanding2k2MCRandHHI_comm[1][1]['avg_HHI'],
                          understanding2k2RORandRPI_comm[1][1]['avg_ROR'], understanding2k2RORandRPI_comm[1][1]['avg_RPI'],
                          understanding2k2NTRandPNR_comm[1][1]['avg_NTR'], understanding2k2NTRandPNR_comm[1][1]['avg_PNR'],
                          understanding2k2MCRandHHI_comm[1][2]['avg_MCR'], understanding2k2MCRandHHI_comm[1][2]['avg_HHI'],
                          understanding2k2RORandRPI_comm[1][2]['avg_ROR'], understanding2k2RORandRPI_comm[1][2]['avg_RPI'],
                            understanding2k2NTRandPNR_comm[1][2]['avg_NTR'], understanding2k2NTRandPNR_comm[1][2]['avg_PNR'],
                          understanding2k2MCRandHHI_comm[1][3]['avg_MCR'], understanding2k2MCRandHHI_comm[1][3]['avg_HHI'],
                            understanding2k2RORandRPI_comm[1][3]['avg_ROR'], understanding2k2RORandRPI_comm[1][3]['avg_RPI'],
                            understanding2k2NTRandPNR_comm[1][3]['avg_NTR'], understanding2k2NTRandPNR_comm[1][3]['avg_PNR'],
                          understanding2k2MCRandHHI_comm[2][1]['avg_MCR'], understanding2k2MCRandHHI_comm[2][1]['avg_HHI'],
                            understanding2k2RORandRPI_comm[2][1]['avg_ROR'], understanding2k2RORandRPI_comm[2][1]['avg_RPI'],
                            understanding2k2NTRandPNR_comm[2][1]['avg_NTR'], understanding2k2NTRandPNR_comm[2][1]['avg_PNR'],
                          understanding2k2MCRandHHI_comm[2][2]['avg_MCR'], understanding2k2MCRandHHI_comm[2][2]['avg_HHI'],
                            understanding2k2RORandRPI_comm[2][2]['avg_ROR'], understanding2k2RORandRPI_comm[2][2]['avg_RPI'],
                            understanding2k2NTRandPNR_comm[2][2]['avg_NTR'], understanding2k2NTRandPNR_comm[2][2]['avg_PNR'],
                          understanding2k2MCRandHHI_comm[2][3]['avg_MCR'], understanding2k2MCRandHHI_comm[2][3]['avg_HHI'],
                            understanding2k2RORandRPI_comm[2][3]['avg_ROR'], understanding2k2RORandRPI_comm[2][3]['avg_RPI'],
                            understanding2k2NTRandPNR_comm[2][3]['avg_NTR'], understanding2k2NTRandPNR_comm[2][3]['avg_PNR'],
                            understanding2k2MCRandHHI_comm[3][1]['avg_MCR'], understanding2k2MCRandHHI_comm[3][1]['avg_HHI'],
                            understanding2k2RORandRPI_comm[3][1]['avg_ROR'], understanding2k2RORandRPI_comm[3][1]['avg_RPI'],
                            understanding2k2NTRandPNR_comm[3][1]['avg_NTR'], understanding2k2NTRandPNR_comm[3][1]['avg_PNR'],
                            understanding2k2MCRandHHI_comm[3][2]['avg_MCR'], understanding2k2MCRandHHI_comm[3][2]['avg_HHI'],
                            understanding2k2RORandRPI_comm[3][2]['avg_ROR'], understanding2k2RORandRPI_comm[3][2]['avg_RPI'],
                            understanding2k2NTRandPNR_comm[3][2]['avg_NTR'], understanding2k2NTRandPNR_comm[3][2]['avg_PNR'],
                            understanding2k2MCRandHHI_comm[3][3]['avg_MCR'], understanding2k2MCRandHHI_comm[3][3]['avg_HHI'],
                            understanding2k2RORandRPI_comm[3][3]['avg_ROR'], understanding2k2RORandRPI_comm[3][3]['avg_RPI'],
                            understanding2k2NTRandPNR_comm[3][3]['avg_NTR'], understanding2k2NTRandPNR_comm[3][3]['avg_PNR'],
                        ])
    


def main():

    t0=time.time()
    rootDir = os.getcwd()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("other sorted CommDir loaded.")

    logFileName = 'descriptive18_computeRankingMonopoly_Log.txt'

    with open(rootDir+'/'+f'allComm_rankingMonopoly_statistics_new.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow( ["commName","original question Count", "queston count for monopoly computation", 
                          "average lifetime", "avg inactive period",
                          "understanding 1 k=1 (MCR)", "understanding 1 k=1 (HHI)",
                          "understanding 1 k=1 (ROR)", "understanding 1 k=1 (RPI)",
                          "understanding 1 k=1 (NTR)", "understanding 1 k=1 (PNR)",
                          "understanding 1 k=2 (MCR)", "understanding 1 k=2 (HHI)",
                            "understanding 1 k=2 (ROR)", "understanding 1 k=2 (RPI)",
                            "understanding 1 k=2 (NTR)", "understanding 1 k=2 (PNR)",
                          "understanding 1 k=3 (MCR)", "understanding 1 k=3 (HHI)",
                            "understanding 1 k=3 (ROR)", "understanding 1 k=3 (RPI)",
                            "understanding 1 k=3 (NTR)", "understanding 1 k=3 (PNR)",
                          "understanding 2 k=1 (MCR)", "understanding 2 k=1 (HHI)",
                            "understanding 2 k=1 (ROR)", "understanding 2 k=1 (RPI)",
                            "understanding 2 k=1 (NTR)", "understanding 2 k=1 (PNR)",
                          "understanding 2 k=2 (MCR)", "understanding 2 k=2 (HHI)",
                            "understanding 2 k=2 (ROR)", "understanding 2 k=2 (RPI)",
                            "understanding 2 k=2 (NTR)", "understanding 2 k=2 (PNR)",
                          "understanding 2 k=3 (MCR)", "understanding 2 k=3 (HHI)",
                            "understanding 2 k=3 (ROR)", "understanding 2 k=3 (RPI)",
                            "understanding 2 k=3 (NTR)", "understanding 2 k=3 (PNR)",
                          "understanding 3 k=1 (MCR)", "understanding 3 k=1 (HHI)",
                            "understanding 3 k=1 (ROR)", "understanding 3 k=1 (RPI)",
                            "understanding 3 k=1 (NTR)", "understanding 3 k=1 (PNR)",
                          "understanding 3 k=2 (MCR)", "understanding 3 k=2 (HHI)",
                            "understanding 3 k=2 (ROR)", "understanding 3 k=2 (RPI)",
                            "understanding 3 k=2 (NTR)", "understanding 3 k=2 (PNR)",
                          "understanding 3 k=3 (MCR)", "understanding 3 k=3 (HHI)",
                            "understanding 3 k=3 (ROR)", "understanding 3 k=3 (RPI)",
                            "understanding 3 k=3 (NTR)", "understanding 3 k=3 (PNR)",])
    
    
    # # test on comm "coffee.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[166][0], commDir_sizes_sortedlist[166][1], rootDir, logFileName)
    # # test on comm "datascience.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[301][0], commDir_sizes_sortedlist[301][1], rootDir, logFileName)
    # test on comm "webapps.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[305][0], commDir_sizes_sortedlist[305][1], rootDir, logFileName)
    # test on comm "philosophy" to debug
    # myFun(commDir_sizes_sortedlist[295][0], commDir_sizes_sortedlist[295][1], rootDir, logFileName)

    # test on comm "stackoverflow" to debug
    # myFun_SOF(commDir_sizes_sortedlist[359][0], commDir_sizes_sortedlist[359][1], rootDir, logFileName)

    
    
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

    
    # run stackoverflow at the last separately
    print(f"start to process stackoverflow alone...")
    myFun_SOF('stackoverflow', stackoverflow_dir, rootDir, logFileName)
    
            
    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('descriptive18 compute ranking monopoly Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
