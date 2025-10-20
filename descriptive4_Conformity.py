import os
#First print the current working directory
print("Current Working Directory", os.getcwd())
###############################################################
from toolFunctions import format_time, writeIntoLog, writeIntoResult,saveModel,savePlot
import time
import datetime
import glob
import multiprocessing as mp
import math
from collections import defaultdict
import matplotlib.pyplot as plt
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
from scipy import stats
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from statistics import mean

import descriptive4_Conformity_forStackOverflow


def negExpFunc(x, a, b, c):
    return a * np.exp(- (b * x)) + c

def ExpFunc(x, a, b, c):
    return a * np.exp(b * x) + c

def powerlawFunc(x, a, b, c):
    return a/(x**b+1)+c

def simplifiedpowerlawFunc(x, b,c):
    return 1/(x**b+1) + c


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

def myFun(commName, commDir, return_conformity_dict, root_dir):
    print(f"comm {commName} running on {mp.current_process().name}")
    logFileName = 'descriptive4_Conformity_log.txt'

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data files
    current_directory = os.getcwd()
    intermediate_directory = os.path.join(current_directory, r'intermediate_data_folder')
    
    """
    try:
        with open(intermediate_directory+'/'+'yearlyValuesForConformityComputation.dict', 'rb') as inputFile:
            year2valuesForConformityComputation_total = pickle.load( inputFile)

        # process Questions chunk by chunk
        all_outputs = []
        n_proc = mp.cpu_count()-2 # left 2 cores to do others
        with mp.Pool(processes=n_proc) as pool:
            args = []
            for year, valuesForConformityComputation in year2valuesForConformityComputation_total.items():
                args.append((commName,commDir, year,valuesForConformityComputation,logFileName))
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

        results.clear()

        year2Conformity= defaultdict()
        for tup in all_outputs:
            year, Conformity, totalVoteCount = tup
            year2Conformity[year] = {'conformtiy':Conformity,'size':totalVoteCount}
        
    except: # don't have one file for all years, try to find in splitted folder, and using the same algorithm for SOF
    """
    # don't have one file for all years, try to find in splitted folder, and using the same algorithm for SOF
    splitted_intermediate_data_folder = os.path.join(intermediate_directory, r'splitted_intermediate_data_folder')
    split_valuesForConformtiyComputation_forEachYear_directory = os.path.join(splitted_intermediate_data_folder, r'totalValuesForConformtiyComputation_forEachYear_parts_folder')
    if not os.path.exists(split_valuesForConformtiyComputation_forEachYear_directory): 
        print("Exception: no split_valuesForConformtiyComputation_forEachYear_directory!")

    yearFiles = [ f.path for f in os.scandir(split_valuesForConformtiyComputation_forEachYear_directory) if f.path.endswith('.dict') ]
    # sort files based on year
    yearFiles.sort(key=lambda p: int(p.strip(".dict").split("_")[-1]))
    yearsCount = len(yearFiles)
                                
    print(f"there are {yearsCount} splitted files in {commName}")

    # compute Conformity
    year2Conformity= defaultdict()
    logSum = 0
    totalVoteCount = 0
    certainRank = 6 # an arbitrarily choosen rank for conformity verifying
    
    for subDir in yearFiles:
        year = int(subDir.strip(".dict").split("_")[-1])
        
        # get valuesForConformtiyComputation of each year
        with open(subDir, 'rb') as inputFile:
            valuesForConformtiyComputation = pickle.load( inputFile)
            print(f"year {year} of {commName} is loaded.")

        year, cur_logSum, cur_voteCount = myAction(commName, commDir, year, valuesForConformtiyComputation, logFileName)
        
        totalVoteCount += cur_voteCount
        logSum += cur_logSum
        if totalVoteCount !=0:
            Conformity = math.exp(logSum/totalVoteCount)
            logtext = f"{commName} year {year} logSum:{logSum}, Conformity: {Conformity}, size: {totalVoteCount}.\n"
            writeIntoLog(logtext, commDir, logFileName)
            print(logtext)
        else:
            Conformity = None
        year2Conformity[year] = {'conformtiy':Conformity,'size':totalVoteCount}
        
        # plot Conformity verifying 
        if year == 2022:  
            # load voteDiff2voteCounts
            certainRank = 6
            
            # plot at certain rank for whole comm
            try:
                with open(f'intermediate_data_folder/voteDiff2voteCounts_comm_atCertainRank{certainRank}.dict', 'rb') as inputFile:
                    voteDiff2voteCounts = pickle.load( inputFile)
                    print(f"loaded voteDiff2voteCounts till 2022 for {commName}.")  
                    descriptive4_Conformity_forStackOverflow.plotVoteProportionAtEachVoteDiffForCertainRank(commName,commDir,year, voteDiff2voteCounts,logFileName, certainRank)
            except Exception as e:
                print(e)
                writeIntoLog(f"fail to plot:{e}", commDir, logFileName)
            """
            # plot at certain rank for certain answer
            try:
                with open(f'intermediate_data_folder/valuesTupleAtCertainRank{certainRank}.dict', 'rb') as inputFile:
                    valuesTupleAtCertainRank = pickle.load( inputFile)
                    print(f"loaded valuesTupleAtCertainRank till 2022 for {commName}.") 
                    descriptive4_Conformity_forStackOverflow.plotVoteProportionAtEachVoteDiffForCertainRankCertainAnswer(commName,commDir,year, valuesTupleAtCertainRank,logFileName, certainRank)
            except Exception as e:
                print(e)
                writeIntoLog(f"fail to plot:{e}", commDir, logFileName)
            """
    # update return_conformity_dict
    return_conformity_dict[commName] = year2Conformity
                
    return_conformity_normalDict = defaultdict()
    for commName, d in return_conformity_dict.items():
        return_conformity_normalDict[commName] = d
    
    os.chdir(root_dir) # go back to root directory
    with open('descriptive_ConformityAndSize_allComm.dict', 'wb') as outputFile:
        pickle.dump(return_conformity_normalDict, outputFile)
        print(f"saved return_conformity_normalDict, {len(return_conformity_normalDict)} comms.")


def main():

    t0=time.time()
    root_dir = os.getcwd()

    
    # # check whether already done
    # with open('descriptive_TrendinessFittingResults_allComm.dict', 'rb') as inputFile:
    #     return_conformity_normalDict = pickle.load( inputFile)
    # print("return_conformity_normalDict loaded.")

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("other sorted CommDir loaded.")

    # use shared variable to communicate among all comm's process
    manager = mp.Manager()
    return_conformity_dict = manager.dict() # to save the conformity results of each community

    # # test on comm "coffee.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[166][0], commDir_sizes_sortedlist[166][1], return_conformity_dict, root_dir)
    # # test on comm "datascience.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[301][0], commDir_sizes_sortedlist[301][1], return_conformity_dict, root_dir)
    # test on comm "webapps.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[305][0], commDir_sizes_sortedlist[305][1], return_conformity_dict, root_dir)

    # skip splitted communities
    splitted_comms = ['tex.stackexchange','meta.stackexchange','softwareengineering.stackexchange','superuser','math.stackexchange','worldbuilding.stackexchange','codegolf.stackexchange','stackoverflow']
    # debug_comm = ['academia.stackexchange']
    # # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for tup in commDir_sizes_sortedlist[:359]:
        commName = tup[0]
        commDir = tup[1]

        # if commName not in debug_comm:
        #     continue

        try:
            p = mp.Process(target=myFun, args=(commName,commDir, return_conformity_dict, root_dir))
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

    return_conformity_normalDict = defaultdict()
    for commName, d in return_conformity_dict.items():
        return_conformity_normalDict[commName] = d
    
    os.chdir(root_dir) # go back to root directory
    with open('descriptive_ConformityAndSize_allComm.dict', 'wb') as outputFile:
        pickle.dump(return_conformity_normalDict, outputFile)
        print(f"saved return_conformity_normalDict, {len(return_conformity_normalDict)} comms.")
    
    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('descriptive 4 conformity computing and plotting Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
