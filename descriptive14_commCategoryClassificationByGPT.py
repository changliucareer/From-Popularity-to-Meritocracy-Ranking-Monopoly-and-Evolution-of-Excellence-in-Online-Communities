import os
import sys
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
import pickle
import csv
import pandas as pd
import operator
import copy
from itertools import groupby
import re
import psutil
# from scipy.sparse import csr_matrix, lil_matrix
import numpy as np
import torch
from tqdm import tqdm
from statistics import mean
from sklearn.model_selection import train_test_split
import sklearn
from CustomizedNN import LRNN_1layer, LRNN_1layer_bias, LRNN_1layer_bias_specify,LRNN_1layer_bias_withoutRankTerm
from torch.utils.data import IterableDataset, DataLoader
from itertools import cycle, islice
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix 
from sklearn.preprocessing import normalize
import random
import math
import scipy.stats
import statsmodels.api as sm
from scipy import stats
import json
from bs4 import BeautifulSoup
from openai import OpenAI
import tiktoken
import csv
csv.field_size_limit(sys.maxsize)


def replace_img_tags(data):
    p = re.compile(r'<img.*(/img)?>')
    return p.sub('image', data)

def replace_code_tags(data):
    p = re.compile(r'<code.*(/code)?>')
    return p.sub('code', data)

def replace_URL(data):
    p = re.compile(r'\S*https?:\S*')
    return p.sub('URL', data)


def cleanComment(comment_text):
    if ('<img' in comment_text) or ('<code' in comment_text) or ('http' in comment_text):
        # replace <img> and <code> segment
        cleaned_comment_text = replace_img_tags(comment_text)
        cleaned_comment_text = replace_code_tags(cleaned_comment_text)
        cleaned_comment_text = replace_URL(cleaned_comment_text)
        return BeautifulSoup(cleaned_comment_text, "lxml").text
    else: 
        return BeautifulSoup(comment_text, "lxml").text

def generate_prompt(postId2text, answer2parentQ):

    sampleSize = 100
    answerIds = [id for id in postId2text.keys() if int(id) in answer2parentQ.keys()]
    # random.seed(44) # for round5
    sampledPostIds = random.sample(answerIds, sampleSize)

    sampledTextList = [v.replace('\n',' ') for (k,v) in postId2text.items() if k in sampledPostIds]

    # zero-shot
    # prompt = 'Classify the *Document into 4 categories (Knowledge, Experience, Belief, Opinion) based on the content of *Document and the *Definition of the categories. Response with name of each category and its corresponding probability in the following format "Knowledge:0.5, Experience:0.3, Belief:0.1, Opinion:0.1" \n'
    prompt = 'Classify the *Document into 4 categories (A, B, C, D) based on the content of *Document and the *Definition of the categories. Response only with name of each category and its corresponding probability in such format "Category A: probability of A, Category B: probability of B, Category C: probability of C, Category D: probability of D". The probabilities of categories should be float numbers and sum up to 1\n'
    
    prompt += '\n*Definition:\n'
    # prompt += 'Knowledge is an awareness of facts, a familiarity with individuals and situations, or a practical skill. \n'
    # prompt += 'Experience refers to conscious events in general, more specifically to perceptions, or to the practical knowledge and familiarity that is produced by these processes.\n'
    # prompt += 'Belief is a subjective attitude that a proposition is true or a state of affairs is the case. \n'
    # prompt += 'Opinion is a judgment, viewpoint, or statement that is not conclusive, rather than facts, which are true statements.\n'

#    prompt += 'Category A talks about objective facts. This category of information is not trendy, and is not changing much over time. People tend to see various content on these issues.\n'
#    prompt += 'Category B talks about objective topics that are more specifically to perceptions, or to practical knowledge. This category of information is trendy, and changing over time. People tend to find popular content on these issues.\n'
#    prompt += 'Category C talks about subjective topics. This category of information is not trendy, and is not changing much over time. People tend to see various content on these issues.\n'
#    prompt += 'Category D talks about subjective opinions, judgments or viewpoints. This category of information is trendy, and changing over time. People tend to find popular content on these issues.\n'

    prompt += 'Category A talks about objective fact. This category of information is not trendy, and is not changing much over time. People tend to see various content on these issues.\n'
    prompt += 'Category B talks about objective topics that are more specifically to the practical knowlege. This category of information is trendy, and changing over time. People tend to find popular content on these issues.\n'
    prompt += 'Category C talks about subjective discussions. This category of information is not trendy, and is not changing much over time. People tend to see various content on these issues.\n'
    prompt += 'Category D talks about subjective opinions. This category of information is trendy, and changing over time. People tend to find popular content on these issues.\n'

    prompt += '\n*Document:\n'
    for text in sampledTextList:
        prompt += text + ' '

    return prompt, sampledPostIds

def askGPT (my_prompt, my_model):
    
    my_prompt = my_prompt.replace('"',"'") # replace " with ' to avoid "" break in script
    
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key="your-key",
    )

    response = client.chat.completions.create(
        model=my_model,
        messages=[{"role": "user", "content": my_prompt}]
        )
    
    res_text = response.choices[0].message.content.strip()
    print(res_text)
    textlist = res_text.split(',')

    GPT_judgement = {'knowledge': None, 'experience':None, 'belief':None, 'opinion':None}
    try:
        for text in textlist:
            text = text.lower()
            cat = text.split(':')[0]
            prob = float(text.split(':')[1].strip('.'))
            if 'category a' in cat:
                GPT_judgement['knowledge'] = prob
            elif 'category b' in cat:
                GPT_judgement['experience'] = prob
            elif 'category c' in cat:
                GPT_judgement['belief']= prob
            elif 'category d' in cat:
                GPT_judgement['opinion'] = prob
            else:
                print(f"found none in {res_text}")
    except Exception as e:
        print(e)
    
    return GPT_judgement, res_text

def myFun(commName, commDir,root_dir):
   
    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())
    print(f"processing {commName}")

    # load intermediate_data files
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')

    with open(intermediate_directory+'/'+'answer2parentQLookup.dict', 'rb') as inputFile:
        answer2parentQ = pickle.load( inputFile)

    
    # check whether have done this part
    if commName != 'stackoverflow':
        resultFileDir = intermediate_directory+'/'+'whole_postId2text.json'
    else: # for stackoverflow
        resultFileDir = intermediate_directory+'/'+'sampled_1percent_postId2text.json'
    if os.path.exists(resultFileDir):
        print(f"{commName} has done whole_postId2text extraction.")

    else:
        if commName != 'stackoverflow':
            # extract all post ids to get post text
            postId2text = defaultdict()
            
            # extract question and answer post text
            chunk_size = 1000000
            chunkIndex = 0
            for df in pd.read_csv('Posts.csv', chunksize=chunk_size, engine='python',sep=','):
                for line_count, row in df.iterrows():
                    print(f"processing processing {commName} chunk {chunkIndex} line {line_count}...")
                    targetPost = int(row['Id'])
                    if (targetPost in answer2parentQ.keys()) or (targetPost in answer2parentQ.values()):
                        if isinstance(row['Body'],str):
                            cur_text = cleanComment(row['Body'])
                            postId2text[targetPost]=cur_text
                    
            # Convert and write JSON object to file
            with open('intermediate_data_folder/whole_postId2text.json', "w") as outfile: 
                json.dump(postId2text, outfile)
                print(f"saved whole_ postId2text.json, len{len(postId2text)}")
        
        else: # for stackoverflow (only extract the sampled 1%)
            """
            # load year2SampledQids
            with open(intermediate_directory+'/'+f'Year2SampledQids_1percent.dict', 'rb') as inputFile:
                sampled_year2qids =  pickle.load( inputFile)
            
            # get all years sampled qids
            sampledQids = []
            for year, qids in sampled_year2qids.items():
                sampledQids.extend(qids)
            print(f"{commName} sampled questions count : {len(sampledQids)}")

            # get all sampled postIds
            # sampledAids = [aid for aid, qid in answer2parentQ.items() if qid in sampledQids]
            sampledAids = []

            # creat a folder if not exists to store splitted data files
            splitFolder_directory = os.path.join(intermediate_directory, r'splitted_intermediate_data_folder')
            if not os.path.exists(splitFolder_directory):
                print("Exception: no splitted_intermediate_data_folder")
            
            split_sampled_QuestionsWithEventList_files_directory = os.path.join(splitFolder_directory, f'Sampled1percent_QuestionsPartsWithEventList')
            if not os.path.exists(split_sampled_QuestionsWithEventList_files_directory): # didn't find the parts files
                print("Exception: no split_sampled_QuestionsWithEventList_files_directory!")


            Questions_subfolders = [ f.path for f in os.scandir(split_sampled_QuestionsWithEventList_files_directory) if f.path.split('/')[-1].startswith("Sampled_QuestionsWithAnswersWithVotesWithPostHistoryWithMatrixWithEventList_part_") ]
            Questions_subfolders.sort(key=lambda f: int(f.split('/')[-1].split('_')[-1].split('.')[0]))
            # scan parts
            for qf in Questions_subfolders:
                part = int(qf.split('/')[-1].split('_')[-1].split('.')[0])
                print(f"scanning part {part}...")
                with open(qf, 'rb') as inputFile:
                    Questions_part = pickle.load( inputFile)
                # adding answers to sampledAids
                for qid, content in Questions_part.items(): 
                    filtered_answerList = content['filtered_answerList']
                    sampledAids.extend(filtered_answerList)

            print(f"{commName} sampled answers count : {len(sampledAids)}")
            Questions_part.clear()
            
            sampledPostIds = sampledQids + sampledAids
            totalPostCount = len(sampledPostIds)
            # saved the sampled PostIds
            with open(intermediate_directory+'/'+f'sampled_1percent_postIds.dict', 'wb') as outputFile:
                pickle.dump(sampledPostIds, outputFile)
                print(f"saved {commName} sampled postIds, post count : {totalPostCount}")
            """
            # load sampledPostIds
            with open(intermediate_directory+'/'+f'sampled_1percent_postIds.dict', 'rb') as inputFile:
                sampledPostIds =  pickle.load( inputFile)
                totalPostCount = len(sampledPostIds)

            # extract sampled post ids to get post text
            postId2text = defaultdict()
            
            # extract question and answer post text
            chunk_size = 1000000
            chunkIndex = 0
            postCount = 0
            for df in pd.read_csv('Posts.csv', chunksize=chunk_size, engine='python',sep=','):
                for line_count, row in df.iterrows():
                    print(f"processing processing {commName} chunk {chunkIndex} line {line_count}...")
                    targetPost = int(row['Id'])
                    if targetPost in sampledPostIds:
                        postCount +=1
                        try:
                            cur_text = cleanComment(row['Body'])
                            postId2text[targetPost]=cur_text
                            print(f"{commName}, added {postCount}th/{totalPostCount} post to postId2text.")
                        except:
                            print(f"Exception in chunk {chunkIndex} line {line_count} of {commName}. raw Body is {row['Body']}.")
                    
                    if line_count%1000 ==0:
                                # Convert and write JSON object to file
                                with open('intermediate_data_folder/sampled_1percent_postId2text_till.json', "w") as outfile: 
                                    json.dump(postId2text, outfile)
                                    print(f"saved sampled_1percent_ postId2text_till.json, len{len(postId2text)}")
                chunkIndex += 1
                    
            # Convert and write JSON object to file
            with open('intermediate_data_folder/sampled_1percent_postId2text.json', "w") as outfile: 
                json.dump(postId2text, outfile)
                print(f"saved sampled_1percent_ postId2text.json, len{len(postId2text)}")
    
    """
    # load post text
    if commName == 'stackoverflow':
        with open(intermediate_directory+'/'+'sampled_1percent_postId2text.json') as json_file:
            postId2text = json.load(json_file)
    else:
        with open(intermediate_directory+'/'+'whole_postId2text.json') as json_file:
            postId2text = json.load(json_file)

    # generate prompts
    print(f"generating prompt for {commName}...")
    my_prompt, sampledPostIds = generate_prompt(postId2text, answer2parentQ)

    
    # # count tokens
    print(f"counting prompt tokens for {commName}...")
    # enc = tiktoken.get_encoding("cl100k_base")
    # assert enc.decode(enc.encode("hello world")) == "hello world"
    # To get the tokeniser corresponding to a specific model in the OpenAI API:
    enc = tiktoken.encoding_for_model("gpt-4-turbo-preview")
    my_prompt = my_prompt.replace('"',"'") # replace " with ' to avoid "" break in script
    tokenCount = len(enc.encode(my_prompt))
    
    csvfile = open(root_dir+'/descriptive14_prompt3_round5_tokenCounts.csv', 'a', newline='')
    writer = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow([commName, tokenCount, sampledPostIds])
    csvfile.close()
    
    
    # ask GPT
    print(f"asking GPT for {commName}...")
    # my_model = "gpt-3.5-turbo"
    my_model = "gpt-4-turbo-preview"
    GPT_judgement, res_text = askGPT(my_prompt, my_model)
    
    # save results
    GPTresponse_files_directory = os.path.join(commDir, r'GPTresponse_folder')
    if not os.path.exists(GPTresponse_files_directory):
        print("no GPTresponse_files_directory, create one")
        os.makedirs(GPTresponse_files_directory)
    with open('GPTresponse_folder/descriptive14_prompt3_round5_'+my_model+'.json', "w") as outfile: 
        json.dump( (GPT_judgement,res_text), outfile) 
        # print(f"saved round5 for {commName}: my_prompt token count {tokenCount}, GPT_judgement:{GPT_judgement}\n")
        print(f"saved round5 for {commName}: GPT_judgement:{GPT_judgement}\n")

    """
    
def main():

    t0=time.time()
    root_dir = os.getcwd()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("sorted CommDir loaded.")

    ## Load top 120 selected comm names for Map
    with open('topSelectedCommNames_descriptive.dict', 'rb') as inputFile:
        topSelectedCommNames = pickle.load( inputFile)
    print("seleted comm names loaded.")

    """
    # save results of all comms
    print(f"start to save the results as csv...")
    csvfile = open('descriptive14_prompt3_round5_tokenCounts.csv', 'w', newline='')
    writer = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow( ["commName","token count", "sampledAnswerIds"])
    csvfile.close()
    
    
    # test on comm "coffee.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[166][0], commDir_sizes_sortedlist[166][1],root_dir)
    # test on comm "webapps.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[305][0], commDir_sizes_sortedlist[305][1],root_dir)
    
    # test on comm "politics.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[283][0], commDir_sizes_sortedlist[283][1],root_dir)
    # test on comm "philosophy.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[295][0], commDir_sizes_sortedlist[295][1],root_dir) 
    # test on comm "askubuntu" to debug
    # myFun(commDir_sizes_sortedlist[356][0], commDir_sizes_sortedlist[356][1],root_dir) 
    # test on comm "unix.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[353][0], commDir_sizes_sortedlist[353][1],root_dir) 
    # test on comm "math" to debug
    # myFun(commDir_sizes_sortedlist[358][0], commDir_sizes_sortedlist[358][1],root_dir) 
    # test on comm "mathoverflow.net" to debug
    # myFun(commDir_sizes_sortedlist[343][0], commDir_sizes_sortedlist[343][1],root_dir) 
    # test on comm "buddhism.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[273][0], commDir_sizes_sortedlist[273][1],root_dir) 
    # test on comm "judaism.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[309][0], commDir_sizes_sortedlist[309][1],root_dir) 
    # test on comm "hinduism.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[263][0], commDir_sizes_sortedlist[263][1],root_dir) 
    # test on comm "rpg.meta.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[204][0], commDir_sizes_sortedlist[204][1],root_dir) 
    # test on comm "stats.meta.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[170][0], commDir_sizes_sortedlist[170][1],root_dir) 
    # test on comm "cstheory.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[256][0], commDir_sizes_sortedlist[256][1],root_dir) 
    # test on comm "islam.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[270][0], commDir_sizes_sortedlist[270][1],root_dir) 
    # test on comm "webmasters.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[311][0], commDir_sizes_sortedlist[311][1],root_dir) 
    # test on comm "lifehacks.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[233][0], commDir_sizes_sortedlist[233][1],root_dir) 
    # test on comm "stats.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[347][0], commDir_sizes_sortedlist[347][1],root_dir) 
    # test on comm "tex.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[352][0], commDir_sizes_sortedlist[352][1],root_dir) 
    # test on comm "codegolf.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[333][0], commDir_sizes_sortedlist[333][1],root_dir) 
    # test on comm "writers.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[286][0], commDir_sizes_sortedlist[286][1],root_dir) 

    # test on comm "matheducators.stackexchange" to debug
    myFun(commDir_sizes_sortedlist[236][0], commDir_sizes_sortedlist[236][1],root_dir)
    # test on comm "cseducators.stackexchange" to debug
    myFun(commDir_sizes_sortedlist[185][0], commDir_sizes_sortedlist[185][1],root_dir)
    # test on comm "mathematica.stackexchange" to debug
    myFun(commDir_sizes_sortedlist[331][0], commDir_sizes_sortedlist[331][1],root_dir)
    # test on comm "bicycles.stackexchange" to debug
    myFun(commDir_sizes_sortedlist[300][0], commDir_sizes_sortedlist[300][1],root_dir)
    # test on comm "chess.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[258][0], commDir_sizes_sortedlist[258][1],root_dir)
    # test on comm "boardgames.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[276][0], commDir_sizes_sortedlist[276][1],root_dir)

    # test on comm "stackoverflow" to debug
    # myFun(commDir_sizes_sortedlist[359][0], commDir_sizes_sortedlist[359][1],root_dir)
    """
    # seleted communities
    selected_comms = ['matheducators.stackexchange','cseducators.stackexchange','mathematica.stackexchange','bicycles.stackexchange', 'chess.stackexchange','boardgames.stackexchange']
    selected_comms = ['stackoverflow']


    # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for commIndex, tup in enumerate(commDir_sizes_sortedlist):
        commName = tup[0]
        commDir = tup[1]

        # if commName not in topSelectedCommNames:
        #     print(f"{commName} is not seleted, skip")
        #     continue
        if commName not in selected_comms:
            # print(f"{commName} is not seleted, skip")
            continue

        # myFun(commName, commDir, root_dir)
        # finishedCount +=1
        # print(f"finished {finishedCount} comm.")
    
    
        try:
            p = mp.Process(target=myFun, args=(commName,commDir,root_dir))
            p.start()
        except Exception as e:
            print(e)
            pscount = sum(1 for proc in psutil.process_iter() if proc.name() == 'python3')
            print(f"current python3 processes count {pscount}.")
            sys.exit()

        processes.append(p)
        if len(processes)==24:
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
    print('descriptive 14  Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
