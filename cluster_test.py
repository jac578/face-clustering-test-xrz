#coding=utf-8
from __future__ import division
from my_cluster import *

import argparse


def load_label(testsetDir):
    alone_cluster_label_cnt = 1000
    labelDict = {}
    for dir in os.listdir(testsetDir):
        if dir.startswith('.'):
            continue
        else:
            currentDir = os.path.join(testsetDir, dir)
            for fileName in os.listdir(currentDir):
                if fileName.endswith('.jpg'):
                    if str(dir) != '-1':
                        labelDict[fileName] = dir
                    else:
                        labelDict[fileName] = str(alone_cluster_label_cnt)
                        alone_cluster_label_cnt += 1
    return labelDict

def cluster_and_test_from_video_dir(videoDir, picDir, labelDict, methodList=['DBSCAN']):
    if methodList[0] == 'API':
        methodResultDict = {}
        methodResultDict['API'] = test_former_api(videoDir)
    else:    
        methodResultDict = cluster_from_video_dir(videoDir, picDir, methodList)
    for method in methodResultDict.keys():
        resultDict = methodResultDict[method]
        resultClusterDict = make_clusterDict_from_resultDict(resultDict)
        labelClusterDict = make_clusterDict_from_resultDict(labelDict)
        
        '''
        #calculate precision
        clusterPrecision = 0
        for key in resultClusterDict.keys():
            mostLabel, mostLabelNum = find_cluster_most_label(resultClusterDict[key], labelDict)
            currentClusterPrecision = mostLabelNum / len(resultClusterDict[key])
            clusterPrecision += currentClusterPrecision
        clusterPrecision = clusterPrecision / len(resultClusterDict)
        
        #calculate recall
        recallDict = {}
        for k,v in labelClusterDict.items():
            maxClusteredNum = find_max_clustered_num(k, resultClusterDict, labelDict)
            recallDict[k] = maxClusteredNum / len(labelClusterDict[k])
        
        return clusterPrecision, recallDict
        '''
        f_score = pairwise_f_score(resultClusterDict, labelClusterDict, labelDict)
        return f_score

def find_max_clustered_num(label, resultClusterDict, labelDict):
    maxNum = 0
    for k,cluster in resultClusterDict.items():
        num = count_label_in_cluster(label, cluster, labelDict)
        if num > maxNum:
            maxNum = num
    return maxNum

def count_label_in_cluster(label, cluster, labelDict):
    num = 0
    for x in cluster:
        if labelDict[x] == label:
            num += 1
    return num

def make_clusterDict_from_resultDict(resultDict):
    clusterDict = {}
    for key in resultDict.keys():
        if not clusterDict.has_key(resultDict[key]):
            clusterDict[resultDict[key]] = []
        clusterDict[resultDict[key]].append(key)
    return clusterDict

def find_cluster_most_label(listOfSameCluster, labelDict):
    _statLabelDict = {}
    for fileName in listOfSameCluster:
        if not _statLabelDict.has_key(labelDict[fileName]):
            _statLabelDict[labelDict[fileName]] = 1
        else:
            _statLabelDict[labelDict[fileName]] += 1
    mostLabel = -1
    mostLabelNum = -1
    for k,v in _statLabelDict.items():
        if v > mostLabelNum:
            mostLabelNum = v
            mostLabel = k
    return mostLabel, mostLabelNum

    
def test_former_api(videoDir):
    classCnt = 0
    resultDict = {}
    for dirName in os.listdir(videoDir):
        if str(dirName).startswith('.'):
            continue
        currentDir = os.path.join(videoDir, dirName)
        for fileName in os.listdir(currentDir):
            if fileName.endswith('.jpg'):
                resultDict[fileName] = str(classCnt)
        classCnt += 1
    return resultDict

def pairwise_f_score(resultClusterDict, labelClusterDict, labelDict):
    precision = pairwise_precision(resultClusterDict, labelDict) 
    recall = pairwise_recall(resultClusterDict, labelClusterDict, labelDict)
    f_score = 2 * precision * recall / (precision + recall)
    print precision, recall
    return f_score

def pairwise_precision(resultClusterDict, labelDict):
    above = 0
    below = 0
    for k, cluster in resultClusterDict.items():
        below += compute_combination(2, len(cluster))
        above += count_right_pair_in_result_cluster(cluster, labelDict)
    return above / below

def pairwise_recall(resultClusterDict, labelClusterDict, labelDict):
    above = 0
    below = 0
    for k, cluster in labelClusterDict.items():
        below += compute_combination(2, len(cluster))
    for k, cluster in resultClusterDict.items():
        above += count_right_pair_in_result_cluster(cluster, labelDict)
    return above / below

def count_right_pair_in_result_cluster(cluster, labelDict):
    num = 0
    cntDict = {}
    for x in cluster:
        if cntDict.has_key(labelDict[x]):
            cntDict[labelDict[x]] += 1
        else:
            cntDict[labelDict[x]] = 1
    
    for k, v in cntDict.items():
        num += compute_combination(2, v)
    return num

def compute_combination(upNum, downNum):
    above = 1
    below = 1
    for i in range(upNum):
        above *= downNum - i
        below *= upNum - i
    return above / below




if __name__ == '__main__':
    parser = argparse.ArgumentParser('Clustering and Pairwise F_score Evaluation')
    parser.add_argument('--method', type=str, required=True, help='DBSCAN, API, AP, RankOrder')
    parser.add_argument('--labelDir', type=str, required=False, default='test_set', help='Path of labeled pictures')
    parser.add_argument('--videoDir', type=str, required=True, help='Path of features to be clustered')
    parser.add_argument('--picDir', type=str, required=True, help='Path of pictures to be clustered')
    parser.add_argument('--saveResult', type=bool, required=True, help='Whether to save the result pics')
    args = vars(parser.parse_args())


    #labelDict = load_label(args['labelDir'])
    #f_score = cluster_and_test_from_video_dir('5ab52c0e28734100076d67b9', labelDict, methodList=['API'])
    #f_score = cluster_and_test_from_video_dir(args['videoDir'], args['picDir'], labelDict, methodList=[args['method']])
    #print f_score

    cluster_from_video_dir(args['videoDir'], args['picDir'], methodList=[args['method']], saveResult=args['saveResult'])

    '''
    clusterPrecision, recallDict = cluster_and_test_from_video_dir('5ab52c0e28734100076d67b9', labelDict, methodList=['DBSCAN'])
    print clusterPrecision
    meanRecall = 0
    for k,v in recallDict.items():
        meanRecall += v
    meanRecall /= len(recallDict)
    print meanRecall
    '''