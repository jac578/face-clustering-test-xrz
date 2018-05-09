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

def cluster_and_test_from_video_dir(videoDir, featureList, picDir, methodList=['DBSCAN'], labelDict=None, eps=0.5, nProcess=1):
    if methodList[0] == 'API':
        epsResultDict = {}
        epsResultDict['API'] = test_former_api(videoDir)
    else:    
        epsResultDict = cluster_from_video_dir(videoDir, featureList, picDir, methodList, saveResult=False, eps=eps, nProcess=nProcess)
    for paraEps in epsResultDict.keys():
        resultDict = epsResultDict[paraEps]
        resultClusterDict = make_clusterDict_from_resultDict(resultDict)
        labelClusterDict = make_clusterDict_from_resultDict(labelDict)
        
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
    parser.add_argument('--featureList', type=str, required=True, help='Feature list of feature file name')
    parser.add_argument('--picDir', type=str, required=True, help='Path of pictures to be clustered')
    parser.add_argument('--saveResult', type=bool, required=True, help='Whether to save the result pics')
    parser.add_argument('--saveDir', type=str, required=True, help='Path to save clustered pictures')
    parser.add_argument('--eps', type=float, required=False, default=None, help='DBSCAN parameter')
    parser.add_argument('--nProcess', type=int, required=False, default=1, help='Number of processes to read data')
    parser.add_argument('--evaluate', type=bool, required=False, default=False, help='Do you need evaluation over list of eps?')
    args = vars(parser.parse_args())


    #labelDict = load_label(args['labelDir'])
    #f_score = cluster_and_test_from_video_dir('5ab52c0e28734100076d67b9', labelDict, methodList=['API'])
    #f_score = cluster_and_test_from_video_dir(args['videoDir'], args['picDir'], labelDict, methodList=[args['method']])
    #print f_score
    if args['eps'] is None:
        eps = ''
    else:
        eps = str(args['eps'])

    saveDir = args['saveDir']+'_'+eps
 
    if args['evaluate']:
        print "Will save result"
        saveDir = args['saveDir']+'_'+eps
        cluster_and_test_from_video_dir(args['videoDir'], args['featureList'], args['picDir'], methodList=[args['method']], 
                        eps=args['eps'], nProcess=args['nProcess'])
    else:
        print "Will Evaluate"
        cluster_from_video_dir(args['videoDir'], args['featureList'], args['picDir'], methodList=[args['method']], 
                        saveResult=args['saveResult'], saveDir=saveDir, eps=args['eps'], nProcess=args['nProcess'])
        print args['eps']
        os.system("ls -lR {}|grep \"^-\"|wc -l".format(saveDir))
        os.system("ls {}|wc -l".format(saveDir))
    
    