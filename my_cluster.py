#coding=utf-8
from sklearn import cluster
from scipy.spatial.distance import cosine as ssd_cosine_dist
import numpy as np
from matplotlib import pyplot as plt
import scipy.misc as sc

import time
import os
import shutil

import urllib2
from parse_json import convert_json_file_to_npy
from demo_noLFW import rankOrder_cluster_format


def feature_data_reader(dataPath):
    feature_list = None
    global_pic = None
    filePathList = []
    for dirName in os.listdir(dataPath):
        if str(dirName).startswith('.'):
            continue
        currentDir = os.path.join(dataPath, dirName)
        for fileName in os.listdir(currentDir):    
            if str(fileName).endswith('.npy'):
                fileFullPath = os.path.join(currentDir, fileName)
                featureVec = np.load(fileFullPath)
                if feature_list is None:
                    #第一个矩阵就错的时候会bug
                    #这里简单处理一下这种错误
                    if featureVec.shape[0]<10:
                        print "feature at head error"
                        continue

                    #初始化feature_list
                    feature_list = featureVec
                    filePathList.append(fileFullPath)
                else:
                    try:
                        # append feature_list
                        feature_list = np.vstack((feature_list, featureVec))
                        filePathList.append(fileFullPath)

                    except:
                        # if the feature stack above errors neglect current pic and feature
                        pass
                
    return np.asarray(feature_list), global_pic, filePathList 

def cluster_face_features(feature_list, method=None, precomputed=True, eps=0.5):
    if feature_list is not None:
        face_feature_list = feature_list

    
    if face_feature_list is None:
        return None
        
    if precomputed:
        metric_type = 'precomputed'
        dist_matrix = __compute_pairwise_distance(face_feature_list)
        dist_matrix = dist_matrix
    else:
        metric_type = 'euclidean'
        dist_matrix = np.vstack(face_feature_list)
        dist_matrix = None  
    
    if method == 'AP':
        cluster_estimator = cluster.AffinityPropagation(affinity=metric_type, damping=.55, preference=-1) 
        if precomputed:
            dist_matrix = -dist_matrix
    elif method == 'DBSCAN':
        cluster_estimator = cluster.DBSCAN(metric=metric_type, eps=eps, min_samples=2)

    t0 = time.time()
    cluster_estimator.fit(dist_matrix)
    t1 = time.time()
    
    t = t1 - t0
    print 'Clustering takes: %f seconds' % t
    
    if hasattr(cluster_estimator, 'labels_'):
        y_pred = cluster_estimator.labels_.astype(np.int)
    else:
        y_pred = cluster_estimator.predict(dist_matrix)
                
    return y_pred

def __compute_pairwise_distance(face_feature_list):
    nsamples = len(face_feature_list)
    assert(nsamples > 0)
    dist_matrix = 1 - np.dot(face_feature_list, face_feature_list.T)
    return dist_matrix

def my_cluster(videoDir, picDir, method, saveResult=False, saveDir='result', eps=0.5, **kwargs):

    resultDict = {}

    feature_list, global_pic, filePathList = feature_data_reader(videoDir)
    if method == 'RankOrder':
        if (kwargs.has_key('RO_n_neighbors')) and (kwargs.has_key('RO_thresh')):
            y_pred = rankOrder_cluster_format(feature_list, kwargs['RO_n_neighbors'], kwargs['RO_thresh'])
        else:
            print 'in else'
            y_pred = rankOrder_cluster_format(feature_list)
    else:
        y_pred = cluster_face_features(feature_list=feature_list, method=method, eps=eps)
    if saveResult:
        #saveDirPrefix = 'result_' + method + videoDir.replace('./', '')
        saveDirPrefix = saveDir
        for i in range(len(y_pred)):
            classDir = saveDirPrefix+'/'+str(y_pred[i])+'/'
            try:
                os.makedirs(classDir)
            except:
                pass
            picName = filePathList[i].replace('.npy', '.jpg').split('/')[-1]
            if picName.startswith('/'):
                picName = picName[1:]
            picPath = os.path.join(picDir, picName)
            
            shutil.copyfile(picPath, classDir+picName)
        #shutil.copytree(saveDirPrefix, os.path.join(videoDir, saveDirPrefix))

    assert len(y_pred) == len(filePathList)
    for i in range(len(y_pred)):
        resultDict[filePathList[i].split('/')[-1].replace('.npy', '')] = y_pred[i]
    return resultDict

def cluster_from_video_dir(videoDir, picDir, methodList=['DBSCAN'], saveResult=False, saveDir='result', eps=0.5):
    methodResultDict = {}
    for method in methodList:
        t0 = time.time()
        print "method: " + method
        print "start time: ", t0
        
        methodResultDict[method] = my_cluster(videoDir, picDir, method, saveResult, saveDir, eps)
        t1 = time.time()
        print "end time: ", t1
        print "time cost: ", t1-t0
    return methodResultDict

def download_json(httpLink):
    strHtml = urllib2.urlopen(httpLink).read()
    with open('extraSample1.json', 'w') as f:
        f.write(strHtml)
    return 'extraSample1.json'

def cluster_from_httpLink(httpLink):
    jsonFile = download_json(httpLink)
    videoDir = convert_json_file_to_npy(jsonFile)
    #cluster_from_video_dir(videoDir=videoDir)

def cluster_from_httpLinkList(httpLinkList):
    for httpLink in httpLinkList:
        cluster_from_httpLink(httpLink)

if __name__ == "__main__":
    httpLinkList = ['http://100.100.62.235:8000/v1/videos/5aba6d7a4d7eac0007611734/faces',
                    #'http://100.100.62.235:8000/v1/videos/5aba6d7a4d7eac0007611736/faces',
                    #'http://100.100.62.235:8000/v1/videos/5aba6da14d7eac0007611738/faces',
                    #'http://100.100.62.235:8000/v1/videos/5aba6da94d7eac000761173a/faces',
                    ]
    #cluster_from_httpLinkList(httpLinkList)
    #cluster_from_video_dir(videoDir='5ab52c0e28734100076d67b9', methodList=['DBSCAN'])


