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

import multiprocessing


def read_txtlist(sourceDir, path):
    pathList = []
    with open(path, 'r') as f:
        aPath = f.readline().strip()
        while aPath:
            if aPath.startswith('/'):
                aPath = aPath[1:]
            pathList.append(os.path.join(sourceDir, aPath))
            aPath = f.readline().strip()
    return pathList

def feature_data_reader(dataPath, featureList):
    feature_list = None
    global_pic = None
    filePathList = read_txtlist(dataPath, featureList)
    #Use first one to initialize
    feature_list = np.load(filePathList[0])
    #Concat else
    cnt = 0
    for fileFullPath in filePathList[1:]:    
        cnt += 1 
        print cnt
        featureVec = np.load(fileFullPath)
        try:
            feature_list = np.vstack((feature_list, featureVec))
        except:
            print feature_list.shape, featureVec.shape, fileFullPath
    return np.asarray(feature_list), global_pic, filePathList 

def feature_data_reader_fromList(filePathList):
    name = multiprocessing.current_process().name
    #Use first one to initialize
    feature_list = np.load(filePathList[0])
    #Concat else
    cnt = -1
    noHeadFilePathList = filePathList[1:]
    for fileFullPath in noHeadFilePathList:    
        cnt += 1
        if cnt%1000 == 0:
            print "Process", name, "done concating", cnt
        featureVec = np.load(fileFullPath)
        try:
            feature_list = np.vstack((feature_list, featureVec))
        except:
            noHeadFilePathList.pop(cnt)
            cnt -= 1
            print feature_list.shape, featureVec.shape, fileFullPath
    return np.asarray(feature_list)

def multiprocess_feature_data_reader(dataPath, featureList, nProcess=1):
    if nProcess == 1:
        return feature_data_reader(dataPath, featureList)
    else:
        feature_list = None
        global_pic = None
        filePathList = read_txtlist(dataPath, featureList)
        total_line = len(filePathList)
        p = multiprocessing.Pool(nProcess)
        pos = 0
        step = total_line / nProcess + 1
        resList = []
        for i in range(nProcess):
            if i == nProcess - 1:
                resList.append(p.apply_async(feature_data_reader_fromList,args=(filePathList[pos:],)))
            else: 
                resList.append(p.apply_async(feature_data_reader_fromList,args=(filePathList[pos:pos+step],)))
                pos += step
        p.close()
        p.join()
        for i in range(nProcess):
            if i == 0:  
                feature_list = resList[i].get()
            else:
                feature_list = np.vstack((feature_list, resList[i].get()))
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

def my_cluster(videoDir, featureList, picDir, method, saveResult=False, saveDir='result', eps=0.5, nReaderProcess=1, nClusterProcess=1, **kwargs):

    resultDict = {}
    t0 = time.time()
    print "Start loading data: ", t0
    #feature_list, global_pic, filePathList = feature_data_reader(videoDir, featureList)
    feature_list, global_pic, filePathList = multiprocess_feature_data_reader(videoDir, featureList, nReaderProcess)

    t1 = time.time()
    print "Done loading data. Start clustering: ", t1, "Loading data time cost: ", t1 - t0

    # pool = multiprocessing.Pool(nClusterProcess)
    # pos = 0
    # total = 
    # step = 

    # pos = 0
    #     step = total_line / nProcess + 1
    #     resList = []
    #     for i in range(nProcess):
    #         if i == nProcess - 1:
    #             resList.append(p.apply_async(feature_data_reader_fromList,args=(filePathList[pos:],)))
    #         else: 
    #             resList.append(p.apply_async(feature_data_reader_fromList,args=(filePathList[pos:pos+step],)))
    #             pos += step
    #     p.close()
    #     p.join()
    #     for i in range(nProcess):
    #         if i == 0:  
    #             feature_list = resList[i].get()
    #         else:
    #             feature_list = np.vstack((feature_list, resList[i].get()))

    my_cluster_after_read(feature_list, filePathList, picDir, method, saveResult, saveDir, eps)
    '''
    if method == 'RankOrder':
        if (kwargs.has_key('RO_n_neighbors')) and (kwargs.has_key('RO_thresh')):
            y_pred = rankOrder_cluster_format(feature_list, kwargs['RO_n_neighbors'], kwargs['RO_thresh'])
        else:
            print 'in else'
            y_pred = rankOrder_cluster_format(feature_list)
    else:
        y_pred = cluster_face_features(feature_list=feature_list, method=method, eps=eps)

    t2 = time.time()
    print "Done clustering. Start copying result: ", t2, "Clustering time cost", t2 - t1

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
    '''
    

    # assert len(y_pred) == len(filePathList)
    # for i in range(len(y_pred)):
    #     resultDict[filePathList[i].split('/')[-1].replace('.npy', '')] = y_pred[i]
    return #resultDict

def my_cluster_after_read(feature_list, filePathList, picDir, method, saveResult=False, saveDir='result', eps=0.5):
    t1 = time.time()
    y_pred = cluster_face_features(feature_list=feature_list, method=method, eps=eps)

    t2 = time.time()
    print "Done clustering. Start copying result: ", t2, "Clustering time cost", t2 - t1

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
    t3 = time.time()
    print "Done copying: ", t3, "Copying time cost", t3 - t2
    print "Exiting..."
    exit(0)

    return


def cluster_from_video_dir(videoDir, featureList, picDir, methodList=['DBSCAN'], saveResult=False, saveDir='result', eps=0.5, nProcess=1):
    methodResultDict = {}
    for method in methodList:
        t0 = time.time()
        print "method: " + method
        print "start time: ", t0
        
        methodResultDict[method] = my_cluster(videoDir, featureList, picDir, method, saveResult, saveDir, eps, nProcess)
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


