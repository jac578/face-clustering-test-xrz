import json
import os
import numpy as np
import urllib2
from unpack_stream_to_float32 import unpack_feature_from_stream

def convert_json_file_to_npy(jsonFile):
    jsonDict = load_json_file(jsonFile)
    videoId = jsonDict['faces'][0]['videoId']
    videoPath = './'+videoId
    try:
        os.makedirs(videoPath)
    except:
        pass
    for individualDict in jsonDict['faces']:
        individual_to_npy(individualDict, videoPath)
    return videoPath

def downloadPic(picUrl, path):
    response = urllib2.urlopen(picUrl)
    pic = response.read()
    picName = picUrl.split('/')[-1]
    with open(path+'/'+picName,'w') as f:
        f.write(pic)
    return 

def load_json_file(filePath):
    with open(filePath, 'r') as f:
        jsonString = f.read()
        return json.loads(jsonString)

def code_feature_to_npy(jsonString):
    return unpack_feature_from_stream(jsonString)

def individual_to_npy(individualDict, videoPath):
    id = individualDict['id']
    individualPath = videoPath + '/' + id
    try:
        os.makedirs(individualPath)
    except:
        pass
    for singlePicDict in individualDict['features']:
        downloadPic(singlePicDict['faceUri'], individualPath)
        picName = singlePicDict['faceUri'].split('/')[-1] 
        npyFeature = code_feature_to_npy(singlePicDict['data'])
        npyFeature = np.asarray(npyFeature, dtype=np.float32)
        np.save(file=individualPath+'/'+picName, arr=npyFeature)
    return 


if __name__ == "__main__":
    convert_json_file_to_npy('video1.json')
    convert_json_file_to_npy('video2.json')
