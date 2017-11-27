# -*- coding: utf-8 -*-

import os
import os.path as op
import struct as st
import numpy as np
import matplotlib.pyplot as plt
import random 
import random as ra
import math
import pickle as pkl
# MNIST 데이터 경로
_SRC_PATH = u'mnist\\raw_binary'
_TEST_DATA_FILE = _SRC_PATH + u'\\t10k-images-idx3-ubyte'
_TEST_LABEL_FILE = _SRC_PATH + u'\\t10k-labels-idx1-ubyte'


# MNIST 데이터 크기 (28x28)
_N_ROW = 28                 # 세로 28픽셀
_N_COL = 28                 # 가로 28픽셀
_N_PIXEL = _N_ROW * _N_COL
_N_SAMPLE = 10000

    
def loadData(fn):
    print 'loadData', fn
    
    fd = open(fn, 'rb')
    
    # header: 32bit integer (big-endian)
    magicNumber = st.unpack('>I', fd.read(4))[0]
    nData = st.unpack('>I', fd.read(4))[0]
    nRow = st.unpack('>I', fd.read(4))[0]
    nCol = st.unpack('>I', fd.read(4))[0]
    
    print 'magicNumber', magicNumber
    print 'nData', nData
    print 'nRow', nRow
    print 'nCol', nCol
    
    # data: unsigned byte
    dataList = []
    for i in range(nData):
        dataRawList = fd.read(_N_PIXEL)
        dataNumList = st.unpack('B' * _N_PIXEL, dataRawList)
        dataArr = np.array(dataNumList)
        dataList.append(dataArr.astype('int32'))
        
    fd.close()
    
    print 'done.'
    print
    
    return dataList
    


def loadLabel(fn):
    print 'loadLabel', fn
    
    fd = open(fn, 'rb')
    
    # header: 32bit integer (big-endian)
    magicNumber = st.unpack('>I', fd.read(4))[0]
    nData = st.unpack('>I', fd.read(4))[0]
    
    print 'magicNumber', magicNumber
    print 'nData', nData
    
    # data: unsigned byte
    labelList = []
    for i in range(nData):
        dataLabel = st.unpack('B', fd.read(1))[0]
        labelList.append(dataLabel)
        
    fd.close()
    
    print 'done.'
    print
    
    return labelList

def loadMNIST():
    # 테스트 데이터 / 레이블 로드
    tsDataList = loadData(_TEST_DATA_FILE)
    tsLabelList = loadLabel(_TEST_LABEL_FILE)
    
    return tsDataList, tsLabelList

def load(fn):
    fd = open(fn,'rb')
    obj = pkl.load(fd)
    fd.close()
    return obj

def test(param):
    
    fd = open('test_output.txt','w')

    # 트레인 데이터의 배열 마지막에 1을 넣어서 bias설정
    bias = np.ones((_N_SAMPLE,1))
    dataSet = tsDataList
    # 데이터를 계속 곱하면 숫자가 커져서 overflow 가능성이 있기때문에 정규화해준다
    # 데이터 값은 0~1이가 된다
    dataSet = np.hstack([dataSet,bias])/255.0
    
    # 정답을 판별하는 label을 만듦
    labelSet = np.zeros((10,10)).astype('float32')
    for i in range(0,10):
        labelSet[i][i]=1.

    error_n = 0

    # 테스트 시작 
    inner = np.zeros((10,_N_SAMPLE)).astype('float32')
    o = np.zeros((10,_N_SAMPLE)).astype('float32')
        # weigt 와 traindata를 내적 
        # g_w -> 10*785 , batchTrData -> 10000*785
        # inner ->10*10000
    inner = np.dot(param,np.transpose(dataSet))

        # activation 값이 제일 큰 인덱스에 1을 넣어준다
    for j in range(0,_N_SAMPLE):
        max_index = np.unravel_index(inner[:,j].argmax(),inner[:,j].shape)[0]
         
        if (tsLabelList[j] != max_index):
            text ='%d%s%d%s%d%s' % (j,' th original test data-> ',tsLabelList[j],' learn test data-> ',max_index,' ==> false!\n')                
            fd.write(text)
            error_n +=1
        else:
            text ='%d%s%d%s%d%s' % (j,' th original test data-> ',tsLabelList[j],' learn test data-> ',max_index,' ==> correct!\n')                
            fd.write(text)

    fd.write('%s%d%s' % ('total -> ',error_n, '/10000\n'));
    fd.close()
    return error_n  
    
if __name__ == '__main__':
    tsDataList, tsLabelList = loadMNIST()
    
    print 'len(tsDataList)', len(tsDataList)
    print 'len(tsLabelList)', len(tsLabelList)
    
    
    loadedParam = load('best_param.pkL')
    test(loadedParam)