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
import time
# MNIST 데이터 경로
_SRC_PATH = u'mnist\\raw_binary'
_TRAIN_DATA_FILE = _SRC_PATH + u'\\train-images-idx3-ubyte'
_TRAIN_LABEL_FILE = _SRC_PATH + u'\\train-labels-idx1-ubyte'

# MNIST 데이터 크기 (28x28)
_N_ROW = 28                 # 세로 28픽셀
_N_COL = 28                 # 가로 28픽셀
_N_PIXEL = _N_ROW * _N_COL
_BATCH_SIZE= 100                      # batch 크기 
_N_BATCH = int(60000 / _BATCH_SIZE)    # batch 수 
_N_SAMPLE = 60000

learningrate = 5

def save(fn,obj):
    fd = open(fn,'wb')
    pkl.dump(obj,fd)
    fd.close()   
    
    
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

    # 학습 데이터 / 레이블 로드
    trDataList = loadData(_TRAIN_DATA_FILE)
    trLabelList = loadLabel(_TRAIN_LABEL_FILE)
    
    return trDataList, trLabelList

# logistic activation
def activation(z):
    return (1.0 / (1.0 + np.exp(-z))).astype('float32')

def mini_batch_training():

    fd = open("train_log.txt",'w')

    t = 0
    # w(t)랜덤 초기화
    w = np.zeros(0)
    #weight 28*28 + 1
    temp = np.zeros(_N_ROW*_N_COL+1)
    for i in range(0,10):
        for j in range(0,_N_ROW*_N_COL+1):
            temp[j] = random.uniform(-1.0,1.0)
        if(i == 0):
            w = temp
        else:
            w=np.vstack([w,temp])

    # 트레인 데이터의 배열 마지막에 1을 넣어서 bias설정
    bias = np.ones((_N_SAMPLE,1))
    dataSet = trDataList
    # 데이터를 계속 곱하면 숫자가 커져서 overflow 가능성이 있기때문에 정규화해준다
    # 데이터 값은 0~1이가 된다
    dataSet = np.hstack([dataSet,bias])/255.0

    # 학습 데이터 셋
    g_w = w.astype('float32')
    gg_w = w
    error_n = _N_SAMPLE

    # 학습 시작 
    for i in range(0,10000):
        print 'error num -> %d' % (error_n)
        fd.write('%dth error num -> %d\n' % (i,error_n))
        # 에러율이 20프로 미만이면 루프 탈출 
        if(error_n <= _N_SAMPLE*0.10): 
            return gg_w
            break
        error_n = 0
        w_gradient = np.zeros((10,785)).astype('float32')
        
        # batch 순서 섞기
        sampleIndexList = range(_N_SAMPLE)
        ra.shuffle(sampleIndexList)

        #batch size만큼 미니배치로 학습 
        for batchIndex in range(_N_BATCH):

            start = batchIndex * _BATCH_SIZE
            end = np.min([_N_SAMPLE,(batchIndex + 1) * _BATCH_SIZE])
            batchSampleIndexList = sampleIndexList[start:end]
            # 배치 샘플 선택 
           
            batchTrData = dataSet[batchSampleIndexList,:] 
            batchHot = np.zeros((10,_BATCH_SIZE)).astype('float32')

            #정답 리스트 만들기 
            for j in range(0,_BATCH_SIZE):
            	batchHot[trLabelList[batchSampleIndexList[j]]][j] = 1

            inner = np.zeros((10,_BATCH_SIZE)).astype('float32')
            o = np.zeros((10,_BATCH_SIZE)).astype('float32')

            # weight 와 traindata를 내적 
            # g_w -> 10*785 , batchTrData -> 10000*785
            # inner ->10*10000
            inner = np.dot(g_w,np.transpose(batchTrData))
            # 열마다 하나의 트레인 데이터가 들어있다 
            o = activation(inner)

            for j in range(0,_BATCH_SIZE):
                max_index = np.unravel_index(inner[:,j].argmax(),inner[:,j].shape)[0]
                # 출력함수로 계산했을때 제일 큰값과 실제 값을 비교해서 오류 검사 
                if (np.argmax(batchHot[:,j]) != max_index):
                	error_n +=1
            
            # 10 * 10000
            y = np.multiply((batchHot - o),o).astype('float32')
            # 10 * 10000
            one_array = np.ones((10,_BATCH_SIZE)).astype('float32')
            # 미분 계산 
            z = np.multiply((one_array-o),y).astype('float32')

            # (10*10000) * (10000*785)
            # 미분방향으로 가중치 학습 
            w_gradient = (learningrate*np.dot(z,batchTrData)/_BATCH_SIZE).astype('float32')
            gg_w = g_w
            g_w = g_w+w_gradient
    fd.close()
    return g_w   
    
if __name__ == '__main__':
    start = time.time()
    trDataList, trLabelList = loadMNIST()
    
    print 'len(trDataList)', len(trDataList)
    print 'len(trLabelList)', len(trLabelList)
    
    w=mini_batch_training()
    save('best_param.pkl',w)
    end = time.time() - start

    print 'time : ', end