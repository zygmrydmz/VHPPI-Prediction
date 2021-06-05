# -*- coding: utf-8 -*-

import numpy as np
import os
import glob
import math
import datetime
import csv
import pickle as cPickle
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

AminoAcids = ['A','G','V','C','F','I','L','P','M','S','T','Y','H','N','Q','W','D','E','K','R']
def CreateData(file1,file2,file3):
    label_num = []
    for line in open(file1):
        label = line.strip(',')
        label_num.append(int(label))
    feature1 = RFAT(file2)
    feature2 = FDAT(file2)
    feature3 = AC(file3)
    features = []
    for i in range(len(feature1)):
        feature = feature1[i] + feature2[i] + feature3[i]
        features.append(feature)
    return features,label_num

def triplet():
    triplet_feature = []
    list = ['1', '2', '3', '4', '5', '6', '7']
    base = len(list)  # 7
    end = len(list) ** 3
    for i in range(0, end):
        n = i
        ch0 = list[n % base]
        n = int(n / base)
        ch1 = list[n % base]
        n = int(n / base)
        ch2 = list[n % base]
        # print(i, ch3, ch2, ch1, ch0)
        triplet_feature.append(ch2 + ch1 + ch0)
        #f1.write(ch3 + ch2 + ch1 + ch0 + '\n')
    return triplet_feature

def RFAT(file):
    f1 = open(file,'r')
    triplet_feature = triplet()

    RFAT_features = []
    for line in f1:
        item = line.strip('\n').split(',')
        virus_seq = item[1]
        host_seq = item[2]
        host_f = []
        virus_f = []
        virus_q1 = []
        host_q1 = []
        for xxx in triplet_feature:
            xxx = ''.join(xxx)
            virus_f.append(virus_seq.count(xxx))
            host_f.append(host_seq.count(xxx))

        avg_vf = float(sum(virus_f))/len(virus_f)
        avg_hf = float(sum(host_f))/len(host_f)
        max_vf = float(max(virus_f))
        max_hf = float(max(host_f))
        for i in virus_f:
            virus_q = np.round(np.exp((i-avg_vf)/(max_vf-avg_vf)),10)
            #virus_q = np.round((i - min(virus_f)) / np.exp(max(virus_f) - min(virus_f)), 4)
            virus_q1.append(virus_q)
        for j in host_f:
            host_q = np.round(np.exp((j-avg_hf)/(max_hf-avg_hf)),10)
            #host_q = np.round((j - min(host_f)) / np.exp(max(host_f) - min(host_f)), 4)
            host_q1.append(host_q)

        virus_q1.extend(host_q1)
        RFAT_features.append(virus_q1)
    return RFAT_features

def FDAT(file):
    f1 = open(file,'r')
    triplet_feature = triplet()

    FDAT_features = []
    for line in f1:
        item = line.strip('\n').split(',')
        virus_seq = item[1]
        host_seq = item[2]
        f = []
        q1 = []
        for xxx in triplet_feature:
            xxx = ''.join(xxx)
            fv = virus_seq.count(xxx)
            fh = host_seq.count(xxx)
            f.append(abs(fv-fh))

        avg_FD = float(sum(f))/len(f)
        max_FD = float(max(f))

        for i in f:
            q = np.round(np.exp((i-avg_FD)/(max_FD-avg_FD)),10)
            q1.append(q)
        FDAT_features.append(q1)
    return FDAT_features

def AC(file):
    f1 = open(file, 'r')
    AC_features = []
    for line in f1:
        f = []
        q1 = []
        item = line.strip('\n').split(',')
        seq = item[0] + item[1]

        for aa in AminoAcids:
            aa = ''.join(aa)
            fa = seq.count(aa)
            f.append(fa)
        for i in f:
            q = i/float(max(f))
            q1.append(q)
        AC_features.append(q1)
    return AC_features