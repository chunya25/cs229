#!/usr/bin/env python
import csv
import json
import math
import os
import re
import sys
import numpy as np
from sets import Set
from copy import deepcopy
from collections import defaultdict
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
from sklearn.decomposition import PCA

if __name__ == '__main__':
    track_id = [i.strip().split() for i in open("shuffled_valid_track_ids.csv").readlines()]
    genre_id = [i.strip().split() for i in open("shuffled_valid_track_top_level.csv").readlines()]
    top_ids = {2 : [], 3 : [], 4 : [],
               5 : [], 8 : [], 9 : [], 10 : [], 12 : [], 13 : [],
               14 : [], 15 : [], 17 : [], 20 : [], 21 : [], 38 : [], 1235 : []}; 
    examples = deepcopy(top_ids);
    k = 1000 # Number of examples to pick per genre (split into 70% train, 15% dev, 15% test)
    map_selected_examples = Set([]);
    genre_id_map = {}
    print '############## GETTING K TEST EXAMPLES PER GENRE ############'
    for i in range(0, len(track_id)):
        # If we have more than k examples, we are done for this genre
        genre_i = int(genre_id[i][0])
        genre_id_map[i] = genre_i
        if (genre_i is not 0):
            arr = top_ids[genre_i];
            if arr.__len__() < k:
                # Append the track as example for genre id
               id_of_track = float(track_id[i][0])
               top_ids[genre_i].append(id_of_track)
               map_selected_examples.add(id_of_track);
    #for key, value in top_ids.iteritems():
    #    print key, value
    #print map_selected_examples
    csvfile = open('features.csv','rb')
    feature = None
    statistics = None
    feature_num = None
    # empty line
    print '############## RETRIEVING EXAMPLES ############'
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    GENRE_ID_MISSING = 0;
    TOTAL_EXAMPLES = k * 16;
    NUM_EXAMPLES_SO_FAR = 0;
    with open('features.csv', 'rb') as f:
        reader = csv.reader(f)
        features = next(reader)
        statistics = next(reader)
        feature_num = next(reader)
        # empty line
        next(reader)
        for i in range(0, len(track_id)):
          if NUM_EXAMPLES_SO_FAR > TOTAL_EXAMPLES:
            break;
          feature_set = next(reader)
          tr_id = int(feature_set[0]);
          if tr_id in map_selected_examples and (tr_id in genre_id_map) and genre_id_map[tr_id] is not 0:
            genre_label = float(genre_id[tr_id][0])
            feature_vector_i = [float(x) for x in feature_set[1:]]
            examples[genre_label].append(feature_vector_i)
            # We want a percentage for training and some for testing
            NUM_EXAMPLES_SO_FAR +=1;
            if len(examples[genre_label]) < (0.85 * k):
                X_train.append(feature_vector_i)
                y_train.append(genre_label)
            else:
                X_test.append(feature_vector_i)
                y_test.append(genre_label)
          else:
            #print (i , " * ", int(feature_set[0]))
            GENRE_ID_MISSING += 1;
    print 'READ INPUT: GENRE_ID_MISSING for', GENRE_ID_MISSING, ' examples'
    print '############## TRAINING ############'
    X = X_train#np.array(X_train);
    y = y_train#np.array(y_train)

    print '*** Training KNeighborsClassifier ****'
    NUM_COMPONENTS = 20# [20, 30, 40, 70, 100, 200]
    #for k in NUM_COMPONENTS:
    pca_X_train = PCA(n_components=NUM_COMPONENTS, whiten=True)
    pca_X_test = PCA(n_components=NUM_COMPONENTS, whiten=True) 
    pca_X_train.fit(preprocessing.scale(X_train))
    pca_X_test.fit(preprocessing.scale(X_test))
    X_train = pca_X_train.transform(X_train) #preprocessing.scale(X_train)
    X_test = pca_X_test.transform(X_test)  #preprocessing.scale(X_test)
    X = X_train#np.array(X_train);
    y = y_train#np.array(y_train)
    print("SHAPE=", np.shape(X_train))
    neigh = KNeighborsClassifier(n_neighbors=30, weights='distance')
    neigh.fit(X, y)
    predicted = neigh.predict(np.array(X_test))
    correct = 0
    for i in range(0, len(predicted)):
        if predicted[i] == y_test[i]:
            correct += 1
    print 'correct =',correct, ' out of ', len(predicted)
    print("KNeighborsClassifier Test Accuracy: ", float(correct)/len(predicted) * 100, "%")  


    print '*** Training NearestCentroid ****'
    clf = NearestCentroid()
    clf.fit(X, y)
    NearestCentroid(metric='minkowski', shrink_threshold=None)
    predicted = clf.predict(np.array(X_test))
    correct = 0
    for i in range(0, len(predicted)):
        if predicted[i] == y_test[i]:
            correct += 1
    print 'correct =',correct, ' out of ', len(predicted)
    print("NearestCentroid Test Accuracy: ", float(correct)/len(predicted) * 100, "%")   

    predicted = clf.predict(np.array(X_train))
    correct = 0
    for i in range(0, len(predicted)):
        if predicted[i] == y_train[i]:
            correct += 1
    print 'correct=',correct, ' out of ', len(predicted)
    print("NearestCentroid Training Accuracy: ", float(correct)/len(predicted) * 100, "%")