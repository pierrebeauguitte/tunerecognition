import matplotlib.pyplot as plt
import numpy as np
import glob
import csv
import json
import sys
import argparse
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

def cleanFilename(filename):
    return os.path.splitext(os.path.basename(filename))[0]

def prepareSets(path):

    # x = [] # quantized peaks
    # t = [] # tune index

    t = {}

    dataFiles = glob.glob('%s/*.json' % path)
    for tune in dataFiles:
        x = []
        tuneIndex = cleanFilename(tune)
        fp = open(tune, 'r')
        data = json.load(fp)
        ts = map(int, data.keys())
        ts.sort()
        for a in ts:
            x.append(data[str(a)])
            # t.append(tuneIndex)
        t[tuneIndex] = x

    return t

def predictAll(path, modelType, outfile):

    nFolds = 4 if modelType == 'multinomial' else 10

    folds = {}
    with open('folds_%s.csv' % modelType, 'r') as infile:
        reader = csv.reader(infile)
        for row in reader:
            folds[int(row[0])] = int(row[1])

    models = []
    for i in range(nFolds):
        with open('models/%s_%d.pcl' % (modelType, i), 'r') as modFile:
            models.append(pickle.load(modFile))

    data = prepareSets(path)

    for chunk in data:
        fold = folds[int(chunk[:3])]
        model = models[fold]
        pred = model.predict(data[chunk])
        prob = model.predict_proba(data[chunk])
        toPrint = chunk
        avgProbs = []
        for typeIndex in range(len(model.classes_)):
            avgProbs.append(np.average(prob[:, typeIndex]))
            toPrint += ' %.5f' % np.average(prob[:, typeIndex])
        toPrint += ' %d\n' % (np.argmax(avgProbs))
        outfile.write(toPrint)

def predictVal(path, modelType, outfile):

    nFolds = 4 if modelType == 'multinomial' else 10

    folds = {}
    with open('folds_thr_%s.csv' % modelType, 'r') as infile:
        reader = csv.reader(infile)
        for row in reader:
            folds[int(row[0])] = int(row[1])

    models = []
    for i in range(nFolds):
        with open('models/%s_%d_thr.pcl' % (modelType, i), 'r') as modFile:
            models.append(pickle.load(modFile))

    data = prepareSets(path)

    for chunk in data:
        if int(chunk[:3]) not in folds:
            continue
        fold = folds[int(chunk[:3])]
        model = models[fold]
        pred = model.predict(data[chunk])
        prob = model.predict_proba(data[chunk])
        toPrint = chunk
        avgProbs = []
        for typeIndex in range(len(model.classes_)):
            avgProbs.append(np.average(prob[:, typeIndex]))
            toPrint += ' %.5f' % np.average(prob[:, typeIndex])
        toPrint += ' %d\n' % (np.argmax(avgProbs))
        outfile.write(toPrint)


def predictFinal(path, modelType, outfile):

    trainVal = []
    with open('folds_thr_%s.csv' % modelType, 'r') as infile:
        reader = csv.reader(infile)
        for row in reader:
            trainVal.append(int(row[0]))

    data = prepareSets(path)

    with open('models/%s_final.pcl' % modelType, 'r') as modFile:
        model = pickle.load(modFile)

    for chunk in data:
        if int(chunk[:3]) in trainVal:
            continue
        pred = model.predict(data[chunk])
        prob = model.predict_proba(data[chunk])
        toPrint = chunk
        avgProbs = []
        for typeIndex in range(len(model.classes_)):
            avgProbs.append(np.average(prob[:, typeIndex]))
            toPrint += ' %.5f' % np.average(prob[:, typeIndex])
        toPrint += ' %d\n' % (np.argmax(avgProbs))
        outfile.write(toPrint)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--modelType', required=True,
                        help='type of model to train (binomial/multinomial)')
    args = parser.parse_args()

    if args.modelType != 'binomial' and args.modelType != 'multinomial':
        print 'modelType should be \'binomial\' or \'multinomial\''
        sys.exit(1)

    with open('pred_%s.csv' % args.modelType, 'w') as outfile:
        predictAll('data_chunks', args.modelType, outfile)

    with open('pred_val_%s.csv' % args.modelType, 'w') as outfile:
        predictVal('data_chunks', args.modelType, outfile)

    with open('pred_test_%s.csv' % args.modelType, 'w') as outfile:
        predictFinal('data_chunks', args.modelType, outfile)
