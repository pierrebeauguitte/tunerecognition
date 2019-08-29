import matplotlib.pyplot as plt
import numpy as np
import glob
import csv
import json
import sys
import argparse
import os
import pickle
from math import sqrt, log
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, brier_score_loss, balanced_accuracy_score
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

def printConfMat(cm, tunecm, labels, transpose=False, normalise=False):

    confMat = cm.T if transpose else cm
    tuneConfMat = tunecm.T if transpose else tunecm

    totalPerClass = np.sum(confMat, axis = 0, dtype = np.float32)
    if normalise:
        confMat /= totalPerClass

    tunePerClass = np.sum(tuneConfMat, axis = 0, dtype = np.float32)

    latex = '\\begin{tabular}{l'
    for _ in labels:
        latex += 'c'
    latex += '}\n\\toprule\n '

    for l in range(len(labels)):
        latex += '& %s ' % labels[l]
    latex += '\\\\\n\\midrule\n'

    for l in range(len(labels)):
        latex += labels[l]
        for i in confMat[l]:
            if normalise:
                latex += ' & %.2f' % (i * 100)
            else:
                latex += ' & %d' % i
        latex += '\\\\\n'

    latex += '\\midrule\n'

    for l in range(len(labels)):
        latex += labels[l]
        for i in tuneConfMat[l]:
            latex += ' & %d' % i
        latex += '\\\\\n'
    latex += '\\midrule\n'
    latex += 'accuracy (\\%)'
    for l in range(len(labels)):
        latex += ' & %.2f' % (100 * tuneConfMat[l,l] / tunePerClass[l])
    latex += '\\\\\n'

    latex += '\\bottomrule\n'
    latex += '\\end{tabular}'

    return latex


def cleanFilename(filename):
    return os.path.splitext(os.path.basename(filename))[0]

def loadReference(referenceFile, modelType):

    tunes = {}
    perID = {}

    binary_types = {
        '(hop) jig': 'compound',
        'jig': 'compound',
        'waltz': 'simple',
        'fling': 'simple',
        'slip jig': 'compound',
        'polka': 'simple',
        'barndance': 'simple',
        'slide': 'compound',
        'hornpipe': 'simple',
        'mazurka': 'simple',
        'highland': 'simple',
        'reel': 'simple'
    }

    multi_types = {
        '(hop) jig': 'slipjig',
        'jig': 'jig',
        'waltz': 'waltz',
        'fling': 'other44',
        'slip jig': 'slipjig',
        'polka': 'polka',
        'barndance': 'other44',
        'slide': 'slide',
        'hornpipe': 'hornpipe',
        'mazurka': 'waltz',
        'highland': 'other44',
        'reel': 'reel'
    }

    with open(referenceFile, 'r') as ref:
        reader = csv.DictReader(ref)
        for row in reader:
            tuneIndex = int(row['index'])
            tuneType = multi_types[row['type']] if modelType == 'multinomial' \
                       else binary_types[row['type']]
            if tuneType not in tunes:
                tunes[tuneType] = []
            tunes[tuneType].append(tuneIndex)
            perID[tuneIndex] = tuneType

    return tunes, perID

def prepareSets(path, tuneDict):

    x = [] # quantized peaks
    y = [] # label
    t = [] # tune index
    minNWindows = None

    dataFiles = glob.glob('%s/*.json' % path)
    for tune in dataFiles:
        tuneIndex = int(cleanFilename(tune)[:3])
        if tuneIndex not in tuneDict:
            continue
        target = tuneDict[tuneIndex]
        fp = open(tune, 'r')
        data = json.load(fp)
        ts = map(int, data.keys())
        ts.sort()
        if len(ts) < minNWindows or minNWindows is None:
            minNWindows = len(ts)
        for a in ts:
            x.append(data[str(a)])
            y.append(target)
            t.append(tuneIndex)

    return x, y, t, minNWindows

def makeSplits(instances, nFolds):
    splits = {i:[] for i in instances}
    for tuneType in instances:
        tuneIds = np.array(instances[tuneType])
        shuffle = np.random.permutation(len(tuneIds))
        splitSize = len(tuneIds) / float(nFolds)
        for i in range(nFolds - 1):
            splits[tuneType].append(
                list(tuneIds[ shuffle[int(np.round(i*splitSize)) :
                                      int(np.round((i+1)*splitSize))] ])
            )
        splits[tuneType].append(
            list(tuneIds[ shuffle[int(np.round((nFolds-1)*splitSize)):] ])
        )
    return splits

def doSplits():

    if os.path.isfile('folds_trainval_test.csv'):
        print 'train/test splits already done, skipping...'
        return

    t1, ref1 = loadReference('../dataset.csv', 'binomial')
    t2, ref2 = loadReference('../dataset.csv', 'multinomial')
    splits = makeSplits(t2, 2)

    l1, l2 = 0, 0
    for t, l in splits.iteritems():
        print t, len(l[0]), len(l[1])
        l1 += len(l[0])
        l2 += len(l[1])
    print l1, l2

    folds = {}
    for a in splits:
        for b in range(2):
            for c in splits[a][b]:
                folds[c] = b

    with open('folds_trainval_test.csv', 'w') as outfile:
        for i in range(1, 501):
            outfile.write('%03d,%d\n' % (i, folds[i]))

def getScores(fy, fp, ft, wLen):
    tindices = np.array(ft)
    indices = set(ft)
    globalRes = []
    for i in indices:
        loc = np.where(tindices == i)[0]
        res = []
        for w in range(len(loc) - wLen + 1):
            score = 0
            avgProbs = []
            for t in range(wLen):
                if (fy[loc[w+t]] == fp[loc[w+t]]):
                    score += 1
            score /= float(wLen)
            res.append( 1 if (score > 0.5) else 0 )
        globalRes.extend(res)
    return globalRes

def getScores2(fy, fp, ft, wLen, model):
    tindices = np.array(ft)
    indices = set(ft)
    globalRes = []
    for i in indices:
        loc = np.where(tindices == i)[0]
        res = []
        for w in range(len(loc) - wLen + 1):
            avgProbs = []
            for typeIndex in range(len(model.classes_)):
                avgProbs.append(np.average(fp[loc[w:w+wLen], typeIndex]))
            spanProb = model.classes_[np.argmax(avgProbs)]
            res.append(1 if spanProb == fy[loc[w]] else 0)
        globalRes.extend(res)
    return globalRes

def prepareTrainAndTest(path, referenceFile, modelType):

    print '\n---- TRAINING ON FULL DATASET ----\n'

    nFolds = 4 if modelType == 'multinomial' else 10

    labels = [
        'reel',
        'jig',
        'slide',
        'slipjig',
        'hornpipe',
        'polka',
        'other44',
        'waltz'
    ] if modelType == 'multinomial' else ['simple', 'compound']

    t, ref = loadReference(referenceFile, modelType)
    t2, ref2 = loadReference(referenceFile, 'multinomial')

    # # unnecessarily rewritten every time, but no harm done
    # with open('ref_%s.csv' % modelType, 'w') as refFile:
    #     for tune in range(1, 501):
    #         refFile.write('%d,%s\n' % (tune, ref[tune]))

    X, Y, names, minN = prepareSets(path, ref)
    Ynp = np.array(Y)
    print np.where(Ynp == 'simple')[0].shape, 'simple data'
    print np.where(Ynp == 'compound')[0].shape, 'compound data'

    splits = makeSplits(t, nFolds)

    folds = {}
    for a in splits:
        for b in range(nFolds):
            for c in splits[a][b]:
                folds[c] = b

    # with open('folds_%s.csv' % modelType, 'w') as outfile:
    #     for i in range(1, 501):
    #         outfile.write('%03d, %d\n' % (i, folds[i]))    

    print 'Starting cross-validation'

    wLens = range(1, minN + 1)
    print minN
    wLens = [minN + 1] # when curve not needed...
    finalRes = {s: [] for s in wLens}

    confMat = np.zeros((len(labels),len(labels)), dtype = int)
    tuneConfMat = np.zeros((len(labels),len(labels)), dtype = int)
    tuneErrors = {}

    predPerType = {
        'reel': [],
        'jig': [],
        'slide': [],
        'slipjig': [],
        'hornpipe': [],
        'polka': [],
        'other44': [],
        'waltz': []
    }

    test_y_concat = []
    pred_concat = []
    probas_concat = []
    sig_probas_concat = []
    gt_concat = []

    for i in range(nFolds):
        print 'Fold %d' % i
        testId = [ s[i] for s in splits.values() ]
        testId = [ x for s in testId for x in s ] # flatten list of lists

        train_x = []
        train_y = []
        test_x = []
        test_y = []
        test_t = []
        for j in range(len(X)):
            if names[j] in testId:
                test_x.append(X[j])
                test_y.append(Y[j])
                test_t.append(names[j])
            else:
                train_x.append(X[j])
                train_y.append(Y[j])

        logreg = LogisticRegression(class_weight = 'balanced',
                                    solver='liblinear',
                                    multi_class = 'ovr')
        logreg.fit(train_x, train_y)

        score = logreg.score(test_x, test_y)
        print 'Accuracy on test set: %.3f' % score

        # with open('models/%s_%d.pcl' % (modelType, i), 'w') as modFile:
        #     pickle.dump(logreg, modFile)

        prediction = logreg.predict(test_x)

        print 'balanced: %.3f' % (balanced_accuracy_score(test_y, prediction))

        prediction_proba = logreg.predict_proba(test_x)

        cm = confusion_matrix(test_y, prediction,
                              labels = labels)
        confMat += cm

        if modelType == 'binomial':
            for ind in range(len(test_x)):
                predPerType[ref2[test_t[ind]]].append(prediction[ind])

        for r in wLens:
            # finalRes[r].extend(getScores(test_y, prediction, test_t, r))
            finalRes[r].extend(getScores2(test_y, prediction_proba, test_t, r, logreg))

        tuneRes = {}
        tindices = np.array(test_t)
        indices = set(test_t)
        for ind in indices:
            loc = np.where(tindices == ind)
            res = []
            avgProbs = []
            for typeIndex in range(len(logreg.classes_)):
                avgProbs.append(np.average(prediction_proba[loc, typeIndex]))
                tuneProb = logreg.classes_[np.argmax(avgProbs)]
            tuneRes[ind] = tuneProb

        for k in tuneRes:
            index_p = labels.index(tuneRes[k])
            index_ref = labels.index(ref[k])
            tuneConfMat[ index_ref, index_p ] += 1
            if index_p != index_ref:
                tuneErrors[k] = tuneRes[k]


        pred_concat.extend(prediction)
        gt_concat.extend(test_y)

        binClasses = ['hornpipe', 'jig', 'other44', 'polka',
              'reel', 'slide', 'slipjig', 'waltz'] if modelType == "multinomial" \
              else ['compound', 'simple']

        # for chunk in range(len(test_y)):
        #     test_y_concat.append(1 if test_y[chunk] == prediction[chunk] else 0)
        #     probas_concat.append(prediction_proba[chunk][binClasses.index(test_y[chunk])])

        test_y_concat.extend(test_y)
        # print prediction_proba[:,1].shape
        # sys.exit(0)
        probas_concat.extend(prediction_proba[:,1])

        # sigmoid = CalibratedClassifierCV(logreg, cv=3, method='isotonic')
        # sigmoid.fit(train_x, train_y)
        # sig_proba = sigmoid.predict_proba(test_x)
        # sig_probas_concat.extend(sig_proba[:,1])

        # tune_prediction = {}
        # for p in range(len(test_x)):
        #     tune_id = test_t[p]
        #     if tune_id not in tune_prediction:
        #         tune_prediction[tune_id] = dict.fromkeys(labels, 0)
        #     tune_prediction[tune_id][prediction[p]] += 1
        # for k in tune_prediction:
        #     p = max(tune_prediction[k].iteritems(), key = lambda x: x[1])[0]
        #     index_p = labels.index(p)
        #     index_ref = labels.index(ref[k])
        #     tuneConfMat[ index_ref, index_p ] += 1
        #     if index_p != index_ref:
        #         tuneErrors[k] = p

    clf_score = brier_score_loss(test_y_concat,probas_concat,
                                 pos_label = 'simple' if modelType == 'binomial' else 'jig')
    print "\tBrier logreg: %1.3f" % (clf_score)

    fop = []
    mpv = []

    for f in [lambda x: x,
              lambda x: sqrt(x),
              lambda x: (2*x-x**2),
              lambda x: sqrt(2*x-x**2)]:
        print '\t> %.3f' % brier_score_loss(test_y_concat,
                                            map(f, probas_concat),
                                            pos_label = 'simple' if modelType == 'binomial' else 'jig')
        fraction_of_positives, mean_predicted_value = \
            calibration_curve(test_y_concat, probas_concat, n_bins=10)
        fop.append(fraction_of_positives)
        mpv.append(mean_predicted_value)

    # clf_score = brier_score_loss(test_y_concat,sig_probas_concat,
    #                              pos_label = 'simple' if modelType == 'binomial' else 'jig')
    # print "\tBrier sigmoid: %1.3f" % (clf_score)

    if modelType == 'binomial':
        # fraction_of_positives, mean_predicted_value = \
        #     calibration_curve(test_y_concat, probas_concat, n_bins=10)

        # fraction_of_positives_sig, mean_predicted_value_sig = \
        #     calibration_curve(test_y_concat, sig_probas_concat, n_bins=10)

        plt.figure(figsize=(5, 8))
        plt.subplot(211)
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        for i in range(4):
            plt.plot(mpv[i], fop[i], "s-", label="LogReg")
        plt.ylabel("Fraction of positives")
        plt.ylim([-0.05, 1.05])
        plt.legend(loc="lower right")
        plt.title('Calibration plots  (reliability curve)')

        # plt.plot(mean_predicted_value_sig, fraction_of_positives_sig, "s-", label="LogReg+sig")
        plt.subplot(212)
        plt.hist(probas_concat, range=(0, 1), bins=10, label="LogReg", histtype="step", lw=2)
        plt.xlabel("Mean predicted value")
        plt.ylabel("Count")
        # plt.hist(sig_probas_concat, range=(0, 1), bins=10, label="LogReg+sig", histtype="step", lw=2)
        # plt.savefig("figures/calibration.pdf", bbox_inches='tight')
        plt.show()
        # sys.exit(0)
        
    print 'Overall Balanced Acc: %.3f' % (balanced_accuracy_score(gt_concat, pred_concat) * 100.)

    print 'Frame accuracy: %.2f%%' % (100. * np.trace(confMat) / np.sum(confMat))

    print 'Tune accuracy: %.2f%%' % (100. * np.trace(tuneConfMat) / np.sum(tuneConfMat))

    print printConfMat(confMat.astype(np.float32),
                       tuneConfMat,
                       labels, transpose = True, normalise = True)

    print '\nWrong predictions on tunes:'
    for te in tuneErrors:
        print '\t%s (%s), recognised as %s' % (te, ref[te], tuneErrors[te])

    if modelType == "binomial":
        for tuneType in ['jig', 'slide', 'slipjig']:
            c = 0
            for p in predPerType[tuneType]:
                if p=='compound':
                    c+=1
            print '>> %s: %.5f' % (tuneType, c / float(len(predPerType[tuneType])))

        for tuneType in ['reel', 'hornpipe', 'polka', 'other44', 'waltz']:
            c = 0
            for p in predPerType[tuneType]:
                if p=='simple':
                    c+=1
            print '>> %s: %.5f' % (tuneType, c / float(len(predPerType[tuneType])))

    plotValues = []
    for r in wLens:
        plotValues.append(100. * sum(finalRes[r]) / float(len(finalRes[r])))

    
    print "Highest span acc [%d]: %.2f" % (len(plotValues), plotValues[-1])

    plt.figure(figsize=(5, 4))
    plt.plot(wLens, plotValues)
    plt.xlabel("length of window span")
    plt.ylabel("prediction accuracy (%)")
    plt.savefig("figures/acc_span_%s.pdf" % modelType, bbox_inches='tight')

    return wLens, plotValues

def prepareTrainAndTestOnHalf(path, referenceFile, modelType):

    print '\n---- TRAINING ON HALF DATASET ----\n'

    nFolds = 4 if modelType == 'multinomial' else 10

    labels = [
        'reel',
        'jig',
        'slide',
        'slipjig',
        'hornpipe',
        'polka',
        'other44',
        'waltz'
    ] if modelType == 'multinomial' else ['simple', 'compound']

    t, ref = loadReference(referenceFile, modelType)

    X, Y, names, minN = prepareSets(path, ref)

    # testSplit = makeSplits(t, 2)
    # trainVal = {}
    # for cls in testSplit:
    #     trainVal[cls] = testSplit[cls][0]

    trainVal = {}
    for t in labels:
        trainVal[t] = []
    with open('folds_trainval_test.csv', 'r') as infile:
        reader = csv.reader(infile)
        for row in reader:
            if row[1] == '0':
                trainVal[ref[int(row[0])]].append(int(row[0]))

    trainValSplit = makeSplits(trainVal, nFolds)

    folds = {}
    for a in trainValSplit:
        for b in range(nFolds):
            for c in trainValSplit[a][b]:
                folds[c] = b

    with open('folds_thr_%s.csv' % modelType, 'w') as outfile:
        for i in folds:
            outfile.write('%03d, %d\n' % (i, folds[i]))

    print 'Starting cross-validation'

    confMat = np.zeros((len(labels),len(labels)), dtype = int)
    tuneConfMat = np.zeros((len(labels),len(labels)), dtype = int)
    tuneErrors = {}

    for i in range(nFolds):
        print 'Fold %d' % i
        testId = [ s[i] for s in trainValSplit.values() ]
        testId = [ x for s in testId for x in s ] # flatten list of lists

        train_x = []
        train_y = []
        test_x = []
        test_y = []
        test_t = []
        for j in range(len(X)):
            if names[j] in testId:
                test_x.append(X[j])
                test_y.append(Y[j])
                test_t.append(names[j])
            else:
                train_x.append(X[j])
                train_y.append(Y[j])

        logreg = LogisticRegression(class_weight = 'balanced',
                                    solver='liblinear',
                                    multi_class = 'ovr')
        logreg.fit(train_x, train_y)
        print logreg.classes_

        score = logreg.score(test_x, test_y)
        print 'Accuracy on test set: %.3f' % score

        with open('models/%s_%d_thr.pcl' % (modelType, i), 'w') as modFile:
            pickle.dump(logreg, modFile)

        prediction = logreg.predict(test_x)
        cm = confusion_matrix(test_y, prediction,
                              labels = labels)
        confMat += cm

        tune_prediction = {}
        for p in range(len(test_x)):
            tune_id = test_t[p]
            if tune_id not in tune_prediction:
                tune_prediction[tune_id] = dict.fromkeys(labels, 0)
            tune_prediction[tune_id][prediction[p]] += 1
        for k in tune_prediction:
            p = max(tune_prediction[k].iteritems(), key = lambda x: x[1])[0]
            index_p = labels.index(p)
            index_ref = labels.index(ref[k])
            tuneConfMat[ index_ref, index_p ] += 1
            if index_p != index_ref:
                tuneErrors[k] = p

    print '\nAggregate confusion matrix (frame):'
    print confMat
    print 'Frame acccuracy: %.2f%%' % (100. * np.trace(confMat) / np.sum(confMat))

    print '\nAggregate confusion matrix (tune):'
    print tuneConfMat
    print 'Tune acccuracy: %.2f%%' % (100. * np.trace(tuneConfMat) / np.sum(tuneConfMat))

    print '\nWrong predictions on tunes:'
    for te in tuneErrors:
        print '\t%s (%s), recognised as %s' % (te, ref[te], tuneErrors[te])

def trainFinalModel(path, referenceFile, modelType):

    print '\n---- FINAL TRAINING ----\n'

    trainVal = []
    with open('folds_thr_%s.csv' % modelType, 'r') as infile:
        reader = csv.reader(infile)
        for row in reader:
            trainVal.append(int(row[0]))

    t, ref = loadReference(referenceFile, modelType)

    X, Y, names, minN = prepareSets(path, ref)

    train_x = []
    train_y = []
    for i in range(len(X)):
        if names[i] in trainVal:
            train_x.append(X[i])
            train_y.append(Y[i])

    logreg = LogisticRegression(class_weight = 'balanced',
                                solver='liblinear',
                                multi_class = 'ovr')
    logreg.fit(train_x, train_y)
    with open('models/%s_final.pcl' % modelType, 'w') as modFile:
        pickle.dump(logreg, modFile)
    print 'model trained and saved!'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--modelType', required=True,
                        help='type of model to train (binomial/multinomial)')
    args = parser.parse_args()

    if args.modelType == 'binomial' or args.modelType == 'multinomial':
        np.random.seed(1564)
    else:
        print 'modelType should be \'binomial\' or \'multinomial\''
        sys.exit(1)

    doSplits()
    prepareTrainAndTest('data', '../dataset.csv', args.modelType)
    # prepareTrainAndTestOnHalf('data', '../dataset.csv', args.modelType)
    # trainFinalModel('data', '../dataset.csv', args.modelType)
