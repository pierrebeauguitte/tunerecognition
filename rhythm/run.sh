#!/bin/bash

rm -f folds_trainval_test.csv
python train.py --modelType binomial
python train.py --modelType multinomial
python predict.py --modelType binomial
python predict.py --modelType multinomial
python findMaxF1.py
