import matplotlib.pyplot as plt
import pandas as pd
from skimage.transform import resize
from joblib import dump
from sklearn import svm, tree
import pdb
import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
#resize(image, (100, 100)).shape(100, 100)

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

# 1. set the ranges of hyper parameters
gamma_list = [0.01, 0.005]
c_list = [0.1, 0.2, 0.5]

h_param_comb = [{'gamma':g, 'C':c} for g in gamma_list for c in c_list]

assert len(h_param_comb) == len(gamma_list)*len(c_list)


train_frac = 0.8
test_frac = 0.1
dev_frac = 0.1

#PART: load dataset -- data from csv, tsv, jsonl, pickle
digits = datasets.load_digits()


#PART: data pre-processing -- to remove some noise, to normalize data, format the data to be consumed by mode
# flatten the images
n_samples = len(digits.images)
#digits = datasets.load_digits()
data = digits.images
a  = data
data = digits.images.reshape((n_samples, -1))


X_train1, X_test1, y_train1, y_test1 = train_test_split(data, digits.target, test_size=0.52, random_state=28)

X_train2, X_test2, y_train2, y_test2 = train_test_split(data, digits.target, test_size=0.52, random_state=28)

X_train3, X_test3, y_train3, y_test3 = train_test_split(data, digits.target, test_size=0.52, random_state=72)

def test_if_correct():
    assert X_train1.all() == X_train2.all()
    assert X_test1.all() == X_test2.all()
def test_if_not_correct():
    assert (X_test1 != X_test3).any()
    assert (X_train2 != X_train3).any()