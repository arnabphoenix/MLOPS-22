import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
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


report = pd.DataFrame(h_param_comb)
#print(report.head())


train_frac = 0.8
test_frac = 0.1
dev_frac = 0.1

#PART: load dataset -- data from csv, tsv, jsonl, pickle
digits = datasets.load_digits()

#PART: sanity check visualization of the data
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)


#PART: data pre-processing -- to remove some noise, to normalize data, format the data to be consumed by mode
# flatten the images
n_samples = len(digits.images)
#digits = datasets.load_digits()
data = digits.images
a  = data
#data = resize(data, (1797,10, 10))
#print('Resized image: ')
#print(data[0].shape)
data = digits.images.reshape((n_samples, -1))


dev_test_frac = 1-train_frac
X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.52, random_state=28)


best_acc = -1.0
best_model = None
best_h_params = None




accuracy_train = []
accuracy_dev = []
accuracy_test = []

for cur_h_params in h_param_comb:

   
    clf = svm.SVC()

   
    hyper_params = cur_h_params
    clf.set_params(**hyper_params)

   
    clf.fit(X_train, y_train)

    
    predicted_train = clf.predict(X_train)
    #predicted_dev = clf.predict(X_dev)
    predicted_test = clf.predict(X_test)

    
    cur_acc1 = metrics.accuracy_score(y_pred=predicted_train, y_true=y_train)
    #cur_acc2 = metrics.accuracy_score(y_pred=predicted_dev, y_true=y_dev)
    cur_acc3 = metrics.accuracy_score(y_pred=predicted_test, y_true=y_test)
    accuracy_train.append(cur_acc1)
    #accuracy_dev.append(cur_acc2)
    accuracy_test.append(cur_acc3)


    
    if cur_acc1 > best_acc:
        best_acc = cur_acc1
        best_model = clf
        best_h_params = cur_h_params
        #print("Found new best acc with :"+str(cur_h_params))
        #print("New best val accuracy:" + str(cur_acc1))


accuracy_val=accuracy_score(y_test, predicted_test)
print("test accuracy:",accuracy_val)
f1=f1_score(y_test.reshape(-1,1), predicted_test.reshape(-1,1), average='macro')
print("test macro-f1:",f1)

best_param_config = "_".join(
        [h + "=" + str(best_h_params) for h in best_h_params]
    )
val=28



dump(clf,"models/"+"Svm" + "_" + str(best_param_config) +"Random_state: "+str(val)+".joblib")