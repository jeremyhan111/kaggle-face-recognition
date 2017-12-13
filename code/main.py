#! /usr/bin/env python
########################################
# CS63: Artificial Intelligence, Final Project
# Recognizing Facial Expressions with Machine Learning Algorithms
# Spring 2017, Swarthmore College
########################################
# full name(s): Do June Min, Jeremy Han
# username(s): dmin1, jhan2
########################################
import sys
from os.path import exists
#numpy
import numpy as np
from numpy.linalg import norm
from numpy import argsort,argmin, argmax, random
from numpy import array, add, multiply
from random import shuffle
#image display
import scipy.misc as smp
import matplotlib.pyplot as plt
from matplotlib import image as img
from PIL import Image
from resizeimage import resizeimage
#image preprocessing
from skimage import io, filters
from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.feature import CENSURE
#system library
import time
import csv as csv
import random
import warnings, sys
#machine learning algorithms
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score
#neural network
import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import layers
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from keras.optimizers import SGD
#visualization
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix 
import itertools
from itertools import cycle
import time

#number of base learner for ensemble methods
NBASE = 200
#epochs for neural networks
EPOCH = 30
#toggle option to write training history
WRITE = 1
#Validation Set rate during training
VALRA = 0.1
#Label listing
LABELS = [0,1,2,3,4,5,6]

def usage():
    print >> sys.stderr, "Usage: python main.py option size"
    print >> sys.stderr, "  option - 0 for k-fold cv, 1 for test"
    print >> sys.stderr, "  size - set the limit for the training data, 0 is full"

def output_info(clfs):
    """
    Outputs basic information of class labels and avaialble ML algorithms
    """
    print "============="
    print "Emotions"
    print "============="
    print "0: anger"
    print "1: disgust"
    print "2: fear"
    print "3: happy"
    print "4: sad"
    print "5: surprised"
    print "6: neutral"
    print
    print "============="
    print "Classifiers"
    print "============="
    for i in range(len(clfs)):
        print str(i)+": "+clfs[i].name
    print 


def output_msg(images, labels,testx,testy):    
    """
    Given data set, outputs basic info on the set
    """
    print "image read to np arrays"       
    print "image size: 48 X 48"
    print "train x size: "+str(len(images))
    print "train y size : "+str(len(labels))
    print "test x size: "+str(len(testx))
    print "test y size : "+str(len(testy))
    print

def shuffle(images, labels):
    """
    Shuffle data set and labels while preserving their matching
    """
    if len(images) != len(labels):
        print "error!: images and label do not have same size"
        exit()
    index = []
    for i in range(len(images)):
        index.append(i)
    
    random.shuffle(index)
    n_images = []
    n_labels = []
    for num in index:
        n_images.append(images[num])
        n_labels.append(labels[num])
        
    return n_images, n_labels

def show(image):
    """
    Function to display and save a data image to a file
    """
    fig = plt.figure()
    plt.plot
    img.imsave("face.png", (np.reshape(np.multiply(image,255), (48,48))), cmap="gray")
    plt.show()
    plt.savefig("face1.png")
    
def addNoise(image):
    #expand this to data_augmentation function
    #with mirror flip, white noise, distortion etc
    num_noise = np.random.randint(1000,1100)
    cx = []
    cy = []
    for i in range(num_noise):
        cx.append(np.random.randint(0,48))
        cy.append(np.random.randint(0,48))

    for i in range(num_noise):
        image[cx[i]][cy[i]] = [np.random.randint(0,1000)/1000.]
    image = np.array(image)
    
    return image

def readData():
    """
    Read data from /scratch/ storage and 
    apply data processing/augmentation
    """
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    
    with open("/scratch/dmin1/lab10/data/icml_2013/fer2013/fer2013.csv","r") as fer:
        
        header = fer.readline()
        for line in fer:
            
            data1 = np.zeros((48,48,1), dtype = np.float32)
            data1.astype('float32')
            mirror1 = np.zeros((48,48,1), dtype = np.float32)
            mirror1.astype('float32')
        
            numbers = line.split(",")
            emotion = int(numbers[0].strip())
            dataclass = numbers[len(numbers)-1].strip("\n")

            numbers = numbers[1:len(numbers)-1]
            assert(len(numbers) == 1)

            numbers = numbers[0].split()

            #Converting vector format to 2d array format
            for i in range(48):
                for j in range(48):       
                    data1[i][j] = ( float(numbers[48*i+j]) /   255.0 )
                    mirror1[i][j] = ( float(numbers[48*i+j])/  255.0 )                     

            if dataclass != "PrivateTest":
                
                mirror1 = np.fliplr(mirror1) #creating mirror image 
                train_images.append(data1)
                train_images.append(mirror1)
                
                train_labels.append(emotion)
                train_labels.append(emotion)
                
            else:
                #no need to do data augmentation for test set
                test_images.append(data1)
                test_labels.append(emotion)
 
    #shuffling the data
    train_images, train_labels = shuffle(train_images, train_labels)   
    test_images, test_labels = shuffle(test_images, test_labels)
    
    print "train length: ", len(train_images)
    print "test length: ", len(test_images)
    
    return np.array(train_images), np.array(train_labels), np.array(test_images), np.array(test_labels)

def upshape(X):
    """
    Change ab np array of 1d vectors to an np array of 2d image arrays
    """
    upshaped = []
    for item in X:
        data1 = np.zeros((48,48,1), dtype = np.float32)
        
        for i in range(48):
            for j in range(48):       
                data1[i][j] = ( float(item[48*i+j])   )
        upshaped.append(data1)
    return np.array(upshaped)


def reshape(X):
    """
    Reverse operation of upshape
    """
    nsamples, nx, ny, nz = X.shape
    return X.reshape((nsamples,nx*ny*nz))

class Classifier(object):
    """
    Parent class for various classifiers
    """
    def __init__(self):
        pass

    def fit(self):
        pass
    
    def train(self, images, labels):
        pass
class SVM(Classifier):
    """
    Support vector machines (SVMs) are a set of supervised learning
    methods used for classification, regression, and outliers detection
    kernel: {'rbf' by default}
    """
    def __init__(self):
        self.name = "Support Vector Machine"
        
    def fit(self, training_set, training_ans):
        self.clf = svm.SVC(C=10.0)
        self.clf = self.clf.fit((training_set), training_ans)
        return self

    def predict(self, test_set):
        return self.clf.predict((test_set))
    
    def score(self, xtest, ytest):
        return self.clf.score((xtest),ytest)

    #def save(self):

class DecisionTree(Classifier):
    """
    Decision Trees are a non-parametric supervised learning method
    used for classification and regression. The goal is to create a model
    that predicts the value of a target variable by learning simple decision
    rules inferred from the data features
    """
    def __init__(self):
        self.name = "Decision Tree"
        
    def fit(self, training_set, training_ans):
        self.clf = tree.DecisionTreeClassifier()
        self.clf = self.clf.fit((training_set), training_ans)
        return self

    def predict(self, test_set):
        return self.clf.predict((test_set))
    
    def score(self, xtest, ytest):
        return self.clf.score((xtest),ytest)

    #def save(self):

class RandomForest(Classifier):
    """
    RandomForest classifier - an ensemble method
    By default, base classifier is decision tree.
    """
    def __init__(self):
        self.name = "Random Forest"
        
    def fit(self, training_set, training_ans):
        self.clf = RandomForestClassifier(n_estimators = NBASE)
        self.clf = self.clf.fit((training_set), training_ans)
        return self

    def predict(self, test_set):
        return self.clf.predict((test_set))
    
    def score(self, xtest, ytest):
        return self.clf.score((xtest),ytest)

    #def save(self):

class AdaBoost(Classifier):
    """
    AdaBoost classifier - an ensemble method
    By default, base classifier is decision tree.
    """
    def __init__(self):
        self.name = "AdaBoost"
        
    def fit(self, training_set, training_ans):
        clf = AdaBoostClassifier(n_estimators=NBASE)
        self.clf = RandomForestClassifier(n_estimators = NBASE)
        self.clf = self.clf.fit((training_set), training_ans)
        return self

    def predict(self, test_set):
        return self.clf.predict((test_set))
    
    def score(self, xtest, ytest):
        return self.clf.score((xtest),ytest)


class Convnet(Classifier):
    """
    Convnet - a Convolutional Neural Network
    """
    def __init__(self):
        self.name = "Convnet"
        
    def fit(self, x_train, y_train):
 
        num_classes = 7
        batch_size = 128
        
        input_shape = (48,48,1)
        
    
        y_train = keras.utils.to_categorical(y_train,num_classes)
    
        model = Sequential()

        model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
        model.add(Activation('relu'))   
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9)#, nesterov=True)
   
        model.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=['accuracy'])
    
        hist = model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=EPOCH,
                validation_split=VALRA,
                shuffle=True)
    
        self.clf = model
        
        plot_model(model, to_file='../paper/figures/conv2_model.png')


        if WRITE:
            self.save(hist)
        return self

    def predict(self, test_set):
        return self.clf.predict(upshape(test_set))
    
    def save(self,hist):
        
        timestr=time.strftime("%Y%m%d-%H%M%S")
        self.clf.save('/scratch/dmin1/lab10/model/conv2-'+timestr+'.h5')
        f = open("/scratch/dmin1/lab10/history/conv2-"+timestr+".hist","w")
        f.write(str(hist.history))
        f.close()

class LightConv(Classifier):
    """
    LightConv - a base learner for ConvnetGroup
    """
    def __init__(self):
        self.name = "LightConv"
        
    def fit(self, x_train, y_train):
        num_classes = 1
        batch_size = 128
        epochs = 1 #1
        input_shape = (48,48,1)
        
        x_train = upshape(x_train)
        
        model = Sequential()

        model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=x_train.shape[1:]))
        model.add(Activation('relu'))   
        model.add(Conv2D(32, (3, 3)))#
        model.add(Activation('relu'))#
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(num_classes))
        model.add(Activation('sigmoid'))
        
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9)#, nesterov=True)

        model.compile(loss='binary_crossentropy',
                optimizer='sgd',
                metrics=['accuracy'])

        hist = model.fit(x_train, y_train,
                  batch_size=batch_size,
                epochs=EPOCH,
                validation_split = VALRA,
                shuffle=True)

        plot_model(model, to_file='../paper/figures/light_conv_model.png')

        self.clf = model
        return self
        
    def predict(self, test_set):
        return self.clf.predict(upshape(test_set))
    
    def save():
        return
        #timestr=time.strftime("%Y%m%d-%H%M%S")
        #self.clf.save('../model/conv-'+timestr+'.h5')


class ConvnetGroup(Classifier):
    """
    ConvnetGroup - ensemble method consisting of 7 LightConv weak base learners
    """
    def __init__(self):
        self.name = "ConvnetGroup"
    

    def fit(self, x_train, y_train):
        x_trains = []
        y_trains = []
 
        for i in range(7):
            x_trains.append(x_train)
            temp_train = []
            for item in y_train:
                if (item) == i:
                    temp_train.append(1)
                else:
                    temp_train.append(0)
            y_trains.append(temp_train)
        
        self.clf = []
        for i in range(7):

            self.clf.append(LightConv().fit(x_trains[i],np.array(y_trains[i])))
        
        if WRITE:
            self.save()
        return self
  
    def predict(self, test_x, test_y):           
        x_tests = []
        y_tests = []

        for i in range(7):
            x_tests.append(test_x)
            temp_test = []
            for item in test_y:
                if (item) == i:
                    temp_test.append(1)
                else:
                    temp_test.append(0)
            y_tests.append(temp_test)

        results = []
        answers = []
        for i in range(7):
            results.append(self.clf[i].clf.predict_classes(upshape(x_tests[i])))
            answers.append(np.array(test_y))
        return results, answers
    
    def save(self):
        for i in range(7):
            timestr=time.strftime("%Y%m%d-%H%M%S")
            self.clf[i].clf.save('/scratch/dmin1/lab10/model/convLight'+str(i)+'-'+timestr+'.h5')

def ConvnetGroupTest(images, labels):
    """
    Since ConvnetGroup is a ensemble of neural networks, 
    a different method to calculate accuracy is needed
    """
    print
    print "===================="
    print "ConvNet Group Test"
    model = ConvnetGroup()
    images = reshape(images)

    kfold = KFold(n_splits = NFOLD)
    results =  [model.fit(images[train], labels[train]).predict(images[test], labels[train]) \
            for train, test in kfold.split(images)]
    
    predictions = [a for (a,b) in results ] 
    answers =  [b for (a,b) in results ] 
    
    assert(len(answers) == len(predictions))
    scores = []
   
    #inside each fold
    for i in range(len(answers)):
        assert(len(answers[i]) == len(predictions[i])) #==7
        total = 0.0
        correct = 0.0
        #for each data point
        for j in range(len(answers[i][0])):
            total += 1.0
            temp = []
            #inside each classifier
    
            for k in range(len(predictions[i])):
                temp.append(predictions[i][k][j])
            output = argmax(temp)
            
            if output == answers[i][0][j]:
                correct += 1.0
        #print "Fold: ",i
        #print "Correct: ", correct
        #print "Total: ", total
        scores.append(correct/total)
        #print correct, total

    print "ConvNet Group Result: ", scores 
    print scores
    sumt =0.0
    for item in scores:
        sumt += item
    print "average: ", sumt/len(scores)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix(without normalization)')
    
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def main():   
 
    clfs = [ SVM(), DecisionTree(), RandomForest(),AdaBoost(), \
         Convnet(), ConvnetGroup()]
    output_info(clfs)

    index = raw_input("Enter the index of the classifier you want to use: ")
    index = index.strip()
    index = int(index)
    if index not in LABELS:
        print "Error! Enter valid input!"
        exit()
   
    images, labels, testx,testy = readData()

    #output some statistics on data
    output_msg(images, labels, testx, testy)

    clf = clfs[index]
    if index != 5:
            
        print
        print "Testing with : ",clf.name
        if index == 4:
            model = load_model("/scratch/dmin1/lab10/model/conv-0423-60.h5")
            #predictions = model.predict_classes(testx)
            predictions = clf.fit((images),labels).clf.predict_classes((testx))
        else:
            predictions = clf.fit(reshape(images),labels).predict(reshape(testx))
        print
        print "Accuracy: ", accuracy_score(testy, predictions)
            
        cm =  confusion_matrix(testy, predictions, LABELS)
        print cm
        plt.figure()
        plot_confusion_matrix(cm, classes = LABELS, normalize = False, title = 'Confusion Matrix')           
        timestr=time.strftime("%Y%m%d-%H%M%S")
        plt.savefig("../paper/figures/"+timestr+".png")
        

    else:
        print "Testing with : ",clf.name
 
        predictions, answers = clf.fit(reshape(images),labels).predict(reshape(testx), testy)
            
        assert(len(answers) == len(predictions))
          
        total = 0.0
        correct = 0.0
        #for each data point
        for j in range(len(predictions[0])):
            total += 1.0
            temp = []
            #inside each binary classifier
            for k in range(7):
                temp.append(predictions[k][j])
            output = argmax(temp)
            
            if output == answers[0][j]:
                correct += 1.0
        print "Accuracy: ", correct/total
        cm =  confusion_matrix(answers, predictions, LABELS)
        print cm
        plt.figure()
        plot_confusion_matrix(cm, classes = LABELS, normalize = False, title = 'Normalized Confusion Matrix')           
        timestr=time.strftime("%Y%m%d-%H%M%S")
        plt.savefig("../paper/figures/"+timestr+".png")
        
if __name__ == '__main__':
    main()



