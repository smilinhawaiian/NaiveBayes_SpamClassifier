# Sharice Mayer
# 2/23/19
# Gaussian Naive Bayes Classifier
# Classify Spambase data from UCI ML repository

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


# 1. 
# Create training and test set:

# Read in Spambase data set
# Split instances into spam and not spam examples
# Place each type equally into test and train datasets
# Make sure each test and train have 40% spam, 60% not spam


# 2. 
# Create probabilistic model.

# Compute prior probability for each class
# 1(spam), and 0(not spam) in the training data
# *P(1) should be roughly .4
# For each of the 57 features, compute the mean and 
#  standard deviation in the training set of the values
#  given each class
# To avoid the problem any standard deviation = 0
#  add a small value - epsilon - 0.0001
#  to each standard deviation that you compute.


# 3. 
# Run Naive Bayes on the test data.

# Randomly mix all the test data
# Use Gaussian Naive Bayes algorithm to classify instances in your test set,
# using the means and standard deviations computed in part 2.
# P(x_i | c_j) = N(x_i ; m_(i,c_j) , stdev_(i,c_j) )
# where
# N(x ; m , stdev) = [ (1/ (sqrt(2pi)*stdev) )*( e^ ((x-m)^2 / (2*o^2) ) ) ]# *Because product of 58 probabilities will be small, use log of product
#   since argmax f(z) = argmax log(f(z))


# In report, include:
# Description of what I did &&
# Results for test set: 
# compute accuracy
# compute precision
# compute recall
# create confusion matrix


# python for main method:
if __name__ == "__main__":
    # Read in and create matrice of data
    spambase_data = pd.read_csv("spambase/spambase_copy.csv", header=None).values

    # split row instances into spam and not spam for training
    # mixed for testing - keeping same proportions test/train sets

    print("\n\n")
    # counts for making sure data is split evenly
    spam_count = -1
    nospam_count = -1

    # arrays for test and train data
    train_data_spam = np.zeros(spambase_data.shape)
    train_data_nospam = np.zeros(58)
    test_data = np.zeros(58)
    #print(train_data_spam)
    #train_data_nospam = np.zeros(58)
    #test_data = np.zeros(58)
    print(train_data_spam)

    #** add randomization of spambase data array here

    for an_instance in spambase_data:
        # check if an_instance is spam
        # instance is spam
        if(an_instance[-1] == 1):
            spam_count = spam_count + 1
            if(spam_count == 0): #first instance
                train_data_spam = np.r_[train_data_spam, an_instance]
            elif((spam_count%2) == 0):
                # add to test_data array
            else:
                test_data = np.r_[test_data, an_instance]
            #if(spam_count == 0): #first instance
            #    train_data_spam = np.r_[train_data_spam, an_instance]
            #print(an_instance[-1])
        # instance not spam
        else:
            nospam_count = nospam_count + 1
            if(nospam_count%2 ==0):
                train_data_nospam = np.r_[train_data_nospam, an_instance]
            else:
                test_data = np.r_[test_data, an_instance]





        #np.concatenate(an_instance,
        #print(an_instance)

#    train_data_spam = 
#    train_data_nospam = 

    
#    print("\n\n")
#    print(spambase_data)
#    print("\n\n")

    print(train_data_spam)
#    transposed_spam = spambase_data.transpose()
#    print(transposed_spam)
    print("\n\n")
#    test_labels = spambase_data[:,0]


