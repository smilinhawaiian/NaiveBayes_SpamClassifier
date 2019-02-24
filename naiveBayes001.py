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

    print("\n")
    # counts for making sure data is split evenly
    spam_count = -1
    nospam_count = -1

    # arrays for test and train data
    train_data_spam = np.empty((0,58), int)
    train_data_nospam = np.empty((0,58), int)
    testing_data = np.empty((0,58), int)

    #** add randomization of spambase data array here

    # for every row, check if spam. Add equal parts spam/not spam to test/train sets
    for an_instance in spambase_data:
        # check if an_instance is spam
        # instance is spam (last value in column is 1)
        if(an_instance[-1] == 1):
            #print(an_instance[-1]) for testing
            # increment spam counter
            spam_count = spam_count + 1
            # if even, add row to train_data_spam
            if((spam_count%2) == 0):
                train_data_spam = np.append(train_data_spam, np.array([an_instance]), 0)
            # if spam count is odd, add row to test_data
            else:
                testing_data = np.append(testing_data,np.array([an_instance]), 0)
        # instance not spam
        else:
            # increment no_spam counter
            nospam_count = nospam_count + 1
            # if counter is even, add row to train_data_nospam
            if(nospam_count%2 ==0):
                train_data_nospam = np.append(train_data_nospam, np.array([an_instance]), 0)

            # if count is odd, add row to test_data
            else:
                testing_data = np.append(testing_data,np.array([an_instance]), 0)
    # // endfor

    # remove spam/no spam identifying column from each matrix
    train_spam = np.delete(train_data_spam, -1, axis=1)
    train_nospam = np.delete(train_data_nospam, -1, axis=1)
    test_data = np.delete(testing_data, -1, axis=1)

    # declare vars for checking prior probability
    spam_test_num = 0
    nospam_test_num = 0
#    check that there is 40% spam
    for spam_test_instance in train_data_spam:
        spam_test_num = spam_test_num + 1

    for nospam_test_instance in train_data_nospam:
        nospam_test_num = nospam_test_num + 1

    total_test_num = spam_test_num + nospam_test_num

    # calculate percent of spam
    prob_test_spam = spam_test_num / total_test_num
    # calculate percent not spam
    prob_test_nospam = nospam_test_num / total_test_num

    # print prior probability of spam test data
    print("Prior Probability of spam = %f \n" %prob_test_spam)
    # print prior probability of not spam test data
    print("Prior Probability of not spam = %f \n" %prob_test_nospam)

    #print("total spam examples = %d\n" %spam_count) #for testing
    #print("test spam examples = %d\n" %spam_test_num) #for testing

# For each of the 57 features, compute the mean and 
#  standard deviation in the training set of the values
#  given each class
# To avoid the problem any standard deviation = 0
#  add a small value - epsilon - 0.0001
#  to each standard deviation that you compute.

    # Transpose matrices for computation
    ttrain_spam = train_spam.transpose()
    ttrain_nospam = train_nospam.transpose()
    ttest_data = test_data.transpose()

    # Compute the mean ttrain_spam
    train_smean = np.empty((57,0), int)
    for a_feature in ttrain_spam:
        #row_mean = np.empty((1,0), int)
        #row_mean = np.mean(np.array([a_feature]), axis=0)
        row_mean = np.array([a_feature.mean()])
        print(row_mean) #for testing
        #train_smean = np.append(train_smean, np.array([row_mean]))  #works
        train_smean = np.append(train_smean, row_mean)
        #train_smean = np.append(train_smean, np.mean(np.array([a_feature]), axis=0, out=train_smean))
        #train_data_spam = np.append(train_data_spam, np.array([an_instance]), 0)
    # // endfor

    #train_smean = np.empty((58,0), int)
    #train_smean = np.empty(57)
    #train_smean = np.mean(ttrain_spam, axis=0, dtype=None, out=train_smean)
    print("")
    print(train_smean) # for testing

    # Compute the mean ttrain_nospam
    train_nosmean = np.empty((57,0),int)
    for b_feature in ttrain_nospam:
        brow_mean = np.array([b_feature.mean()])
        print(brow_mean) #for testing
        train_nosmean = np.append(train_nosmean, brow_mean)
    # // endfor

    print("")
    print(train_nosmean) # for testing
    # Compute the standard deviation ttrain_spam
    #train_ssdev = 
    # Compute the standard deviation ttrain_nospam
    #train_nossdev = 


#    print("\n\n")
#    print(f"train_data_spam: \n{train_data_spam}")
#    print(f"train_spam: \n{train_spam}")
#    print(f"ttrain_spam: \n{ttrain_spam}")
    print("\n\n")

#    for notes below only
#    test_labels = spambase_data[:,0]


