# Sharice Mayer
# 2/23/19
# Gaussian Naive Bayes Classifier
# Classify Spambase data from UCI ML repository

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import statistics
import math


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


    # 1. 
    # Create training and test set:


    # Read in and create matrice of data
    spambase_data = pd.read_csv("spambase/spambase_copy.csv", header=None).values
    print("\n")

    # counts for making sure data is split evenly
    spam_count = -1
    notspam_count = -1

    # arrays for test and train data
    train_data_spam = np.empty((0,58), int)
    train_data_notspam = np.empty((0,58), int)
    testing_data = np.empty((0,58), int)

    # add randomization of spambase data array here
    shuffled_base = np.zeros(spambase_data.shape)
    np.take(spambase_data,np.random.permutation(spambase_data.shape[0]),axis=0,out=shuffled_base)

    # split row instances into spam and not spam for training
    # mixed for testing - keeping same proportions test/train sets
    # for every row, check if spam. Add equal parts spam/not spam to test/train sets
    for an_instance in shuffled_base:
        # check if an_instance is spam
        # instance is spam (last value in column is 1)
        if(an_instance[-1] == 1):
            #print(an_instance[-1]) for testing
            # increment spam counter
            spam_count = spam_count + 1
            # if even, add row to train_data_spam
            if((spam_count%2) == 0):
                train_data_spam = np.append(train_data_spam, np.array([an_instance]), axis=0)
            # if spam count is odd, add row to test_data
            else:
                testing_data = np.append(testing_data,np.array([an_instance]), axis=0)
        # instance not spam
        else:
            # increment no_spam counter
            notspam_count = notspam_count + 1
            # if counter is even, add row to train_data_nospam
            if(notspam_count%2 ==0):
                train_data_notspam = np.append(train_data_notspam, np.array([an_instance]), axis=0)
            # if count is odd, add row to test_data
            else:
                testing_data = np.append(testing_data,np.array([an_instance]), axis=0)
    # // endfor

    # remove spam/no spam identifying column at end row from each matrix
    train_spam = np.delete(train_data_spam, -1, axis=1)
    train_notspam = np.delete(train_data_notspam, -1, axis=1)
    test_data = np.delete(testing_data, -1, axis=1)


    # 2. 
    # Create probabilistic model.


    # declare vars for checking prior probability
    spam_train_num = 0
    notspam_train_num = 0
    # check that there is 40% spam
    for spam_train_instance in train_data_spam:
        spam_train_num = spam_train_num + 1
    # check that there is 60% not spam
    for notspam_train_instance in train_data_notspam:
        notspam_train_num = notspam_train_num + 1

    total_train_num = spam_train_num + notspam_train_num

    # calculate percent of spam
    prob_train_spam = spam_train_num / total_train_num
    # calculate percent not spam
    prob_train_notspam = notspam_train_num / total_train_num

    # print prior probability of spam test data
    print("Prior Probability of spam = %f \n" %prob_train_spam)
    # print prior probability of not spam test data
    print("Prior Probability of not spam = %f \n" %prob_train_notspam)

    # Transposed matrices for computation
    ttrain_spam = train_spam.transpose()
    ttrain_notspam = train_notspam.transpose()

    # Vectors to hold mean and stdev of training spam class instances
    train_spam_mean = np.empty((57,0), int)
    train_spam_stdev = np.empty((57,0), int)

    # Compute the mean and std deviation of training spam instances
    for a_feature in ttrain_spam:
        # Compute the mean of each row of ttrain_spam
        a_row_mean = np.array([a_feature.mean()])
        # Compute the standard deviation of each row of ttrain_spam - add .0001 so no zeros
        a_row_stdev = (np.array([a_feature.std()]) + 0.0001)
        # add row mean and std dev to their respective vectors
        train_spam_mean = np.append(train_spam_mean, a_row_mean)
        train_spam_stdev = np.append(train_spam_stdev, a_row_stdev)
    # // endfor

    # Vectors to hold mean and stdev of notspam instances
    train_notspam_mean = np.empty((57,0),int)
    train_notspam_stdev = np.empty((57,0),int)

    # Compute the mean and std deviation of ttrain_nospam
    for b_feature in ttrain_notspam:
        # Compute the mean of each row ttrain_notspam
        b_row_mean = np.array([b_feature.mean()])
        # Compute the standard deviation ttrain_notspam - add .0001 for no zeros
        b_row_stdev = (np.array([b_feature.std()]) + .0001)
        # add row mean and std dev to their respective vectors
        train_notspam_mean = np.append(train_notspam_mean, b_row_mean)
        train_notspam_stdev = np.append(train_notspam_stdev, b_row_stdev)
    # // endfor


    # 3. 
    # Run Naive Bayes on the test data.


    #randomly mix the test data (shuffle columns) 
    shuffled_test = np.zeros(test_data.shape)
    np.take(test_data,np.random.permutation(test_data.shape[0]),axis=0,out=shuffled_test)

    # Results vectors to hold instance probability given spam or not spam
    spam_results = np.zeros(train_spam_stdev.shape)
    notspam_results = np.zeros(train_spam_mean.shape)

    # Vars to hold spam count during testing
    test_spam_count = 0
    test_notspam_count = 0

    # Use gaussian naive bayes to classify instances in data set
    for test_i in shuffled_test:
        # Calculate probability(spam)
        spam_results = -(np.log(math.sqrt(2*np.pi)*train_spam_stdev))-(((test_i - train_spam_mean)**2)/(2*(train_spam_stdev**2)))
        # Calculate probability(not spam)
        notspam_results = -(np.log(math.sqrt(2*np.pi)*train_notspam_stdev))-(((test_i - train_notspam_mean)**2)/(2*(train_notspam_stdev**2)))
        # calculate if prediction spam or notspam is greater
        spam_max = (sum(spam_results)) + (np.log(prob_train_spam))
        notspam_max = (sum(notspam_results)) + (np.log(prob_train_notspam))
        if(spam_max > notspam_max):
            test_spam_count = test_spam_count + 1
        else:
            test_notspam_count = test_notspam_count + 1

    # output results
    total_tested = test_spam_count + test_notspam_count
    percent_spam_predicted = test_spam_count/total_tested
    percent_notspam_predicted = test_notspam_count/total_tested
    #print(f"Total spam predicted = {test_spam_count}\n")
    #print(f"Total notspam predicted = {test_notspam_count}\n")
    print(f"Percentage of spam predicted in test: {percent_spam_predicted}\n")
    print(f"Percentage of not spam predicted in test: {percent_notspam_predicted}\n")

    #STILL NEED TO DO
    #accuracy
    #precision
    #recall
    #confusion matrix





#   KEEP ALL BELOW FOR TESTING PURPOSES AS NEEDED UNTIL COMPLETION
    #print("\n\n")
    #print("")

    # Spambase data read in from csv file
    #print(f"spambase_data: \n{spambase_data}\n") # for testing 00.
    # Shuffled spambase data
    #print(f"shuffled_base: \n{shuffled_base}\n") # for testing 0.

    # Original matrices created to store training and testing data
    #print(f"train_data_spam: \n{train_data_spam}\n") # for testing 1.
    #print(f"train_data_notspam: \n{train_data_notspam}\n") # for testing 2.
    #print(f"testing_data: \n{testing_data}\n") # for testing 3.

    # Arrays removed spam/no spam identifying column at end row from each matrix
    #print(f"train_spam: \n{train_spam}\n") # for testing 4.
    #print(f"train_notspam: \n{train_notspam}\n") # for testing 5.
    #print(f"test_data: \n{test_data}\n") # for testing 6.
    
    # Transposed matrices for computation
    #print(f"ttrain_spam: \n{ttrain_spam}\n") # for testing 7.
    #print(f"ttrain_notspam: \n{ttrain_notspam}\n") # for testing 8.
    #print(f"ttest_data: \n{ttest_data}\n") # for testing 9.

    # Vectors to hold mean and stdev of training spam class instances
    #print(f"train_spam_mean: \n{train_spam_mean}\n") # for testing 10.
    #print(f"train_spam_stdev: \n{train_spam_stdev}\n") # for testing 11.
    #print(f"train_notspam_mean: \n{train_notspam_mean}\n") # for testing 12.
    #print(f"train_notspam_stdev: \n{train_notspam_stdev}\n") # for testing 13.

    # Shuffled test data
    #print(f"shuffled_test: \n{shuffled_test}\n") # for testing 14.

    #print(f"spam_results: \n{spam_results}\n") # for testing 15.
    #print(f"notspam_results: \n{notspam_results}\n") # for testing 16.

    #num_rows = name_data.shape[0]
    #num_cols = name__data.shape[1]
    #print(f"num rows: {num_rows}\n")
    #print(f"num cols: {num_cols}\n")

    #print("")
    print("\n\n")

#    for notes below only
#    test_labels = spambase_data[:,-1]

#    randomly shuffle data by rows
    #np.take(test_data,np.random.permutation(X.shape[0]),axis=0,out=X)
    #np.random.shuffle(arrayToShuffle) #This is an option

    # works as well
        #a_row_mean = statistics.mean(a_feature)
        #a_row_stdev = statistics.stdev(a_feature, xbar=a_row_mean)+.0001

