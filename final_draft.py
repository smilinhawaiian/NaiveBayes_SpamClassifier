# Sharice Mayer
# 2/23/19
# Machine Learning
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
# N(x ; m , stdev) = [ (1/ (sqrt(2pi)*stdev) )*( e^ ((x-m)^2 / (2*o^2) ) ) ]
# *Because product of 58 probabilities will be small, use log of product
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

    #print(spambase_data)

    # counts for making sure data is split evenly
    spam_count = -1
    notspam_count = -1

    # arrays for test and train data
    train_data_spam = np.empty((0,58), int)
    train_data_notspam = np.empty((0,58), int)
    test_data = np.empty((0,58), int)

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
                test_data = np.append(test_data,np.array([an_instance]), axis=0)
        # instance not spam
        else:
            # increment no_spam counter
            notspam_count = notspam_count + 1
            # if counter is even, add row to train_data_nospam
            if(notspam_count%2 ==0):
                train_data_notspam = np.append(train_data_notspam, np.array([an_instance]), axis=0)
            # if count is odd, add row to test_data
            else:
                test_data = np.append(test_data,np.array([an_instance]), axis=0)
    # // endfor

    # remove spam/no spam identifying column at end row from each matrix
    train_spam = np.delete(train_data_spam, -1, axis=1)
    train_notspam = np.delete(train_data_notspam, -1, axis=1)


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
    print("Prior Probability of not spam = %f " %prob_train_notspam)

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

    # remove spam label and store for comparison later(True/False Positive/Negative)
    test_labels = shuffled_test[:,-1]
    shuffled_test = np.delete(shuffled_test, -1, axis=1)

    # Results vectors to hold instance probability given spam or not spam
    spam_results = np.zeros(train_spam_stdev.shape)
    notspam_results = np.zeros(train_spam_mean.shape)

    # Vars to hold spam count during testing
    test_spam_count = 0
    test_notspam_count = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    index = 0
    # Use gaussian naive bayes to classify instances in data set
    for test_i in shuffled_test:
        # Calculate prediction(spam)
        spam_results = -(np.log(math.sqrt(2*np.pi)*train_spam_stdev))-(((test_i - train_spam_mean)**2)/(2*(train_spam_stdev**2)))
        # Calculate prediction(not spam)
        notspam_results = -(np.log(math.sqrt(2*np.pi)*train_notspam_stdev))-(((test_i - train_notspam_mean)**2)/(2*(train_notspam_stdev**2)))
        # calculate if prediction spam or notspam is greater
        spam_max = (sum(spam_results)) + (np.log(prob_train_spam))
        notspam_max = (sum(notspam_results)) + (np.log(prob_train_notspam))
        if(spam_max > notspam_max):
            # check if labeled spam
            if(test_labels[index] == 1): # TruePositive
                true_positive +=1
            else: # FalsePositive
                false_positive +=1
            # increment spam count
            test_spam_count = test_spam_count + 1
        else:
            #check if not spam
            if(test_labels[index] == 0): #TrueNegative
                true_negative +=1
            else: #FalseNegative
                false_negative +=1
            # increment notspam count
            test_notspam_count = test_notspam_count + 1
        index +=1
    # // endfor

    #calculate results for output
    total_tested = test_spam_count + test_notspam_count
    percent_spam_predicted = test_spam_count/total_tested
    percent_notspam_predicted = test_notspam_count/total_tested
    accuracy = (true_positive + true_negative) / total_tested
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    false_positive_rate = false_positive / (true_negative + false_positive)

    #accuracy = number of correct classifications/ total
    #  accuracy = TruePositive + TrueNegative / total examples
    # precision = fraction of predicted "positive" that are actually positive
    #  precision = TruePositive / (TruePositive + FalsePositive)
    # recall = fraction of positive examples predicted as "positive" = true positive rate
    #  recall = TruePositive / (TruePositive + FalseNegative)
    # error rate = 1 - accuracy
    # false positive rate = FalsePositive / TrueNegative + FalsePositive

    #confusion matrix for a given class c:
    #____________________________________________________________________________
    #  Actual                   |   Predicted(or "classified")
    #                           |   Positive            Negative
    #                           |   (in class spam)     (not in class spam)
    #                           |------------------------------------------------
    #  Positive(in class spam)  |   TruePositive        FalseNegative
    #                           |
    #                           |
    #  Negative(not in class)   |   FalsePositive       TrueNegative
    #                           |
    #                           |
    #----------------------------------------------------------------------------

    print(f"\nConfusion Matrix for a given class spam:")
    print(f"|____________________________________________________________________________")
    print(f"|  Actual Test Instances    |    \tPredicted(or 'classified')")
    print(f"|  {total_tested}                     |\tPositive\t\tNegative")
    print(f"|                           |\t(in class spam)\t\t(not in class spam)")
    print(f"|                           |------------------------------------------------")
    print(f"|  Positive(in class spam)  |\tTruePositive\t\tFalseNegative")
    print(f"|                           |\t{true_positive}\t\t\t{false_negative}")
    print(f"|                           |")
    print(f"|  Negative(not in class)   |\tFalsePositive\t\tTrueNegative")
    print(f"|                           |\t{false_positive}\t\t\t{true_negative}")
    print(f"|                           |")
    print(f"|----------------------------------------------------------------------------\n")


    # output results
    print(f"Percent spam predicted after test = {percent_spam_predicted}\n")
    print(f"accuracy = {accuracy}\n")
    print(f"precision = {precision}\n")
    print(f"recall = {recall}\n")
    print(f"false positive rate = {false_positive_rate}\n")



    # calculate for ~spam class
    nprecision = true_negative / (true_negative + false_negative)
    nrecall = true_negative / (true_negative + false_positive)
    nfalse_positive_rate = false_negative / (true_positive + false_negative)

    #confusion matrix for a given class c:
    #____________________________________________________________________________
    #  Actual                   |   Predicted(or "classified")
    #                           |   Positive            Negative
    #                           |   (in class ~spam)    (not in class ~spam)
    #                           |------------------------------------------------
    #  Positive(in class ~spam) |   TruePositive        FalseNegative
    #                           |
    #                           |
    #  Negative(not in class)   |   FalsePositive       TrueNegative
    #                           |
    #                           |
    #----------------------------------------------------------------------------

    print(f"\nConfusion Matrix for a given class ~spam:")
    print(f"|____________________________________________________________________________")
    print(f"|  Actual Test Instances    |    \tPredicted(or 'classified')")
    print(f"|  {total_tested}                     |\tPositive\t\tNegative")
    print(f"|                           |\t(in class ~spam)\t(not in class ~spam)")
    print(f"|                           |------------------------------------------------")
    print(f"|  Positive(in class ~spam) |\tTruePositive\t\tFalseNegative")
    print(f"|                           |\t{true_negative}\t\t\t{false_positive}")
    print(f"|                           |")
    print(f"|  Negative(not in class)   |\tFalsePositive\t\tTrueNegative")
    print(f"|                           |\t{false_negative}\t\t\t{true_positive}")
    print(f"|                           |")
    print(f"|----------------------------------------------------------------------------\n")



    # output results
    print(f"Percent ~spam predicted after test = {percent_notspam_predicted}\n")
    print(f"accuracy = {accuracy}\n")
    print(f"precision = {nprecision}\n")
    print(f"recall = {nrecall}\n")
    print(f"false positive rate = {nfalse_positive_rate}\n")
