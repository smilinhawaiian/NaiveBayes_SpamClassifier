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
    print("\n")
    # Read in and create matrice of data
    spambase_data = pd.read_csv("spambase/spambase_copy.csv", header=None).values

    ##print(f"spambase_data: \n{spambase_data}") # for testing 00.
    # split row instances into spam and not spam for training
    # mixed for testing - keeping same proportions test/train sets

    # counts for making sure data is split evenly
    spam_count = -1
    notspam_count = -1

    # arrays for test and train data
    train_data_spam = np.empty((0,58), int)
    train_data_notspam = np.empty((0,58), int)
    testing_data = np.empty((0,58), int)

    #** add randomization of spambase data array here
    #np.random.shuffle(arrayToShuffle) #This is an option
    shuffled_base = np.zeros(spambase_data.shape)
    np.take(spambase_data,np.random.permutation(spambase_data.shape[0]),axis=0,out=shuffled_base)

    ##print(f"spambase_data: \n{spambase_data}") # for testing 00.
    ##print(f"shuffled_base: \n{shuffled_base}") # for testing 0.

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

    # declare vars for checking prior probability
    spam_test_num = 0
    notspam_test_num = 0
    # check that there is 40% spam
    for spam_test_instance in train_data_spam:
        spam_test_num = spam_test_num + 1
    # check that there is 60% not spam
    for notspam_test_instance in train_data_notspam:
        notspam_test_num = notspam_test_num + 1

    total_test_num = spam_test_num + notspam_test_num

    # calculate percent of spam
    prob_test_spam = spam_test_num / total_test_num
    # calculate percent not spam
    prob_test_notspam = notspam_test_num / total_test_num

    # print prior probability of spam test data
    print("Prior Probability of spam = %f \n" %prob_test_spam)
    # print prior probability of not spam test data
    print("Prior Probability of not spam = %f \n" %prob_test_notspam)

    #print("total spam examples = %d\n" %spam_count) #for testing
    #print("test spam examples = %d\n" %spam_test_num) #for testing

    # Transposed matrices for computation
    ttrain_spam = train_spam.transpose()
    ttrain_notspam = train_notspam.transpose()
    ttest_data = test_data.transpose()

    # Vectors to hold mean and stdev of training spam class instances
    train_spam_mean = np.empty((57,0), int)
    train_spam_stdev = np.empty((57,0), int)

    # Compute the mean and std deviation of training spam instances
    for a_feature in ttrain_spam:
        # Compute the mean of each row of ttrain_spam
        a_row_mean = np.array([a_feature.mean()])
        # Compute the standard deviation of each row of ttrain_spam
        # add 0.0001 to stdev to ensure there are no zeros
        a_row_stdev = (np.array([a_feature.std()]) + 0.0001)
        # add row mean and std dev to their respective vectors
        train_spam_mean = np.append(train_spam_mean, a_row_mean)
        train_spam_stdev = np.append(train_spam_stdev, a_row_stdev)
    # // endfor

    #print("")
    #print(f"train_spam_mean: \n{train_spam_mean}") # for testing
    #print(f"train_spam_stdev: \n{train_spam_stdev}") # for testing

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

# Randomly mix all the test data
# Use Gaussian Naive Bayes algorithm to classify instances in your test set,
# using the means and standard deviations computed in part 2.
# P(x_i | c_j) = N(x_i ; m_(i,c_j) , stdev_(i,c_j) )
# where
# N(x ; m , stdev) = [ (1/ (sqrt(2pi)*stdev) )*( e^ ((x-m)^2 / (2*o^2) ) ) ]# *Because product of 58 probabilities will be small, use log of product
#   since argmax f(z) = argmax log(f(z))

    #randomly mix the test data
    #np.take(test_data,np.random.permutation(X.shape[0]),axis=0,out=X)





#   KEEP ALL BELOW FOR TESTING PURPOSES AS NEEDED UNTIL COMPLETION
    #print("\n\n")
    #print("")

    # Spambase data read in from csv file
    #print(f"spambase_data: \n{spambase_data}") # for testing 00.
    # Shuffled spambase data
    #print(f"shuffled_base: \n{shuffled_base}") # for testing 0.

    # Original matrices created to store training and testing data
    #print(f"train_data_spam: \n{train_data_spam}") # for testing 1.
    #print(f"train_data_notspam: \n{train_data_notspam}") # for testing 2.
    #print(f"testing_data: \n{testing_data}") # for testing 3.

    # Arrays removed spam/no spam identifying column at end row from each matrix
    #print(f"train_spam: \n{train_spam}") # for testing 4.
    #print(f"train_notspam: \n{train_notspam}") # for testing 5.
    #print(f"test_data: \n{test_data}") # for testing 6.
    
    # Transposed matrices for computation
    #print(f"ttrain_spam: \n{ttrain_spam}") # for testing 7.
    #print(f"ttrain_notspam: \n{ttrain_notspam}") # for testing 8.
    #print(f"ttest_data: \n{ttest_data}") # for testing 9.

    # Vectors to hold mean and stdev of training spam class instances
    #print(f"train_spam_mean: \n{train_spam_mean}") # for testing 10.
    #print(f"train_spam_stdev: \n{train_spam_stdev}") # for testing 11.
    #print(f"train_notspam_mean: \n{train_notspam_mean}") # for testing 12.
    #print(f"train_notspam_stdev: \n{train_notspam_stdev}") # for testing 13.

    #print("")
    print("\n\n")

#    for notes below only
#    test_labels = spambase_data[:,0]

