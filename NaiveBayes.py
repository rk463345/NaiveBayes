# Naive Bayes

import csv
import math

def get_data_set(filename):
    """loads in the the data set to be analyzed"""
    if filename == "clean_set.csv":
        csv.register_dialect('trades', delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        data = csv.reader(open(filename), 'trades')
        data_set = list(data)
        return data_set
    else:
        csv.register_dialect('trades', delimiter=',', quoting=csv.QUOTE_MINIMAL)
        data = csv.reader(open(filename), 'trades')
        data_set = list(data)
        return data_set

def clean_data_set(data_set):
    default = "clean_set.csv"
    strings = ""
    del data_set[0]
    for trade in data_set:
        if trade[0] == 'Open' or trade[0] == 'Date':
            data_set.remove(trade)
        else:
            del trade[0]
            strings += str(trade).replace("]", "").replace("[", "").replace("\'", "") + "\n"
    clean = open(default, "w")
    clean.write(strings)
    clean.close()
    return default

def convert_data(data_set):
    """converts data from string into int/float"""
    for trade in data_set:
            for index in range(0, len(trade)):
                trade[index] = float(trade[index])
    return data_set

def split_data(data, ratio):
    """splits the data in to training and testing data"""
    ratio = int(len(data) * ratio)
    training_data = data[0:ratio]
    testing = data[ratio:-1]
    return training_data, testing

def separate(data):
    """separates the data by class"""
    sep = dict()
    for item in range(len(data)):
        location = data[item]
        if location[-2] not in sep:
            sep[location[-2]] = []
        sep[location[-2]].append(location)
    return sep

def mean(num):
    """calculates the mean"""
    avg = sum(num)/len(num)
    return avg

def standard_deviation(num):
    """calculates the standard deviation"""
    avg = mean(num)
    var = sum([pow(i - avg, 2) for i in num])/float(len(num))
    stan_dev = math.sqrt(var)
    return stan_dev

def summary(data):
    """creates a summary of the data"""
    s_sum = [(mean(attrib), standard_deviation(attrib)) for attrib in zip(*data)]
    del s_sum[-2]
    return s_sum

def class_summary(data):
    """builds a class summary"""
    sep = separate(data)
    sep_summary = {}
    for value, instance in sep.items():
        sep_summary[value] = summary(instance)
    return sep_summary

def probability(item, p_mean, stan_dev):
    """calculates the probability"""
    if stan_dev > 0:
        exp = math.exp(-(math.pow(item[-2]-p_mean, 2)/(2*math.pow(stan_dev, 2))))
        prob = (1 / (math.sqrt(2*math.pi) * stan_dev)) * exp
        return prob
    else:
        return 0

def class_probability(c_summary, vector):
    """calculates the probability for each class"""
    prob = {}
    for value in c_summary:
        prob[value] = 1
        counter = 0
        for item in c_summary[value]:
            n_mean, stan_dev = item
            selection = vector[counter]
            prob[value] *= probability(selection, n_mean, stan_dev)
            counter += 1
    return prob

def prediction(p_summary, vector):
    """makes a prediction based off of class  probability in the vector"""
    prob = class_probability(p_summary, vector)
    label, new_prob = None, -1
    for key in prob:
        n_probability = prob[key]
        value = key
        if label is None or n_probability > new_prob:
            new_prob = n_probability
            label = value
    return label

def make_prediction(p_sum, test):
    """calls prediction to make a list of predictions based off of probability """
    predictions = []
    for item in range(len(test)):
        result = prediction(p_sum, test[item])
        predictions.append(result)
    return predictions

def accuracy(test, predict):
    """reports the accuracy of predictions made"""
    right = 0
    items = 0
    for item in range(0, len(test)):
        items += 1
        if test[item][-1]== predict:
            right += 1
    percent = (right/len(test)) * 100
    return percent, right, items
