# Driver for running

import NaiveBayes as nb
import os
import sys


def main():
    while True:
        print("1    ->      Run the program\n"
                  "2    ->      View the log file\n"
                  "3    ->      Exit the program")
        selection = input("What would you like to do:   ")
        if selection == '1':
            run()
        elif selection == '2':
            view_log()
        elif selection == '3':
            sys.exit(0)
        else:
            print("INVALID SELECTION")
        os.system("clear")

def run():
    file, training, test = get_input()
    print("File Being Used is " + file)
    print("Ratio of Training Data Being Used: " + str(training))
    print("Ratio of Testing Data Being Used: " + str(test))
    cont = input("Continue with this data? (Y/n)")
    if cont == "n":
        run()
    print("IMPORTING  DATA")
    data = nb.get_data_set(file)
    print("CLEANING DATA")
    data = nb.clean_data_set(data)
    print("IMPORTING CLEANED DATA")
    data = nb.get_data_set(data)
    print("SPLITTING THE DATA INTO TRAINING AND TEST DATA")
    training_data, testing_data = nb.split_data(data, training)
    print("LENGTH OF TRAINING DATA  ->  " + str(len(training_data)))
    print("LENGTH OF TESTING DATA   ->  " + str(len(testing_data)))
    print("CREATING CLASS SUMMARY")
    summary = nb.class_summary(training_data)
    print("MAKING PREDICTIONS ON TESTING DATA BASED OFF OF MODEL")
    testing = nb.prediction(summary, testing_data)
    accuracy, right, items = nb.accuracy(testing_data, testing)
    print("ACCURACY IS  ->  {:2}".format(accuracy))
    print("AMOUNT CORRECT IS    ->  {}".format(right))
    print("OUT OF       ->      {}".format(items))
    print("WRITING TO LOG")
    log = open("log.txt", "w")
    log.write("ACCURACY ->  " + str(accuracy) + "\n" +
                        "AMOUNT CORRECT     ->      " + str(right) + "\n" +
                        "OUT OF     ->      " + str(items))
    log.close()

def get_input():
    """gets input for ratio used for the data file to be used
        and breaking up testing and training data"""
    while True:
        # hard coded in information for the sake of easy running and testing
        filename = 'fullset.txt' #input("Enter File Name:      ")
        training_ratio = 0.8 # float(input("Enter Training Data Ratio (ie. 0.75 or 0.8    "))
        testing_ratio = 0.2 #float(input("Enter Test Data Ratio (ie. .25 or .20):  "))
        if training_ratio + testing_ratio == 1.0:
            if os.path.exists(filename):    # checks to make sure that it is a valid entry
                return filename, training_ratio, testing_ratio
            else:
                print("File Does Not Exist")
        else:
            print("Does Not Add to 100% Of The File")

def view_log():
    log = open("log.txt")
    reader = log.read()
    data = reader.split("\n")
    print("{}\n{}\n{}".format(data[0], data[1], data[2]))
    log.close()

if __name__ == main():
    main()