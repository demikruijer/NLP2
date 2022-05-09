# Implement four baselines for the task.
# Majority baseline: always assigns the majority class of the training data
# Random baseline: randomly assigns one of the classes. Make sure to set a random seed and average the accuracy over 100 runs.
# Length baseline: determines the class based on a length threshold
# Frequency baseline: determines the class based on a frequency threshold
from collections import Counter
from statistics import mean
import pandas as pd
import numpy as np


# Each baseline returns predictions for the test data. The length and frequency baselines determine a threshold using the development data.

def majority_baseline(train_sentences, train_labels, testinput, testlabels):

     # Determine the majority class based on the training data
    counterOff = 0
    counterNot = 0
    for labels in train_labels:
        if labels == 1:
            counterOff += 1
        elif labels == 0:
            counterNot += 1
    if counterOff > counterNot:
        majority_class = 1
    if counterNot > counterOff:
        majority_class = 0

    predictions = []

    for instance in testinput:
        predictions.append(majority_class)

    # Calculate accuracy for the test input
    labelcountOff = 0
    labelcountNot = 0

    for labels in testlabels:
        if labels == 1:
            labelcountOff += 1
        elif labels == 0:
            labelcountNot += 1

    if majority_class == 1:
        accuracy = labelcountOff / (labelcountOff + labelcountNot)

    if majority_class == 0:
        accuracy = labelcountNot / (labelcountOff + labelcountNot)

    print(accuracy, predictions)
    return accuracy, predictions


def random_baseline(train_sentences, train_labels, testinput, test_labels, B = 100):
    np.random.seed(2020)
    predictions = []
    correct = []
    accuracy = []

    for b in range(B):
        for i, instance in enumerate(testinput):
            instance_predictions = np.random.choice([1, 0])
            predictions.append(instance_predictions)
            if test_labels[i] == instance_predictions:
                correct.append(1)
            else: correct.append(0)

        accuracy.append(sum(correct) / len(correct))

    print(accuracy)
    accuracy = mean(accuracy)
    print(accuracy)
    return accuracy, predictions


# TODO: output the predictions in a suitable way so that you can evaluate them
def write_output(outfile, predictions, labels, input):

    with open(outfile, "w", encoding='utf8') as f:
        for i, sentences in enumerate(input):
            f.write("\t".join([sentences, str(labels[i]), str(predictions[i])]))
            f.write("\n")
    return


if __name__ == '__main__':
    train_path = "data/olid-train.csv"
    dev_path = "data/olid-subset-diagnostic-tests.csv"
    test_path = "data/olid-test.csv"

    # Note: this loads all instances into memory. If you work with bigger files in the future, use an iterator instead.

    with open(train_path, encoding='utf8') as train_file:
        train_data = pd.read_csv(train_file)
        train_sentences = train_data["text"]
        train_labels = train_data["labels"]

    with open(test_path, encoding='utf8') as test_file:
        test_data = pd.read_csv(test_file)
        test_sentences = test_data["text"]
        test_labels = test_data["labels"]

    # Baselines
    # Majority baseline
    majority_accuracy, majority_predictions = majority_baseline(train_sentences, train_labels, test_sentences, test_labels)
    write_output("majority_baseline.tsv", majority_predictions, test_labels, test_sentences)

    # Random Baseline
    majority_accuracy, majority_predictions = random_baseline(train_sentences, train_labels, test_sentences,
                                                                test_labels)
    write_output("random_baseline.tsv", majority_predictions, test_labels, test_sentences)


