# Project:
#   VQA
# Description:
#   Declaration of classifiers for final step of VQA
# Author: 
#   Sergio Tascon-Morales

from torch import nn

def get_classfier(input_size, config):
    hidden_size = config['classifier_hidden_size']
    if config['num_answers'] == 2:
        output_size = 1 # if binary classification, output has one neuron and BCEWithLogitsLoss can be used
    else:
        output_size = config['num_answers']
    dropout_percentage = config['classifier_dropout']

    # create MLP
    classifier = Classifier(input_size, hidden_size, output_size, drop=dropout_percentage)
    return classifier

class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_classes, drop=0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = output_classes

        self.drop1 = nn.Dropout(drop)
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU() 
        self.drop2 = nn.Dropout(drop)
        self.fc2 = nn.Linear(self.hidden_size, self.num_classes)

        # apply softmax here?

    def forward(self, x):
        x = self.fc1(self.drop1(x))
        x = self.relu(x)
        x = self.fc2(self.drop2(x))

        return x
