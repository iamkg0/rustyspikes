import numpy as np
import os
import csv
import pandas as pd
'''
Sample random inputs
Stores in patterns (dict)
'''
def sampler(num_inputs=10, num_rand_patterns=2):
    num_patterns = num_rand_patterns + 1
    assert num_rand_patterns < np.math.factorial(num_inputs) - 1, 'too many patterns for an input of this size'
    sequence = np.arange(num_inputs) + 1
    patterns = {}
    patterns[0] = sequence.tolist()
    for p in range(1, num_rand_patterns+2):
        seq = sequence.copy()
        np.random.shuffle(seq)
        seq = seq.tolist()
        if seq not in patterns.values():
            patterns[p] = seq
        else:
            while seq in patterns.values():
                np.random.shuffle(seq)
                if seq not in patterns.values():
                    patterns[p] = seq
                    break
    return patterns, num_inputs, num_patterns



'''
log handler
'''

class Log_handler:
    def __init__(self, path, filename):
        self.map = {}
        self.data = []
        self.path = str(path)
        self.filename = filename

    def create(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        if os.path.isfile(self.path+self.filename):
            print('This file already exists')
        else:
            with open(self.path + '\\' + self.filename, 'w') as file:
                writer = csv.writer(file, delimiter='\t',lineterminator='\n')
        
    def define_cols(self, cols):
        with open(self.path + '\\' + self.filename, 'w') as file:
            writer = csv.writer(file, delimiter='\t',lineterminator='\n')
            self.data = [[] for i in range(len(cols))]
            for i in range(len(cols)):
                self.map[cols[i]] = self.data[i]
            writer.writerow(cols)

    def append_sample(self, sample):
        for i in range(len(sample)):
            self.data[i].append(sample[i])
        
    def write_sample(self, sample):
        with open(self.path + '\\' + self.filename, 'a') as file:
            writer = csv.writer(file, delimiter='\t',lineterminator='\n')
            writer.writerow(sample)
                