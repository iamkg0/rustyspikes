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
    for p in range(1, num_rand_patterns+1):
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


def reverse_signal(model, stimulus_size=5):
    '''
    Switches order of SAW neurons spiking to opposite
    Input SAW neurons must have smallest id values
    '''
    timings = []
    for i in range(stimulus_size):
        timings.append(model[i].awaiting_time)
    for i in range(stimulus_size):
        model[i].awaiting_time = timings[stimulus_size - i - 1]


'''
log handler
'''

class LogHandler:
    def __init__(self, path=None, filename=None, delimeter='\t', lineterminator='\n'):
        self.map = {}
        self.data = []
        self.path = str(path)
        self.filename = filename
        self.delim = delimeter
        self.lineterm = lineterminator
        self.df = None

    def create(self):
        '''
        Creates log file
        '''
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        if os.path.isfile(self.path + '\\' + self.filename):
            print('===== WARNING! =====')
            print(f'{self.path}\{self.filename} already exists!')
            print('Data hasnt been overwritten yet!')
            print('NOTICE:')
            print('.define_cols() method will remove all previous data')
            print('.write_sample() method will append new data')
        else:
            with open(self.path + '\\' + self.filename, 'w') as file:
                writer = csv.writer(file, delimiter=self.delim, lineterminator=self.lineterm)
        
    def define_cols(self, cols):
        '''
        Removes all data from file and defines columns
        '''
        with open(self.path + '\\' + self.filename, 'w') as file:
            writer = csv.writer(file, delimiter=self.delim, lineterminator=self.lineterm)
            self.data = [[] for i in range(len(cols))]
            for i in range(len(cols)):
                self.map[cols[i]] = self.data[i]
            writer.writerow(cols)
            print('===== WARNING! =====')
            print('If there was any data, now theres none')
            print('Columns were configured')
            print(f'{self.path}\{self.filename}')
    
    def comment(self, comment):
        with open(self.path + '\\' + self.filename, 'a') as file:
            writer = csv.writer(file, delimiter=',', lineterminator=self.lineterm)
            writer.writerow('#'+str(comment))

    def append_sample(self, sample):
        '''
        Adds data to list
        Doesnt write data to a file
        '''
        for i in range(len(sample)):
            self.data[i].append(sample[i])
        
    def write_sample(self, sample):
        '''
        Adds data to a file
        '''
        if not self.map:
            raise Exception('Columns are unnamed! Data will not be readable!')
        with open(self.path + '\\' + self.filename, 'a') as file:
            writer = csv.writer(file, delimiter=self.delim, lineterminator=self.lineterm)
            writer.writerow(sample)
                
    def read_data(self, **kwargs):
        '''
        kinda useless
        '''
        path = kwargs.get('path', self.path)
        filename = kwargs.get('filename', self.filename)
        delim = kwargs.get('delimeter', self.delim)
        self.df = pd.read_csv(path + '\\' + filename, sep=delim)

    def show_df(self):
        return self.df