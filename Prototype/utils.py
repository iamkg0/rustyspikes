import numpy as np
import matplotlib.pyplot as plt
import json
import pickle as pkl
import copy

def read_config(config):
    # LEGACY (and useless)
    with open(config) as f:
        data = f.read()
    return json.loads(data)


def save_model(model, path):
    with open(path, 'wb') as saving:
        pkl.dump(model, saving, pkl.HIGHEST_PROTOCOL)
        print('Model saved at ' + path)

def load_model(path):
    with open(path, 'rb') as loading:
        model = pkl.load(loading)
        print('Loaded the following model:')
        print(model.show_config())
    return model

def clone_model(model):
    return copy.deepcopy(model)