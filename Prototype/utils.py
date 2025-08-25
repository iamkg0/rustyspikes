import numpy as np
import matplotlib.pyplot as plt
import json
import pickle as pkl

def read_config(config):
    with open(config) as f:
        data = f.read()
    return json.loads(data)


def save_model(model, path):
    with open(path, 'wb') as saving:
        pkl.dump(model, saving, pkl.HIGHEST_PROTOCOL)

def load_model(path):
    with open(path, 'rb') as loading:
        model = pkl.load(loading)
        print('Loaded the following model:')
        print(model.show_config())
    return model