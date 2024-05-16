import numpy as np
import matplotlib.pyplot as plt
import json

def read_config(config):
    with open(config) as f:
        data = f.read()
    return json.loads(data)
