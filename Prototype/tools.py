import numpy as np

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