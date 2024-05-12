import numpy as np

class Synapse:
    def __init__(self, presynaptic, postsynaptic, **kwargs):
        self.presynaptic = presynaptic
        self.postsynaptic = postsynaptic
        self.scale = kwargs.get('scale', 1)
        self.w = 1

    def forward(self):
        self.postsynaptic.apply_current(self.presynaptic.get_output_current() * self.w * 100)

    def pair_stdp(self, lr=.01, asymmetry=5):
        # !!!!!!!!!!!!!!! THERE'S A MISTAKE WITH THE FORMULA !!!!!!!!!!!!!!!!!!!!!
        dw = 0
        if self.presynaptic.get_spike_status():
            dw += self.postsynaptic.get_output_current() * (1 - self.w) * lr
        if self.postsynaptic.get_spike_status():
            dw -= self.presynaptic.get_output_current() * self.w * lr * asymmetry
        self.w += dw

    '''
    Get info:
    '''
    def get_weight(self):
        return self.w