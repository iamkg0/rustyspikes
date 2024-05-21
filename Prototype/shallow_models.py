from model import *

def net_v0():
    '''
    5-to-1 test subject
    '''
    snn = SNNModel()
    rt=50
    scale = 10
    in0 = Spikes_at_will(awaiting_time=2, refresh_time=rt)
    in1 = Spikes_at_will(awaiting_time=6, refresh_time=rt)
    in2 = Spikes_at_will(awaiting_time=10, refresh_time=rt)
    in3 = Spikes_at_will(awaiting_time=14, refresh_time=rt)
    in4 = Spikes_at_will(awaiting_time=16, refresh_time=rt)
    out = Izhikevich()
    syn0 = Synapse(in0, out, scale=scale)
    syn1 = Synapse(in1, out, scale=scale)
    syn2 = Synapse(in2, out, scale=scale)
    syn3 = Synapse(in3, out, scale=scale)
    syn4 = Synapse(in4, out, scale=scale)
    snn.add_neuron(in0)
    snn.add_neuron(in1)
    snn.add_neuron(in2)
    snn.add_neuron(in3)
    snn.add_neuron(in4)
    snn.add_neuron(out)
    snn.add_synapse(syn0)
    snn.add_synapse(syn1)
    snn.add_synapse(syn2)
    snn.add_synapse(syn3)
    snn.add_synapse(syn4)
    snn.reload_graph()
    return snn

def single_synapse():
    '''
    1-to-1 test subject
    '''
    snn = SNNModel()
    rt = 100
    input = Spikes_at_will(awaiting_time=2, refresh_time=rt)
    output = Spikes_at_will(awaiting_time=5, refresh_time=rt)
    syn = Synapse(input, output, scale=1)
    snn.add_neuron(input)
    snn.add_neuron(output)
    snn.add_synapse(syn)
    snn.reload_graph()
    return snn

def delayed_single():
    '''
    1-to-1 delayed synapse test
    '''
    dsnn = SNNModel()
    rt = 20
    input = Spikes_at_will(awaiting_time=10, refresh_time=rt, synaptic_limit=1)
    #output = Spikes_at_will(awaiting_time=15, refresh_time=rt, synaptic_limit=1)
    output = Izhikevich()
    d_syn = Delayed_synapse(input, output, scale = 40, delay=50)
    dsnn.add_neuron(input)
    dsnn.add_neuron(output)
    dsnn.add_synapse(d_syn)
    dsnn.reload_graph()
    return dsnn

def delayed_3_to_1():
    dsnn = SNNModel()
    rt = 30
    in0 = Spikes_at_will(awaiting_time=2, refresh_time=rt, synaptic_limit=1)
    in1 = Spikes_at_will(awaiting_time=12, refresh_time=rt, synaptic_limit=1)
    in2 = Spikes_at_will(awaiting_time=22, refresh_time=100, synaptic_limit=1)
    out = Izhikevich(synaptic_limit=1)
    syn0 = Delayed_synapse(in0, out, scale=40, delay=5)
    syn1 = Delayed_synapse(in1, out, scale=40, delay=50)
    syn2 = Delayed_synapse(in2, out, scale=40, delay=150)
    dsnn.add_neuron(in0)
    dsnn.add_neuron(in1)
    dsnn.add_neuron(in2)
    dsnn.add_neuron(out)
    dsnn.add_synapse(syn0)
    dsnn.add_synapse(syn1)
    dsnn.add_synapse(syn2)
    dsnn.reload_graph()
    return dsnn