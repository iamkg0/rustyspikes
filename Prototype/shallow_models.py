from model import *

def net_v0():
    '''
    5-to-1 test subject
    '''
    snn = SNNModel()
    rt=100
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
    input = Spikes_at_will(awaiting_time=5, refresh_time=rt, tau=30)
    output = Spikes_at_will(awaiting_time=2, refresh_time=rt, tau=30)
    syn = Synapse(input, output, scale=1)
    snn.add_neuron(input)
    snn.add_neuron(output)
    snn.add_synapse(syn)
    snn.reload_graph()
    return snn

def dr_single():
    snn = SNNModel()
    rt0 = 25
    aw0 = 0
    rt1 = 25
    aw1 = 5
    input = Spikes_at_will(awaiting_time=aw0, refresh_time=rt0, synaptic_limit=1)
    output = Spikes_at_will(awaiting_time=aw1, refresh_time=rt1, synaptic_limit=1)
    syn = Delayed_synapse(input, output, scale=1, delay=100)
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
    d_syn = Delayed_synapse(input, output, scale = 15, delay=50)
    dsnn.add_neuron(input)
    dsnn.add_neuron(output)
    dsnn.add_synapse(d_syn)
    dsnn.reload_graph()
    return dsnn

def delayed_3_to_1():
    dsnn = SNNModel()
    rt = 100
    sc = 5
    in0 = Spikes_at_will(awaiting_time=2, refresh_time=rt, synaptic_limit=1)
    in1 = Spikes_at_will(awaiting_time=6, refresh_time=rt, synaptic_limit=1)
    in2 = Spikes_at_will(awaiting_time=10, refresh_time=rt, synaptic_limit=1)
    out = Izhikevich(synaptic_limit=1)
    syn0 = Delayed_synapse(in0, out, scale=sc, delay=0)
    syn1 = Delayed_synapse(in1, out, scale=sc, delay=0)
    syn2 = Delayed_synapse(in2, out, scale=sc, delay=0)
    dsnn.add_neuron(in0)
    dsnn.add_neuron(in1)
    dsnn.add_neuron(in2)
    dsnn.add_neuron(out)
    dsnn.add_synapse(syn0)
    dsnn.add_synapse(syn1)
    dsnn.add_synapse(syn2)
    dsnn.reload_graph()
    return dsnn

def bfnaics_24_model(num_input, rt=100, scale=10):
    snn = SNNModel()
    neu = []
    for i in range(num_input):
        neu.append(Spikes_at_will(id=i, refresh_time=rt))
        snn.add_neuron(neu[i])
    out = Izhikevich()
    snn.add_neuron(out)
    for i in range(num_input):
        syn = Synapse(neu[i], out, scale=scale)
        snn.add_synapse(syn)
    snn.reload_graph()
    return snn