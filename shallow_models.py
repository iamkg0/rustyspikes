from model import *

def m_v0():
    snn = SNNModel()
    rt=60
    in0 = Spikes_at_will(awaiting_time=2, refresh_time=rt)
    in1 = Spikes_at_will(awaiting_time=6, refresh_time=rt)
    in2 = Spikes_at_will(awaiting_time=8, refresh_time=rt)
    in3 = Spikes_at_will(awaiting_time=14, refresh_time=rt)
    in4 = Spikes_at_will(awaiting_time=40, refresh_time=rt)
    out = Izhikevich()
    syn0 = Synapse(in0, out, scale=10)
    syn1 = Synapse(in1, out, scale=10)
    syn2 = Synapse(in2, out, scale=10)
    syn3 = Synapse(in3, out, scale=10)
    syn4 = Synapse(in4, out, scale=10)
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