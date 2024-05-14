import networkx as nx
from neurons import *
from synaptics import *
from utils import *


class SNNModel:
    def __init__(self, **kwargs):
        self.incidence_matrix = []
        self.in_layer = []
        self.hid_layer = []
        self.out_layer = []
        self.types_of_neurons = {"Spikes_at_will": Spikes_at_will,
                                 "Izhikevich": Izhikevich,
                                 "Probability_neuron": Probability_neuron}
        self.types_of_synapses = {"FC": Synapse,
                                  "Random": Synapse,
                                  "None": None}
        self.layers = {}
        self.neurons = {}
        self.graph = nx.DiGraph()
        self.color_map = []
        self.color_set = kwargs.get("color_set", ("red", "green", "blue", "yellow", "cyan", "pink", "brown"))

    def __getitem__(self, idx):
        return self.neurons[idx]
    
    def generate_model(self, config):
        cfg = read_config(config)
        id = 0
        cfg_neu = cfg["Neurons"]
        cfg_syn = cfg["Synapses"]
        c_indx = 0
        for neural_layer in cfg_neu:
            layer, id = self.create_layer(id, cfg_neu[neural_layer], color=self.color_set[c_indx])
            self.layers[neural_layer] = layer
            c_indx += 1
        self.idxs()
        print(self.graph)
        


    def create_neuron(self, id, type, I, preset, color, resolution=.1):
        neuron = self.types_of_neurons[type](preset=preset, id=id, I=I, resolution=resolution)
        self.graph.add_node(neuron)
        self.color_map.append(color)
        return neuron

    def create_layer(self, id, cfg_neu, color):
        neu_type = cfg_neu["Type"]
        neu_num = cfg_neu["Number"]
        neu_Is = cfg_neu["I"]
        neu_preset = cfg_neu["Preset"]
        layer = []
        if type(neu_Is) is not list:
            temp = neu_Is
            neu_Is = [temp for i in range(neu_num)]
        for i in range(neu_num):
            layer.append(self.create_neuron(id, neu_type, neu_Is[i], neu_preset, color))
            id += 1
        return layer, id
    
    def idxs(self):
        idx = 0
        for i in self.graph:
            self.neurons[idx] = i
            idx += 1

    def set_colors_nodes(self, colors=('red', 'green', 'blue')):
        for i in self.neurons:
            pass
    
    def create_collection_edges(self, cfg_syn):
        pass