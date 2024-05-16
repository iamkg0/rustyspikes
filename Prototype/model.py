import networkx as nx
from neurons import *
from synaptics import *
from utils import *

class SNNModel:
    def __init__(self, **kwargs):
        self.types_of_neurons = {"Spikes_at_will": Spikes_at_will,
                                 "Izhikevich": Izhikevich,
                                 "Probability_neuron": Probability_neuron}
        self.types_of_synapses = {"Vanilla": Synapse,
                                  "Delayed": None}
        self.types_of_connections = {"FC": self.fc,
                                     "Rand": self.rc}
        self.layers = {}
        self.neurons = {}
        self.synapses = {}
        self.edge_by_syn = {}
        self.syn_by_edge = {}
        self.graph = nx.DiGraph()
        self.color_map = []
        self.color_set = kwargs.get("color_set", ("red", "green", "blue", "yellow", "cyan", "pink", "brown"))

        '''
        Get info:
        '''
    def __getitem__(self, idx):
        return self.neurons[idx]
    
    def show_config(self):
        conf = {}
        conf['Neurons'] = self.neurons
        conf['synapses'] = self.syn_by_edge
        return conf
    
    def get_neuron_stats(self, id):
        v = self.neurons[id].get_voltage_dynamics()
        impulse = self.neurons[id].get_output_current()
        I = self.neurons[id].get_input_current()
        return v, impulse, I
    
    def get_weight(self, id):
        return self.syn_by_edge[id].get_weight()
    
    def get_graph(self):
        return self.graph
   
    '''
    Handling the model:
    '''
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
        # TO DO: Edges
        for syn_layer in cfg_syn:
            temp = self.extract_syn_layer_props(cfg_syn[syn_layer])
            con_type, synapse_type, between_layers, number_connections = temp
            if between_layers == "False":
                self.connect_inside(syn_layer, synapse_type, number_connections)
            else:
                part_l = self.get_particular_layers(between_layers)
                self.connect_between(part_l, synapse_type, self.types_of_connections[con_type], number_connections)
        self.update_weights()
    
    def update_weights(self):
        for syns in self.syn_by_edge:
            self.graph[self.neurons[syns[0]]][self.neurons[syns[1]]]['weight'] = self.syn_by_edge[syns].get_weight()

    def tick(self):
        for i in self.neurons:
            self.neurons[i].dynamics()
            self.neurons[i].apply_cum_current()
        for j in self.syn_by_edge:
            self.syn_by_edge[j].forward()

    

    '''
    Misc:
    '''
    def add_neuron(self, neuron, id=None):
        if not id:
            id = len(self.neurons)
            neuron.id = id
            print(id)
        self.neurons[id] = neuron

    def add_synapse(self, synapse):
        self.syn_by_edge[synapse.get_ids()] = synapse

    def reload_graph(self):
        self.graph = nx.DiGraph()
        for i in self.neurons:
            self.graph.add_node(self.neurons[i])
        for j in self.syn_by_edge:
            self.graph.add_edge(*self.get_neurons_by_edge(j), weight=self.syn_by_edge[j].get_weight())
            # self.neurons[self.syn_by_edge[j].get_ids()[0]], self.neurons[self.syn_by_edge[j].get_ids()[1]]

    def get_neurons_by_edge(self, edge):
        return self.neurons[edge[0]], self.neurons[edge[1]]

    def extract_syn_layer_props(self, syn_layer):
        props = []
        for i in syn_layer:
            props.append(syn_layer[i])
        return props

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

    def get_particular_layers(self, layers):
        ls = layers.split(' ')
        return ls
    
    def connect_inside(self, layer_req, syn_type, num_con):
        nodes = self.layers[layer_req]
        edges = []
        for pre_node_i in range(len(nodes)):
            temp = nodes[:pre_node_i]
            temp_1 = nodes[pre_node_i+1:]
            post_nodes = random.sample(temp + temp_1, num_con)
            for j in post_nodes:
                edge = nodes[pre_node_i], j
                edges.append(edge)
                self.create_synapse(nodes[pre_node_i], j, syn_type)

    def connect_between(self, layers_req, syn_type, method, num_con):
        pre = self.layers[layers_req[0]]
        post = self.layers[layers_req[1]]
        method(pre, post, syn_type, num_con)

    def fc(self, pre, post, syn_type, num_con):
        stack = []
        syns = []
        for i in pre:
            for j in post:
                stack.append((i, j))
                syns.append(self.create_synapse(i, j, syn_type))
        return stack, syns
    
    def create_synapse(self, pre_id, post_id, syn_type):
        syn = self.types_of_synapses[syn_type](self.neurons[pre_id.get_id()], self.neurons[post_id.get_id()])
        self.graph.add_edge(self.neurons[pre_id.get_id()], self.neurons[post_id.get_id()], weight=syn.get_weight())
        self.syn_by_edge[(pre_id.get_id(), post_id.get_id())] = syn
        return syn

    def rc(self, pre, post, syn_type, num_con):
        stack = []
        syns = []
        for i in range(num_con):
            pre_chosen = random.choice(pre)
            post_chosen = random.choice(post)
            stack.append((pre_chosen, post_chosen))
            syns.append(self.create_synapse(pre_chosen, post_chosen, syn_type))

    def empty_function(self):
        pass

    '''
    For advanced visuals:
    '''    
    def spit_for_pyvis(self):
        new_graph = nx.DiGraph()
        for i in self.neurons:
            new_graph.add_node(self.neurons[i].get_id(), title=str(self.neurons[i].get_id()))
            print(self.neurons[i].get_id())
        for j in self.syn_by_edge:
            new_graph.add_edge(self.get_neurons_by_edge(j)[0].get_id(), self.get_neurons_by_edge(j)[1].get_id(), weight=self.syn_by_edge[j].get_weight())
        return new_graph