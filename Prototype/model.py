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
                                  "Delayed": False}
        self.types_of_connections = {"FC": self.fc,
                                     "Rand": self.rc}
        self.layers = {}
        self.neurons = {}
        #self.synapses = {}
        self.l_rules = {}
        self.syn_by_edge = {}
        self.graph = nx.DiGraph()
        self.color_map = []
        self.color_set = kwargs.get("color_set", ("red", "green", "blue", "yellow", "cyan", "pink", "brown"))

        self.output_neurons = {}
        self.preoutput_synapses = {}

        '''
        Get info:
        '''
    def __getitem__(self, idx):
        return self.neurons[idx]
    
    def show_config(self):
        conf = {}
        conf['Neurons'] = self.neurons
        conf['Synapses'] = self.syn_by_edge
        return conf
    
    def get_first_synapse(self):
        return self.syn_by_edge[next(iter(self.syn_by_edge))]
    
    def get_neuron_stats(self, id):
        v = self.neurons[id].get_voltage_dynamics()
        impulse = self.neurons[id].get_output_current()
        I = self.neurons[id].get_input_current()
        return v, impulse, I
    
    def get_weight(self, id):
        return self.syn_by_edge[id].get_weight()
    
    def get_graph(self):
        return self.graph
    
    def get_neurons_by_edge(self, edge):
        return self.neurons[edge[0]], self.neurons[edge[1]]

    def get_presyn_neurons_ids(self, id):
        '''
        Finds presynaptic neurons by id
        Returns list of ids
        '''
        pres = []
        for i in self.syn_by_edge.keys():
            if i[1] == id:
                pres.append(i[0])
        return pres
    
    def get_postsyn_neurons_ids(self, id):
        '''
        Finds postsynaptic neurons by id
        Returns list of ids
        '''
        posts = []
        for i in self.syn_by_edge.keys():
            if i[0] == id:
                posts.append(i[1])
        return posts
    
    def get_neighbors_ids(self, id):
        '''
        Finds all neurons connected to a specific one
        Returns list of ids
        '''
        pres = self.get_presyn_neurons(id)
        posts = self.get_postsyn_neurons(id)
        return [*pres, *posts]
    
    def get_output_neurons(self):
        return self.output_neurons
    
    def get_output_ids(self):
        ids = []
        for i in self.output_neurons.keys():
            ids.append(i)
        return ids
    
    def get_incoming_synapses(self, id):
        '''
        Returns list of adresses of synapses (objects, not ids!!!)
        with presynaptic neurons. In other words, gets input
        connections
        '''
        synapses = []
        presyn_neurons_ids = self.get_presyn_neurons_ids(id)
        for i in presyn_neurons_ids:
            synapses.append(self.syn_by_edge[(i, id)])
        return synapses
    
    def get_outgoing_synapses(self, id):
        synapses = []
        postsyn_neurons_ids = self.get_postsyn_neurons_ids(id)
        for i in postsyn_neurons_ids:
            synapses.append(self.syn_by_edge[(id, i)])
        return synapses

   
    '''
    Handling the model:
    '''
    def generate_model(self, config='config.txt'):
        '''
        Generates the model according to
        certain configuration
        '''
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
        '''
        Updates weight attributes in graph
        '''
        for syns in self.syn_by_edge:
            self.graph[self.neurons[syns[0]]][self.neurons[syns[1]]]['weight'] = self.syn_by_edge[syns].get_weight()

    def tick(self, freeze_delays=False):
        for i in self.neurons:
            self.neurons[i].dynamics()
            self.neurons[i].apply_cum_current()
        for j in self.syn_by_edge:
            self.syn_by_edge[j].forward(freeze_delays)

    def add_neuron(self, neuron, id=None):
        '''
        Adds neuron (object) to the model
        (does not affect graph
        Do not forget to use reaload_graph()
        after the model architecture is complete)
        '''
        if not id:
            id = len(self.neurons)
            neuron.id = id
        self.neurons[id] = neuron

    def add_synapse(self, synapse):
        '''
        Adds synapse (object) to the model
        (does not affect graph
        Do not forget to use reaload_graph()
        after the model architecture is complete)
        '''
        self.syn_by_edge[synapse.get_ids()] = synapse

    def set_rule_to_all(self, rule=None):
        for i in self.syn_by_edge:
            self.syn_by_edge[i].change_learning_rule(rule)

    def set_lr_to_all(self, lr=.1):
        for i in self.syn_by_edge:
            self.syn_by_edge[i].change_lr(lr)

    def set_noise(self, noise=0):
        for i in self.neurons.keys():
            self.neurons[i].noise = noise

    def set_random_weights(self, ranges=(0,1)):
        for i in self.syn_by_edge.keys():
            self.syn_by_edge[i].w = np.random.uniform(*ranges)

    def set_weight_manually(self, id, weight):
        self.syn_by_edge[id].set_weight_manually(weight)

    def set_d_lr(self, d_lr, ids=None):
        if not ids:
            ids = [i for i in self.syn_by_edge.keys()]
        for syn in ids:
            self.syn_by_edge[syn].d_lr = d_lr

    def set_slow_var_limit(self, limit, ids=None):
        if not ids:
            ids = [i for i in self.syn_by_edge.keys()]
        for syn in ids:
            self.syn_by_edge[syn].change_slow_var_limit(limit)

    def set_scale(self, scale):
        '''
        Adjust scale value to all synapses
        '''
        for i in self.syn_by_edge.values():
            i.scale = scale

    def define_output(self, ids):
        for id in ids:
            self.output_neurons[id] = self.neurons[id]
        for i in self.output_neurons.keys():
            self.preoutput_synapses[i] = self.get_incoming_synapses(i)

    
    '''
    SpikeProp algorithm
    '''
    def spikeprop(self, t_d, lr=0.01):
        '''
        t_d - array of desired timings
        '''
        pass

    def calculate_delta_output(self, t_d, ids):
        t_a = self.find_latest_spike_timings_output()
        diff_timings = (t_d - t_a)
        


    def find_latest_spike_timings_output(self, ids):
        t_a = []
        for neu_id in self.preoutput_synapses.keys():
            t_a.append(self.preoutput_synapses[neu_id][0].get_latest_spike_timing_post())
        return np.array(t_a)
    
    def lmse(self, t_a, t_d):
        '''
        Least Mean Squares Error-function
        t_a - np.array of actual timings
        t_d - np.array of desired timings
        '''
        error = .5 * np.sum((t_a - t_d)**2)
        return error

    

    '''
    Misc:
    '''
    def reload_graph(self):
        '''
        re-initializes graph.
        Perhaps, it will be necessary to
        re-implement this function, given that
        this one literally creates a new graph
        '''
        self.graph = nx.DiGraph()
        for i in self.neurons:
            self.graph.add_node(self.neurons[i])
        for j in self.syn_by_edge:
            self.graph.add_edge(*self.get_neurons_by_edge(j), weight=self.syn_by_edge[j].get_weight())

    def extract_syn_layer_props(self, syn_layer):
        '''
        Extracts properties. Works for layers only
        Used in generate_model()
        '''
        props = []
        for i in syn_layer:
            props.append(syn_layer[i])
        return props

    def create_neuron(self, id, type, I, preset, color, resolution=.1):
        '''
        Creates a new neuron
        Used in generate_model()
        '''
        neuron = self.types_of_neurons[type](preset=preset, id=id, I=I, resolution=resolution)
        self.graph.add_node(neuron)
        self.color_map.append(color)
        return neuron
    
    def create_synapse(self, pre_id, post_id, syn_type):
        '''
        Creates a new synapse
        Used in generate_model()
        '''
        syn = self.types_of_synapses[syn_type](self.neurons[pre_id.get_id()], self.neurons[post_id.get_id()])
        self.graph.add_edge(self.neurons[pre_id.get_id()], self.neurons[post_id.get_id()], weight=syn.get_weight())
        self.syn_by_edge[(pre_id.get_id(), post_id.get_id())] = syn
        return syn

    def create_layer(self, id, cfg_neu, color):
        '''
        Creates a layer of neurons by config
        Used in generate_model()
        '''
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
        '''
        Adds indexes to neurons
        Used in generate_model()
        '''
        idx = 0
        for i in self.graph:
            self.neurons[idx] = i
            idx += 1

    def get_particular_layers(self, layers):
        '''
        Misc for generate_model()
        '''
        ls = layers.split(' ')
        return ls
    
    def connect_inside(self, layer_req, syn_type, num_con):
        '''
        Connects neurons inside the layer
        Used in generate_model()
        '''
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
        '''
        Connects layers of neurons
        Used in generate_model()
        '''
        pre = self.layers[layers_req[0]]
        post = self.layers[layers_req[1]]
        method(pre, post, syn_type, num_con)

    def fc(self, pre, post, syn_type, num_con):
        '''
        Algorithm for creating fully-connected layer
        of synapses
        Used in generate_model()
        '''
        stack = []
        syns = []
        for i in pre:
            for j in post:
                stack.append((i, j))
                syns.append(self.create_synapse(i, j, syn_type))
        return stack, syns

    def rc(self, pre, post, syn_type, num_con):
        '''
        Algorithm for creating random connections
        Used in generate_model()
        '''
        stack = []
        syns = []
        for i in range(num_con):
            pre_chosen = random.choice(pre)
            post_chosen = random.choice(post)
            stack.append((pre_chosen, post_chosen))
            syns.append(self.create_synapse(pre_chosen, post_chosen, syn_type))

    def empty_function(self):
        '''
        Helps to avoid if-else statements. Lmao
        '''
        pass

    '''
    For advanced visuals:
    '''    
    def spit_for_pyvis(self):
        '''
        Returns a graph that is digestable for pyvis
        '''
        new_graph = nx.DiGraph()
        for i in self.neurons:
            new_graph.add_node(self.neurons[i].get_id(), title=str(self.neurons[i].get_id()))
            print(self.neurons[i].get_id())
        for j in self.syn_by_edge:
            new_graph.add_edge(self.get_neurons_by_edge(j)[0].get_id(), self.get_neurons_by_edge(j)[1].get_id(), weight=str(self.syn_by_edge[j].get_weight()),
                               title=str(self.syn_by_edge[j].get_weight()))
        return new_graph
    
