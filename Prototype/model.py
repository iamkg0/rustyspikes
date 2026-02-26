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
        self.syn_by_num = {}
        self.graph = nx.DiGraph()
        self.color_map = []
        self.color_set = kwargs.get("color_set", ("red", "green", "blue", "yellow", "cyan", "pink", "brown"))

        self.output_neurons = {}
        self.preoutput_synapses = {}

        self.event_listeners = [] # ids of event listeners
        self.inputs = []

        self.distance_matrix = None

        self.note = ''

        '''
        Get info:
        '''
    def __getitem__(self, idx):
        return self.neurons[idx]
    
    def num_synapses(self):
        return len(self.syn_by_edge)
    
    def num_neurons(self):
        return len(self.neurons)
    
    def show_config(self):
        conf = {}
        conf['Neurons'] = self.neurons
        conf['Synapses'] = self.syn_by_edge
        return conf
    
    def get_first_synapse(self):
        '''
        No idea
        '''
        return self.syn_by_edge[next(iter(self.syn_by_edge))]
    
    def enum_synapses(self):
        for i, (k,v) in enumerate(self.syn_by_edge.items()):
            self.syn_by_num[i] = k
    
    def get_syn(self, idx):
        '''
        Returns synapse by its number
        '''
        return self.syn_by_edge[self.syn_by_num[idx]]


    def get_neuron_stats(self, id):
        '''
        Returns tuple of voltage, trace and input current of neuron by id
        '''
        v = self.neurons[id].get_voltage_dynamics()
        impulse = self.neurons[id].get_output_current()
        I = self.neurons[id].get_input_current()
        return v, impulse, I
    
    def get_weight(self, id):
        '''
        returns weight of given synapse, float
        '''
        return self.syn_by_edge[id].get_weight()
    
    def get_weights(self, ids=None):
        '''
        returns weights of given list of synapses, list
        in case ids is None of False, gathers values from all synapses
        '''
        weights = []
        if ids:
            for i in ids:
                weights.append(self.syn_by_edge[i].get_weight())
        else:
            for i in self.syn_by_edge.values():
                weights.append(i.get_weight())
        return weights
    
    def get_delay(self, id):
        '''
        returns delay of given synapse, float
        '''
        return self.syn_by_edge[id].get_delay()
        
    def get_delays(self, ids=None):
        '''
        returns delays of given synapses, list
        in case ids is None of False, gathers values from all synapses
        '''
        delays = []
        if ids:
            for i in ids:
                delays.append(self.syn_by_edge[i].get_delay())
        else:
            for i in self.syn_by_edge.values():
                delays.append(i.get_delay())
        return delays

    def get_graph(self):
        return self.graph
    
    def get_neurons_by_edge(self, edge):
        '''
        Returns tuple of neurons (pre, post) by edge of synapse
        Useless, i dont really remember what i needed it for
        '''
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
        '''
        Returns dict of output neurons
        '''
        return self.output_neurons
    
    def get_output_ids(self):
        '''
        Returns list of ids of output neurons
        '''
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
        '''
        Returns list of synapses obj, in which neuron of certain id is
        presynaptic
        '''
        synapses = []
        postsyn_neurons_ids = self.get_postsyn_neurons_ids(id)
        for i in postsyn_neurons_ids:
            synapses.append(self.syn_by_edge[(id, i)])
        return synapses
    
    def get_response(self, just_timings=True):
        '''
        Returns list of spikes
        if just_timings == True, list contains only 1 and 0
        if just_timings == False, contains float values of traces
        '''
        response = []
        if just_timings:
            for i in self.output_neurons.keys():
                response.append(int(self.output_neurons[i].spiked))
        else:
            for i in self.output_neurons.keys():
                response.append(self.output_neurons[i].impulse)
        return response
    
    def get_deadends(self):
        '''
        Returns list of neurons that lack postsynaptic connections
        '''
        deadends = []
        for i in self.neurons.keys():
            if not self.get_outgoing_synapses(i):
                deadends.append(i)
        return deadends
    
    def get_sources(self):
        '''
        Returns list of neurons that lack presynaptic connections
        '''
        sources = []
        for i in self.neurons.keys():
            if not self.get_incoming_synapses(i):
                sources.append(i)
        return sources
    
    def get_outsiders(self):
        '''
        Returns list of neurons that are not connected to other neurons
        '''
        outsiders = []
        for i in self.neurons.keys():
            if not self.get_incoming_synapses(i) and not self.get_outgoing_synapses(i):
                outsiders.append(i)
        return outsiders


   
    '''
    Handling the model:
    '''
    def generate_model(self, config='config.txt'):
        '''
        Legacy. Rotten long time ago
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
        '''
        Iteration
        '''
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

    def remove_neurons(self, id):
        '''
        Kicks neurons and appropriate synapses from the server
        '''
        if type(id) == int:
            id = [id]
        for i in id:
            pre_ids = self.get_presyn_neurons_ids(i)
            for j in pre_ids:
                del self.syn_by_edge[j, i]
            post_ids = self.get_postsyn_neurons_ids(i)
            for j in post_ids:
                del self.syn_by_edge[i, j]
            del self.neurons[i]



    def set_rule_to_all(self, rule=None):
        '''
        Switch local plasticity rule of each synapse
        '''
        for i in self.syn_by_edge:
            self.syn_by_edge[i].change_learning_rule(rule)

    def set_lr_to_all(self, lr=.1):
        '''
        Change learning rate of each synapse
        '''
        for i in self.syn_by_edge:
            self.syn_by_edge[i].change_lr(lr)

    def set_noise(self, noise=0):
        '''
        Change noise ranges of each neuron
        '''
        for i in self.neurons.keys():
            self.neurons[i].noise = noise

    def set_random_weights(self, ranges=(0,1)):
        '''
        Randomize weights in the whole model
        '''
        for i in self.syn_by_edge.keys():
            self.syn_by_edge[i].w = np.random.uniform(*ranges)

    def set_weight_manually(self, id, weight):
        '''
        Set weight of desired synapse
        '''
        self.syn_by_edge[id].set_weight_manually(weight)

    def set_d_lr(self, d_lr, ids=None):
        '''
        Set delay of desired synapses
        May cause malfunctions if type is not DelayedSynapse
        '''
        if not ids:
            ids = [i for i in self.syn_by_edge.keys()]
        for syn in ids:
            self.syn_by_edge[syn].d_lr = d_lr

    def set_slow_var_limit(self, limit, ids=None):
        '''
        Manually change limit of slow variable
        Appropriate for tSTDP learning rule
        '''
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
        '''
        Set neurons to be rendered as output
        Purely QoL feature
        '''
        if isinstance(ids, int):
            ids = [ids]
        for id in ids:
            self.output_neurons[id] = self.neurons[id]
        for i in self.output_neurons.keys():
            self.preoutput_synapses[i] = self.get_incoming_synapses(i)

    def drop_impulses(self, ids=None):
        '''
        set neu impulses to 0
        NOT TESTED!!!
        '''
        if ids:
            for i in ids:
                self.neurons[i].impulse = 0
        else:
            for i in self.neurons.keys():
                self.neurons[i].impulse = 0
    

    '''
    Misc:
    '''
    def change_note(self, note=''):
        '''
        Adds note, str
        Previous one will be erased
        '''
        self.note = note

    def add_note(self, note=''):
        '''
        Adds note after existing one, str
        '''
        self.note += note

    def show_note(self):
        return self.note

    def reload_graph(self):
        '''
        re-initializes graph.
        Perhaps, it will be necessary to
        re-implement this function, given that
        this one literally creates a new graph
        '''
        self.graph = nx.DiGraph()
        for i in self.neurons:
            self.graph.add_node(self.neurons[i], pos=self.neurons[i].coords)
        for j in self.syn_by_edge:
            self.graph.add_edge(*self.get_neurons_by_edge(j), weight=self.syn_by_edge[j].get_weight())
        self.enum_synapses()

    def extract_syn_layer_props(self, syn_layer):
        '''
        Legacy. Rotten long time ago
        Extracts properties. Works for layers only
        Used in generate_model()
        '''
        props = []
        for i in syn_layer:
            props.append(syn_layer[i])
        return props

    def create_neuron(self, id, type, I, preset, color, resolution=.1):
        '''
        Legacy. Rotten long time ago
        Creates a new neuron
        Used in generate_model()
        '''
        neuron = self.types_of_neurons[type](preset=preset, id=id, I=I, resolution=resolution)
        self.graph.add_node(neuron)
        self.color_map.append(color)
        return neuron
    
    def create_synapse(self, pre_id, post_id, syn_type):
        '''
        Legacy. Rotten long time ago
        Creates a new synapse
        Used in generate_model()
        '''
        syn = self.types_of_synapses[syn_type](self.neurons[pre_id.get_id()], self.neurons[post_id.get_id()])
        self.graph.add_edge(self.neurons[pre_id.get_id()], self.neurons[post_id.get_id()], weight=syn.get_weight())
        self.syn_by_edge[(pre_id.get_id(), post_id.get_id())] = syn
        return syn

    def create_layer(self, id, cfg_neu, color):
        '''
        Legacy. Rotten long time ago
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
        Legacy. Rotten long time ago
        Adds indexes to neurons
        Used in generate_model()
        '''
        idx = 0
        for i in self.graph:
            self.neurons[idx] = i
            idx += 1

    def get_particular_layers(self, layers):
        '''
        Legacy. Rotten long time ago
        Misc for generate_model()
        '''
        ls = layers.split(' ')
        return ls
    
    def connect_inside(self, layer_req, syn_type, num_con):
        '''
        Legacy. Rotten long time ago
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
        Legacy. Rotten long time ago
        Connects layers of neurons
        Used in generate_model()
        '''
        pre = self.layers[layers_req[0]]
        post = self.layers[layers_req[1]]
        method(pre, post, syn_type, num_con)

    def fc(self, pre, post, syn_type, num_con):
        '''
        Legacy. Rotten long time ago
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
        Legacy. Rotten long time ago
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
    
    def find_distance(self, neu1, neu2):
        coords1 = self.neurons[neu1].get_coords()
        coords2 = self.neurons[neu2].get_coords()
        dist = (coords1[0]-coords1[1])**2 + (coords2[0]-coords2[1])**2
        return dist
    

    def create_distance_matrix(self):
        dists = np.zeros((len(self.neurons), len(self.neurons)))
        for i in self.neurons:
            for j in self.neurons:
                if i != j:
                    dists[i,j] = self.find_distance(i, j)
        self.distance_matrix = dists
        return dists




    '''
    Generate network
    '''
    def generate_network_local_connections(self, num_exc, num_inh, num_syn, coords_dim=1000, max_delay=500):
        num_neu = num_exc+num_inh
        for i in range(num_neu):
            neu = Izhikevich(xy = np.random.randint(0, coords_dim, size=2).tolist())
            self.add_neuron(neu)
        dists = self.create_distance_matrix()
        probs = np.ones_like(dists) - dists / np.sum(dists)
        for presyn_idx in range(num_neu):
            if presyn_idx < num_exc:
                syn_type = False
            else:
                syn_type = True
            to_exclude = [presyn_idx]
            if self.get_presyn_neurons_ids(presyn_idx):
                for i in range(len(self.get_presyn_neurons_ids(presyn_idx))):
                    to_exclude.append(self.get_presyn_neurons_ids(presyn_idx)[i])
            prob_connections = probs[presyn_idx]
            prob_connections[to_exclude]*=0
            prob_connections = prob_connections / np.sum(prob_connections)
            to_connect = np.random.choice(np.arange(len(prob_connections)), size=num_syn, p=prob_connections)
            for post_syn in to_connect:
                synapse = Delayed_synapse(self.neurons[presyn_idx], self.neurons[post_syn], inhibitory=syn_type, max_delay=max_delay)
                self.add_synapse(synapse)
        self.reload_graph()
        

            
        



        

    '''
    event-related
    '''
    def define_listeners(self):
        for i in self.neurons.values():
            if i.event_listener:
                self.event_listeners.append(i.id)

    def listen(self, event):
        e = []
        for j in range(len(event[0])):
            e.append(event[0][j] * event[1][j])
        for i in self.event_listeners:
            self.neurons[i].listen(e)


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
    
