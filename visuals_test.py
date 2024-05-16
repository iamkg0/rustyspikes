from model import *
from neurons import *
from synaptics import *
from utils import *
from protocols import *
from shallow_models import *
from pyvis.network import Network
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def pyvis_test(model):
    net = Network(notebook = True, cdn_resources = "remote",
                bgcolor = "#222222",
                font_color = "white",
                width = "1920px",
                height = "1080px",
                directed=True)
    G = model.spit_for_pyvis()
    print(G)
    net.from_nx(G, show_edge_weights=True)
    label = 0
    for i in net.nodes:
        i['title'] = str(i['label'])
        label += 1
    for j in net.edges:
        print(j)
        j["title"] = j["width"]
    net.show('graph.html')


model = net_v0()
pos = nx.spring_layout(model)
nx.draw_networkx_nodes(model, node_size=20)
nx.draw_networkx_edges(model)
nx.draw_networkx_labels(model)
edge_labels = nx.get_edge_attributes(model, "weight")
nx.draw_networkx_edge_labels(model, edge_labels)

ax = plt.gca()
ax.margins(.08)
plt.show()