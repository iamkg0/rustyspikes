from model import *
from neurons import *
from synaptics import *
from utils import *
from protocols import *
from pyvis.network import Network
res = .1
plt.style.use(['dark_background'])

snn = SNNModel()
snn.generate_model('config.txt')

nx.draw(snn.graph, node_color=snn.color_map, node_size=40)

net = Network(notebook = True, cdn_resources = "remote",
                bgcolor = "#222222",
                font_color = "white",
                width = "1920px",
                height = "1080px",
)

#net = Network(notebook=True, width='500px', height='500px')
G = snn.spit_for_pyvis()
print(G)
net.from_nx(G)
net.show('graph.html')