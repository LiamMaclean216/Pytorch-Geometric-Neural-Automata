from functools import lru_cache
import torch
import networkx as nx
from torch_geometric.utils import grid, remove_self_loops, add_self_loops
import random
import numpy as np
import math
# type_dict = {"hidden": 0, "input": 1, "output": 2}
type_dict = {"hidden": [1, 0, 0], "input": [0, 1, 0], "output": [0, 0, 1]}

def reset_inputs(x_data: torch.Tensor, input_data: torch.Tensor, output_data: torch.Tensor = None, hidden_dim: int = 1):
    """
    x_data: [n_nodes, n_features]
        features: [0:-2] is data
                  [-1] is the node type
    """
    # x_data[torch.argwhere(x_data[:,1] == type_dict["input"]).squeeze(-1), [0]] = input_data
    # print(x_data[torch.argwhere((x_data[:,1:] == torch.tensor(type_dict["input"])).all(1)).squeeze(-1), [0]].shape)
    x_data[torch.argwhere((x_data[:,hidden_dim:] == torch.tensor(type_dict["input"])).all(1)).squeeze(-1), [0]] = input_data
    
    if output_data is not None:
        x_data[torch.argwhere((x_data[:,hidden_dim:] == torch.tensor(type_dict["output"])).all(1)).squeeze(-1), [0]] = output_data

        
    return x_data

def get_output(data, hidden_dim: int = 1):
    """
    Returns network output
    """
    # return data[torch.argwhere(data[:,1] == type_dict["output"]).squeeze(-1), [0]]
    return data[torch.argwhere((data[:,hidden_dim:] == torch.tensor(type_dict["output"])).all(1)).squeeze(-1), [0]]


def get_type(x, hidden_dim: int = 1):
    """
    Get types of all nodes in x
    """
    return x[:,hidden_dim:]#.unsqueeze(-1)

    
    
def add_reverse_edges(edges):
    """
    [
        [0]
        [1]
    ]
    becomes
    [
        [0,1]
        [1,0]
    ]
    """
    edges = edges.transpose(0,1)
    return torch.stack((torch.concat((edges[0], edges[1]), 0), torch.concat((edges[1], edges[0]), 0)), 0).transpose(0,1)


def build_edges(n_inputs: int, n_outputs: int, height: int, width: int, mode="dense", input_mode="dense", n_switches=0):
    """
    Builds edges like 2d_grid_graph
    """
    #hidden neurons
    if mode == "grid":
        edge_list = list(nx.grid_2d_graph(height, width).edges())
        node_list = list(nx.grid_2d_graph(height, width).nodes())

        #replace each element of edge_list with its index in node_list
        for i in range(len(edge_list)):
            edge_list[i] = (node_list.index(edge_list[i][0]), node_list.index(edge_list[i][1]))
            
        edges = torch.tensor(edge_list)
        edges = add_reverse_edges(edges)
    elif mode == "dense":
        edges = grid(height, width)[0].transpose(0,1)
    else:
        raise ValueError("mode must be either 'grid' or 'dense'")
    
    # edges = torch.concat((torch.tensor([[0,3], [3,0], [4,7], [7,4]]), edges))

    if input_mode == "dense":
        #input neurons
        input_edges = torch.tensor([
            [
                [x, (height*width) + y] for x in range(width)
            ] for y in range(n_inputs)
        ]).view(-1, 2)
        #output neurons
        output_edges = torch.tensor([
            [
                [(height*width)-(x+1), (height*width) + y+n_inputs] for x in range(width)
            ] for y in range(n_outputs)
        ]).view(-1, 2)
    elif input_mode == "grid":
        input_edges = torch.tensor([[[x, (height*width) + x] for x in range(width)]]).view(-1, 2)
        output_edges = torch.tensor([[[(height*width)-(x+1), (height*width) + x+n_inputs] for x in range(width)]]).view(-1, 2)
    else:
        raise ValueError("input_mode must be either 'grid' or 'dense'")

    #replace ten random elements of edges with [random, random]
    for i in range(n_switches):
        edges[random.randint(0, edges.shape[0]-1)] = torch.tensor([
            random.randint(0, width*height), random.randint(0, width*height)
        ])

    input_edges = add_reverse_edges(input_edges)
    output_edges = add_reverse_edges(output_edges)
    
    edges = torch.cat((edges, input_edges, output_edges), dim=0).transpose(0,1)
    
    edges = remove_self_loops(edges)[0]
    edges = add_self_loops(edges)[0]
    return edges

@lru_cache(maxsize=30)
def build_edges_3d(input_shape: tuple, height: int):
    G = nx.grid_graph(dim=(input_shape[0], input_shape[1], height))
    
    edge_list = list(G.edges())
    node_list = list(G.nodes())

    #replace each element of edge_list with its index in node_list
    for i in range(len(edge_list)):
        edge_list[i] = (node_list.index(edge_list[i][0]), node_list.index(edge_list[i][1]))
        
    edges = torch.tensor(edge_list)
    edges = add_reverse_edges(edges)
    
    input_edges = torch.tensor([[x, x+len(node_list)] for x in range(0,input_shape[0] * input_shape[1])])
    output_edges = torch.tensor(
        [[len(node_list) - x, x+len(node_list)+input_edges.shape[0] - 1] for x in range(1,(input_shape[0] * input_shape[1]) + 1)]
        )
    
    
    input_edges = add_reverse_edges(input_edges)
    output_edges = add_reverse_edges(output_edges)

    edges = torch.cat((edges, input_edges,output_edges)).transpose(0,1)
    
    edges = remove_self_loops(edges)[0]
    edges = add_self_loops(edges)[0]
    return edges


def run_rule(data_x, update_rule, n_steps = 5):
    x = update_rule.initial
    update_rule.reset()
    # edge_index = data.edge_index.long().clone()
    for i in range(n_steps):
        x = update_rule(x, data_x.float())
    
        network_output = update_rule.get_output(x).detach()
        print(network_output)
    return network_output
    
    
def seed(seed):
    random.seed(seed)                          
    np.random.seed(seed)       
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True  
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec#d554
class PositionalEncoder(torch.nn.Module):
    def __init__(self, d_model, max_seq_len = 80):
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + self.pe[:,:seq_len]
        return x
    
def get_shape_input_indicies(shape: tuple, height: int):
    n_non_io_nodes = shape[0] * shape[1] * height
    
    input_start_index = n_non_io_nodes
    output_start_index = n_non_io_nodes + shape[0] * shape[1]
    n_nodes = n_non_io_nodes + shape[0] * shape[1] * 2
    
    return input_start_index, output_start_index, n_nodes

def get_shape_output_indices(shape: tuple, height: int):
    n_non_io_nodes = shape[0] * shape[1] * height
    
    output_start_index = n_non_io_nodes + shape[0] * shape[1]
    n_nodes = n_non_io_nodes + shape[0] * shape[1] * 2
    
    return output_start_index, n_nodes, n_nodes