import torch
import networkx as nx

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

    
def build_edges(n_inputs: int, n_outputs: int, height: int, width: int):
    """
    Builds edges like 2d_grid_graph
    """
    #hidden neurons
    edge_list = list(nx.grid_2d_graph(height, width).edges())
    node_list = list(nx.grid_2d_graph(height, width).nodes())

    #replace each element of edge_list with its index in node_list
    for i in range(len(edge_list)):
        edge_list[i] = (node_list.index(edge_list[i][0]), node_list.index(edge_list[i][1]))
        
    edges = torch.tensor(edge_list)
    
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

    #merge edges and input_edges
    edges = torch.cat((edges, input_edges, output_edges), dim=0).transpose(0,1)
    edges =  torch.stack((torch.concat((edges[0], edges[1]), 0), torch.concat((edges[1], edges[0]), 0)), 0) #?
    
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
    