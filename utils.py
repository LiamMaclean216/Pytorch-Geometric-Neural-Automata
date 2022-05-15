import torch

# type_dict = {"hidden": 0, "input": 1, "output": 2}
type_dict = {"hidden": [1, 0, 0], "input": [0, 1, 0], "output": [0, 0, 1]}

def reset_inputs(x_data: torch.Tensor, input_data: torch.Tensor, hidden_dim: int = 1):
    """
    x_data: [n_nodes, n_features]
        features: [0:-2] is data
                  [-1] is the node type
    """
    # x_data[torch.argwhere(x_data[:,1] == type_dict["input"]).squeeze(-1), [0]] = input_data
    # print(x_data[torch.argwhere((x_data[:,1:] == torch.tensor(type_dict["input"])).all(1)).squeeze(-1), [0]].shape)
    x_data[torch.argwhere((x_data[:,hidden_dim:] == torch.tensor(type_dict["input"])).all(1)).squeeze(-1), [0]] = input_data
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

def remove_type(x, hidden_dim: int = 1):
    pass
    # """
    # Remove type from x
    # """
    # return x[:,0].unsqueeze(-1), x[:,1].unsqueeze(-1)