from turtle import update
import numpy as np
import torch.nn as nn
import torch
import wandb
from torch_geometric.loader import DataLoader
import copy
# numpy only displays 3 decimal places
np.set_printoptions(precision=3)
import networkx as nx
import torch_geometric.utils as utils
from torch_geometric.data import Data


def train_on_meta_set(
    update_rule, 
    optimizer, 
    meta_set, 
    training_params: dict, 
    edge_attr=None, 
    verbose=True, 
    edge_index=None, 
    wandb_log=True,
    wandb_loss='loss',
    wandb_acc='acc',
    save_dir=None,
    device=torch.device("cpu"),
    **forward_kwargs,
    ):
    """
    Train on meta dataset
    Args:
        update_rule: update rule to use
        optimizer: optimizer to use
        meta_set: meta dataset to use
        params: parameters for the update rule
            params["n_steps"]: int
            params["batch_size"]: int
            params["n_epochs"]: int
    """
    training_params.setdefault("n_epochs", 10000)
    training_params.setdefault("batch_size", 1)
    loss_integral = 0
    
    loader = DataLoader([update_rule.graph]*training_params["batch_size"], batch_size = training_params["batch_size"])
    graph = loader.__iter__().__next__()
    graph.batch = graph.batch.to(device)
    edge_index = utils.sort_edge_index(graph.edge_index).to(device)
    tmp = edge_index[0].clone()
    edge_index[0] = edge_index[1]
    edge_index[1] = tmp
        
    for epoch in range(training_params["n_epochs"]):
        loss = 0
        # accuracy = 0
        # for _ in range(15):#range(training_params["batch_size"]):
        # for set_idx in meta_set.iterate():
        update_rule.reset()
        x = update_rule.initial_state().repeat(training_params["batch_size"], 1)
        # loader = DataLoader([copy.deepcopy(update_rule.graph), copy.deepcopy(update_rule.graph),copy.deepcopy(update_rule.graph)] , batch_size = training_params["batch_size"])
        # loader = DataLoader([copy.deepcopy(update_rule.graph), copy.deepcopy(update_rule.graph)] , batch_size = training_params["batch_size"])
        
        
        
        # graph.edge_index = edge_index
        # nx.draw(utils.to_networkx(graph, to_undirected=False, remove_self_loops = True))
        
        x, batch_loss, network_output, correct, network_in = update_rule(
            x, training_params["n_steps"], meta_set, 
            edge_attr=edge_attr, edge_index=edge_index, batch=graph.batch, **forward_kwargs
        )
        loss += batch_loss

        # print(network_output.round(), correct)
        
        accuracy = (network_output.argmax(1) == correct.argmax(1)).sum().item() / training_params["batch_size"]

        loss /= training_params["batch_size"]
        # accuracy /= training_params["batch_size"]

        if wandb_log:
            wandb.log(
                {wandb_loss: loss, wandb_acc: accuracy}, 
                step=epoch, 
                commit = (epoch % 25 == 0 or epoch == training_params["n_epochs"] - 1)
            )
        loss_integral += loss
        loss.backward()

        nn.utils.clip_grad_norm_(update_rule.parameters(), 1)
        optimizer.step()
        optimizer.zero_grad()
        if verbose and epoch % 20 == 0:
            print(f"""\r 
                Epoch {epoch } |
                Loss {loss:.6} |
                Accuracy {int(accuracy * 100)}% |
                Network out: {network_output[0]} |
                Correct:  {correct[0]} |
                Network In: {network_in}
                """.replace("\n", " ").replace("            ", ""), end="")
            if epoch % 100 == 0:
                print()
        
        if save_dir is not None and epoch % (100 // training_params["batch_size"]) == 0:
            torch.save(update_rule.state_dict(), f"{save_dir}.pt")


    return loss_integral
