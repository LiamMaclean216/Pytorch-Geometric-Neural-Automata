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
    training_set, 
    testing_set,
    training_params: dict, 
    edge_attr=None, 
    verbose=True, 
    edge_index=None, 
    wandb_log=True,
    wandb_loss='loss',
    wandb_acc='acc',
    save_dir=None,
    starting_epoch=0,
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
    
    
        
    for epoch in range(starting_epoch, training_params["n_epochs"]):
        update_rule.reset()
        x, loss, network_output, correct, network_in, metadata = update_rule.train()(
            training_params["n_steps"], training_set, 
            edge_attr=edge_attr, **forward_kwargs
        )
        # accuracy = (network_output.argmax(1) == correct.argmax(1)).sum().item() / training_params["batch_size"]
        correct =  correct.argmax(-1)
        network_output = network_output.argmax(-1)
        accuracy = (network_output == correct).mean().item()
        
        
        loss.backward()

        nn.utils.clip_grad_norm_(update_rule.parameters(), 1)
        optimizer.step()
        optimizer.zero_grad()
        
        if save_dir is not None and epoch % (100 // training_params["batch_size"]) == 0:
            torch.save(update_rule.state_dict(), f"{save_dir}.pt")
            
        if not (verbose and epoch % 50 == 0):
            continue
            
        log = {"training loss": loss, "training acc": accuracy}
        
        if testing_set:
            
            x = update_rule.initial_state().repeat(testing_set.batch_size, 1)
            _, test_loss, network_output_, correct_, _, _ = update_rule.eval()(
                training_params["n_steps"], testing_set,
                edge_attr=edge_attr, **forward_kwargs
            )
            network_output_ =  network_output_.argmax(-1)
            correct_ = correct_.argmax(-1)
            
            # test_accuracy = (network_output_.argmax(1) == correct_.argmax(1)).sum().item() / network_output.shape[1]
            test_accuracy = (network_output_.round() == correct_).mean().item()

            
            log["test loss"] = test_loss
            log["test acc"] = test_accuracy
        
        if wandb_log:
            wandb.log(
                log,
                step=epoch,
            )
        
        out_string = f"""\r 
            Epoch {epoch } |
            Loss {loss:.6} |
            Accuracy {int(accuracy * 100)}% |
            Network out: {network_output[0]} |
            Correct:  {correct[0]} |
            {metadata}
            """.replace("\n", " ").replace("            ", "")
        
        # Test Loss {test_loss:.6} |
        #     Test Accuracy {int(test_accuracy * 100)}% |
        if testing_set:
            out_string += f"""
            Test Loss {test_loss:.6} |
            Test Accuracy {int(test_accuracy * 100)}% |
            Test Network out: {network_output_[0]} |
            Test Correct:  {correct_[0]}
            """.replace("\n", " ").replace("            ", "")
        
        print(out_string, end="")
        if epoch % 100 == 0:
            print()
    
        

