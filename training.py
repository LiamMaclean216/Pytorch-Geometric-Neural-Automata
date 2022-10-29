from turtle import update
import numpy as np
import torch.nn as nn
import wandb

# numpy only displays 3 decimal places
np.set_printoptions(precision=3)


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
    for epoch in range(training_params["n_epochs"]):
        loss = 0
        accuracy = 0
        for _ in range(training_params["batch_size"]):
            for set_idx in meta_set.iterate():

                update_rule.reset()
                x = update_rule.initial_state()

                x, batch_loss, network_output, correct, network_in = update_rule(
                    x, training_params["n_steps"], meta_set.get_set(set_idx), edge_attr=edge_attr, edge_index=edge_index
                )
                loss += batch_loss

            accuracy += (np.round(network_output) == correct).all()

        loss /= training_params["batch_size"]
        accuracy /= training_params["batch_size"]
        if wandb_log:
            wandb.log({wandb_loss: loss, wandb_acc: accuracy})
        loss_integral += loss
        loss.backward()

        nn.utils.clip_grad_norm_(update_rule.parameters(), 1)
        optimizer.step()
        optimizer.zero_grad()
        if verbose:
            print(f"""\r 
                Epoch {epoch * training_params["batch_size"]} |
                Loss {loss:.6} |
                Accuracy {int(accuracy * 100)}% |
                Network out: {network_output} |
                Correct:  {correct} |
                Network In: {network_in}
                """.replace("\n", " ").replace("            ", ""), end="")
            if epoch % (100 // training_params["batch_size"]) == 0:
                print()

    return loss_integral
