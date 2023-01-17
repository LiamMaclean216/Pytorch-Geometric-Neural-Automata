from locale import normalize
from turtle import forward
from torch_geometric.nn import GCNConv, Sequential, SAGEConv, GATConv
# from layers import GATConv
from torch_geometric.nn.norm import LayerNorm, PairNorm, MeanSubtractionNorm
# from torch_geometric.nn.aggr import LSTMAggregation, Aggregation
from typing import Optional
from torch import Tensor
import torch

from torch_geometric.nn.aggr import LSTMAggregation, MaxAggregation, AttentionalAggregation
from torch.nn import LSTM, MultiheadAttention
from torch.nn.functional import one_hot

from torch_geometric.nn.aggr import Aggregation

# cuda_device = torch.device("cpu")
cuda_device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

class SelfAttnAggregation(Aggregation):
    def __init__(self, in_channels, n_heads = 1):
        super().__init__()
        self.node_degree = 9
        self.in_channels = in_channels + self.node_degree
        self.attention1 = MultiheadAttention(self.in_channels, n_heads, batch_first=True, dropout=0)
        self.attention2 = MultiheadAttention(self.in_channels, n_heads, batch_first=True, dropout=0)
        # self.attention3 = MultiheadAttention(self.in_channels, n_heads, batch_first=True, dropout=0.15)
        # self.attention4 = MultiheadAttention(self.in_channels, n_heads, batch_first=True, dropout=0.15)
        self.n_heads = n_heads

        # self.attention_fogor = MultiheadAttention(self.in_channels, heads, batch_first=True)
        # self.attention_update = MultiheadAttention(self.in_channels, heads, batch_first=True)


    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        x, _ = self.to_dense_batch(x, index, ptr, dim_size, dim)

        #adding directionality increases stability with more nodes
        x = torch.concat((
            x,
            one_hot(torch.arange(x.shape[1]), x.shape[1]).unsqueeze(0).repeat(x.shape[0],1,1).to(cuda_device)
        ), -1)

        x = torch.nn.functional.pad(x, (0, self.in_channels - x.shape[2]), "constant", 0)
        
        # fogor = self.attention_fogor(x, x, x)[0]
        # update = self.attention_update(x, x, x)[0]

        #self attention on x
        #repeat x for each head
        # print(x_repeated.shape)
        # print(self.attention1(x_repeated,x_repeated,x_repeated)[0])
        x = x + (
            self.attention1(x,x,x)[0] + 
            self.attention2(x, x, x)[0])# +
            # self.attention3(x, x, x)[0] +
            # self.attention4(x, x, x)[0]
        # ) 

        

        #remove the directionality
        x = x[:, :, :self.in_channels - self.node_degree]
        # fogor = fogor[:, :, :self.in_channels - self.node_degree]
        return torch.max(x, dim=1)[0]# * torch.sum(fogor, dim=1).sigmoid()
        # sum sucks    
        # return torch.sum(x, dim=1)
        # return self.reduce(x, index, ptr, dim_size, dim, reduce='max')

from torch.nn import ReLU, LeakyReLU
import torch
import torch.nn as nn
from utils import *
from torch_geometric.data import Data
import torch_geometric.utils as utils
import torch.nn.functional as F

import numpy as np


class UpdateRule(torch.nn.Module):
    def __init__(self, 
                n_inputs, 
                n_outputs,
                hidden_dim,
                edge_dim,
                network_width = 80,
                heads = 1,
                cuda_device = torch.device("cpu")
                ):
        
        super(UpdateRule, self).__init__()
        self.cuda_device = cuda_device
        self.edge_dim = edge_dim
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.hidden_dim = hidden_dim
        skip_size = hidden_dim
        self.total_hidden_dim = hidden_dim# + skip_size
        self.network_width = network_width
        
        fill_value = 'mean'
        if edge_dim is not None:
            fill_value = nn.Parameter(torch.zeros([edge_dim])).to(self.cuda_device)
        
       
        # self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.relu = nn.ReLU()
        
        # self.input_vectorizer = nn.Linear(n_inputs, self.total_hidden_dim, bias=True)
        self.input_vector_size = 2
        self.input_vectorizer = nn.Linear(1, self.input_vector_size)
        
        # Vectorizes training targets
        self.reverse_output_vectorizer = nn.Linear(2, self.input_vector_size)
        self.output_vectorizer = nn.Linear(3, 1)
        
        # self.layer_norm1 = LayerNorm(network_width*heads)
        # self.layer_norm2 = LayerNorm(network_width*heads)
        # self.layer_norm3 = LayerNorm(network_width*heads)
        # self.layer_norm4 = LayerNorm(network_width*heads)
        # self.layer_norm5 = LayerNorm(network_width*heads)

        self.layer_norm1 = PairNorm()#network_width*heads)
        self.layer_norm2 = PairNorm()#(network_width*heads)
        self.layer_norm3 = PairNorm()#(network_width*heads)
        self.layer_norm4 = PairNorm()#(network_width*heads)
        self.layer_norm5 = PairNorm()#(network_width*heads)

        # aggr = LSTMAggregation(network_width,network_width)#'lstm'
        # self.conv1 = SAGEConv(
        #     self.total_hidden_dim+2, network_width, aggr=LSTMAggregation(self.total_hidden_dim+2,self.total_hidden_dim+2), normalize=True
        #     ,root_weight=True, project=False
        # )
        # self.conv2 = SAGEConv(network_width, network_width, aggr=LSTMAggregation(network_width,network_width), normalize=True)
        # self.conv3 = SAGEConv(network_width, network_width, aggr=LSTMAggregation(network_width,network_width), normalize=True)
        # self.conv4 = SAGEConv(network_width, network_width, aggr=LSTMAggregation(network_width,network_width), normalize=True)
        # self.conv_out = SAGEConv(
        #     network_width, hidden_dim, aggr=LSTMAggregation(network_width,network_width), normalize=True, root_weight=True, project=False
        #     )


        # aggr = SelfAttnAggregation(network_width)#'max'
        
        kwargs = {'add_self_loops': True, 'normalize':False}
        self.conv1 = GCNConv(self.total_hidden_dim+2, network_width, aggr= SelfAttnAggregation(network_width, heads), **kwargs)
        self.conv2 = GCNConv(network_width, network_width, aggr=SelfAttnAggregation(network_width, heads), **kwargs)
        self.conv3 = GCNConv(network_width, network_width, aggr=SelfAttnAggregation(network_width, heads), **kwargs)
        self.conv4 = GCNConv(network_width, network_width, aggr=SelfAttnAggregation(network_width, heads), **kwargs)
        self.conv_out = GCNConv(network_width, hidden_dim, aggr=SelfAttnAggregation(hidden_dim, heads), **kwargs)

        self.forgor1 = nn.Linear(network_width+2, network_width)
        self.forgor2 = nn.Linear(network_width, hidden_dim+2)
        
        self.update1 = nn.Linear(network_width+2, network_width)
        self.update2 = nn.Linear(network_width, hidden_dim)
        
        self.reset()


    def initial_state(self, height = None, width = None, build = False):
        if height is None:
            height = self.height
            
        if width is None:
            width = self.width
        
        #initial state is all zeros
        if build:
            self.initial = torch.zeros([height*width + self.n_outputs + self.n_inputs, self.total_hidden_dim]).to(self.cuda_device)


        #trained initial on only input and output nodes
        # if build:
        #     self.initial = nn.parameter.Parameter(
        #         torch.zeros([self.n_outputs + self.n_inputs, self.total_hidden_dim]), requires_grad=True
        #     ).to(self.cuda_device)
        
        # return torch.concat(
        #     (torch.zeros([height*width, self.total_hidden_dim]).to(self.cuda_device),
        #     self.initial)
        # )
        
        # Trained initial on all nodes including hidden
        # if build:
        #     self.initial = nn.parameter.Parameter(
        #         torch.zeros([height*width + self.n_outputs + self.n_inputs, self.total_hidden_dim])#, requires_grad=True
        #     ).to(self.cuda_device)
        
        return self.initial
        
    
    def build_graph(self, height, width, mode="dense", input_mode="dense"):
        self.initial_state(height, width, build=True)
        self.width = width
        self.height = height
        self.n_non_io_nodes = width * height
        self.n_nodes = height*width + self.n_inputs + self.n_outputs
        
        edges = build_edges(self.n_inputs, self.n_outputs, height, width, mode=mode, input_mode=input_mode)
        
        
        self.graph = Data(edge_index=edges, x=torch.zeros(self.n_nodes, self.total_hidden_dim))
        self.edge_index = self.graph.edge_index.long().clone().to(self.cuda_device)
        
        self.edge_attr = None
        if self.edge_dim is not None:
            self.edge_attr = nn.parameter.Parameter(
                torch.zeros([self.graph.edge_index[0].shape[0], self.edge_dim])
            ).to(self.cuda_device)

        # self.edge_weight = nn.parameter.Parameter(
        #     torch.ones([self.graph.edge_index[0].shape[0], 1]) / 100
        # ).to(self.cuda_device)

        
    def get_edge_weight(self):
        # return (self.edge_weight * 100).sigmoid()
        return None

    def draw(self):
        graph = utils.to_networkx(self.graph, to_undirected=False, remove_self_loops = True)
        nx.draw(graph)
    
    
    def vectorise_input(self, x, input_data):
        
        vectorized_input = self.input_vectorizer(input_data)
        
        mask = torch.zeros_like(x).to(self.cuda_device)
        for i in range(mask.shape[0]//self.n_nodes):
            mask[
                    self.n_non_io_nodes+(i*self.n_nodes):self.n_non_io_nodes+self.n_outputs+(i*self.n_nodes), :self.input_vector_size
                ] = vectorized_input[i]
        
        # print(x.shape)
        # print(x)
        
        x = x + mask
        # print(x)
        return x
            
    def vectorize_output(self, x, input_data):
        vectorized_output = self.reverse_output_vectorizer(input_data)
        
        mask = torch.zeros_like(x).to(self.cuda_device)
        for i in range(mask.shape[0]//self.n_nodes):
            mask[
                    (self.n_non_io_nodes) + self.n_outputs + (i*self.n_nodes): self.n_nodes + (i*self.n_nodes), :self.input_vector_size
                ] = vectorized_output[i]
            
        x = x + mask
        return x
    
    def forward(
        self, x, n_steps, data, return_all = True, edge_attr = None, edge_index = None, last_idx = 1, batch=None
    ):
        network_in = []
        network_out = []
        
        for idx, (problem_data_x, problem_data_y) in enumerate(data):
            last = idx == last_idx#len(data) - 1
            # print(problem_data_x, problem_data_y)
            # problem_data_y = torch.concat((problem_data_y,problem_data_y), 0)
            problem_data_y_ = problem_data_y.float().unsqueeze(-1) 
            problem_data_y_ = torch.cat((problem_data_y_, torch.ones_like(problem_data_y_)), dim = 2)
            input_data = problem_data_x.float().unsqueeze(-1)
            # print(input_data, problem_data_y_)
            
            # input_data = torch.concat((input_data,input_data), 0)
            
            if not last:
                x = self.vectorize_output(x, problem_data_y_)
            else:
                x = self.vectorize_output(x, torch.zeros_like(problem_data_y_).to(self.cuda_device))
            
            x = self.vectorise_input(x, input_data)
            for _ in range(n_steps):
                
                #this makes a big difference when increasing last_idx
                if not last:
                    x = torch.cat((torch.zeros([x.shape[0], 1]).to(self.cuda_device), x), dim = 1)
                    x = torch.cat((torch.ones([x.shape[0], 1]).to(self.cuda_device), x), dim = 1)
                else:
                    x = torch.cat((torch.ones([x.shape[0], 1]).to(self.cuda_device), x), dim = 1)
                    x = torch.cat((torch.zeros([x.shape[0], 1]).to(self.cuda_device), x), dim = 1)
                x = self.step(x, edge_attr=edge_attr, edge_index=edge_index, batch=batch)
            
            if last:
                break
            # break

        network_output = self.get_output(x)
        # loss = F.binary_cross_entropy_with_logits(network_output, problem_data_y.float())
        
        #l2 loss
        loss = F.mse_loss(network_output, problem_data_y.float())
        
        
        if return_all:
            return (
                x, 
                loss, 
                network_output.cpu().detach().numpy(),
                problem_data_y.cpu().float().numpy(), 
                np.array(network_out)
            )
        
        return x
        
    
    def act(self, x, env_state, n_step = 1):
        input_data = env_state.float().unsqueeze(-1).to(self.cuda_device)
        x = self.vectorise_input(x, input_data)
        for _ in range(n_step):
            x = self.step(x, env_state.float(), None)
        
        return x
        
    
    def step(self, x, edge_attr = None, edge_index = None, batch=None):
        # if edge_index is None:
        #     edge_index = self.edge_index
        
        # for i in x:
        #     print(i)
        # print("#######")
        # forgor = self.forgor1(x).sigmoid()
        # forgor = self.forgor2(forgor).sigmoid()
        # x = (x * forgor)# + (
            # (1 - forgor) * torch.cat(
            #     (self.initial_state().repeat(x.shape[0] // self.initial_state().shape[0], 1), torch.zeros([x.shape[0], 2]).to(self.cuda_device))
            #     , dim = -1))

        # update = self.update1(x).sigmoid()
        # update = self.update2(update).sigmoid() 
        

        edge_weight = None#self.get_edge_weight()#None
        updatet = self.conv1(x, edge_index)#, edge_attr=edge_attr)
        updatet = self.layer_norm1(updatet, batch=batch)
        updatet = self.relu(updatet)
        updatet = self.conv2(updatet, edge_index)#, edge_attr=edge_attr)
        
        updatet = self.layer_norm2(updatet, batch=batch)
        updatet = self.relu(updatet)
        # updatet = self.conv3(updatet, edge_index, edge_weight=edge_weight)#, edge_attr=edge_attr)
        # updatet = self.layer_norm3(updatet)
        # updatet = self.conv4(updatet, edge_index, edge_weight=edge_weight)#, edge_attr=edge_attr)
        # updatet = self.layer_norm4(updatet)

        # updatet = self.conv_out(updatet, edge_index, edge_weight=edge_weight)#, edge_attr=edge_attr).tanh()
        # x = x + updatet #* update
        x = x[:, :-2] + updatet# * update

        # temporal nonlinearity
        x[:, -1][x[:, -1] > 1] = 0
        x[:, -1][x[:, -1] < -1] = 0

        x = x / 2
        
        return x
    
    
    def get_output(self, x, softmax=True):
        """
        Returns last n_outputs nodes in x
        Args:
            x: Network state after rule application
        """
        
        # output = self.output_vectorizer(x[-self.n_outputs:, :self.hidden_dim]).squeeze(-1)
        # output = self.output_vectorizer(x[-self.n_outputs:, :3]).squeeze(-1)
        # print(x)
        outputs = []
        for i in range(x.shape[0]//self.n_nodes):
            outputs.append(
                self.output_vectorizer
                (
                x[self.n_non_io_nodes + self.n_inputs + (i*self.n_nodes):self.n_nodes + (i*self.n_nodes), :3]
            ).squeeze(-1))
            
        output = torch.stack(outputs)
        # print(x)
        # print(output)
        
        if softmax:
            output = output.sigmoid()#.softmax(-1)
        # print(output)
        
        # print(x[self.n_non_io_nodes + self.n_inputs + (i*self.n_nodes):self.n_nodes + (i*self.n_nodes), :3])
        return output#.squeeze(0)
        
    
    def reset(self):
        self.vectorized_input = None
        self.vectorized_output = None
        
        # self.conv1.reset()
        # self.conv2.reset()
        # self.conv3.reset()
        # self.conv4.reset()
        # self.conv_out.reset()
        
        
        # self.edge_weight = torch.zeros([162, 1])
        

