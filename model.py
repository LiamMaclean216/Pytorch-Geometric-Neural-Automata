from locale import normalize
from turtle import forward
from torch_geometric.nn import GCNConv, Sequential, GATConv, GATv2Conv
from torch_geometric.nn.norm import LayerNorm, PairNorm
 
from torch_geometric_temporal.nn.attention import STConv
from torch_geometric_temporal.nn.recurrent import GConvGRU, A3TGCN


from torch.nn import ReLU, LeakyReLU
import torch
import torch.nn as nn
from utils import *
from torch_geometric.data import Data
import torch_geometric.utils as utils
import torch.nn.functional as F

import numpy as np

# cuda_device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
cuda_device = torch.device("cpu")

class UpdateRule(torch.nn.Module):
    def __init__(self, 
                n_inputs, 
                n_outputs,
                hidden_dim,
                edge_dim,
                network_width = 80,
                heads = 4):
        
        super(UpdateRule, self).__init__()
        # torch.manual_seed(12312345)
        
        self.edge_dim = edge_dim
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.hidden_dim = hidden_dim
        skip_size = hidden_dim
        self.total_hidden_dim = hidden_dim# + skip_size
        
        # self.initial = nn.parameter.Parameter(
        #     torch.zeros([n_outputs + n_inputs, self.total_hidden_dim]), requires_grad=True
        # ).to(cuda_device)
        
        
       
        # self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.relu = nn.ReLU()
        
        # self.input_vectorizer = nn.Linear(n_inputs, self.total_hidden_dim, bias=True)
        self.input_vector_size = 8
        self.input_vectorizer = nn.Linear(1, self.input_vector_size)
        
        # Vectorizes training targets
        self.reverse_output_vectorizer = nn.Linear(2, self.input_vector_size)
        self.output_vectorizer = nn.Linear(hidden_dim, 1)
        
        
        
        # self.conv1 = GATConv(2*self.total_hidden_dim+1, network_width, heads = heads, edge_dim = edge_dim)
        self.layer_norm1 = LayerNorm(network_width*heads)
        self.layer_norm2 = LayerNorm(network_width*heads)
        self.layer_norm3 = LayerNorm(network_width*heads)
        self.layer_norm4 = LayerNorm(network_width*heads)
        self.layer_norm5 = LayerNorm(network_width*heads)
        
        
        self.layer_norm6 = LayerNorm(hidden_dim)
        
        self.conv1 = GCNConv(self.total_hidden_dim+1, network_width, aggr='max')
        
        # self.linear1 = nn.Linear(network_width*heads, network_width*heads)
        # self.linear2 = nn.Linear(network_width*heads, network_width*heads)
        
        
        # self.conv1 = A3TGCN(self.total_hidden_dim, hidden_dim, 5)
        
        self.n_hidden_rule_layers = 1
        # self.conv2 = Sequential('x, edge_index', 
        #                                      [x for i in range(n_hidden_rule_layers) for x in 
        #                                       [(GCNConv(width, width), 'x, edge_index -> x'), LeakyReLU(0.1, True)]])
        # self.conv2 = GATConv(network_width*heads, network_width, heads = heads, edge_dim = edge_dim)
        # self.conv2 = A3TGCN(network_width, network_width, 5)
        self.conv2 = GCNConv(network_width, network_width, aggr='max')
        self.conv3 = GCNConv(network_width, network_width, aggr='max')
        self.conv4 = GCNConv(network_width, network_width, aggr='max')
        self.conv5 = GCNConv(network_width, network_width, aggr='max')
        
        # self.forgor = nn.Linear(network_width, 2)
        self.forgor1 = GCNConv(network_width+1, network_width, aggr='max')
        self.forgor2 = GCNConv(network_width, hidden_dim+1, aggr='max')
         
         
        self.update1 = GCNConv(network_width+1, network_width, aggr='max')
        self.update2 = GCNConv(network_width, hidden_dim, aggr='max')
        
        # self.conv_out = GATConv(network_width*heads, hidden_dim, heads = 1, edge_dim = edge_dim)
        self.conv_out = GCNConv(network_width, hidden_dim, aggr='max')
        
        self.reset()


    def initial_state(self, height = None, width = None, build = False):
        if height is None:
            height = self.height
            
        if width is None:
            width = self.width
        
        
        #trained initial on only input and output nodes
        # if height is not None and width is not None:
        #     self.initial = nn.parameter.Parameter(
        #         torch.zeros([self.n_outputs + self.n_inputs, self.total_hidden_dim]), requires_grad=True
        #     ).to(cuda_device)
        
        # return torch.concat(
        #     (torch.zeros([height*width, self.total_hidden_dim]).to(cuda_device),
        #     self.initial)
        # )
        
        # Trained initial on all nodes including hidden
        if build:
            self.initial = nn.parameter.Parameter(
                torch.zeros([height*width + self.n_outputs + self.n_inputs, self.total_hidden_dim])#, requires_grad=True
            ).to(cuda_device)
        
        return self.initial
        
    
    def build_graph(self, height, width):
        self.initial_state(height, width, build=True)
        self.width = width
        self.height = height
        
        # Build initial state
        n_nodes = height*width + self.n_inputs + self.n_outputs
        # self.initial = nn.parameter.Parameter(torch.zeros([n_nodes, self.total_hidden_dim]), requires_grad=True)
        
        # Build graph
        edges = build_edges(self.n_inputs, self.n_outputs, height, width)
        
        self.graph = Data(edge_index=edges, x=torch.zeros(n_nodes, self.total_hidden_dim))
        self.edge_index = self.graph.edge_index.long().clone()
        
        self.edge_attr = nn.parameter.Parameter(
            torch.zeros([self.graph.edge_index[0].shape[0], self.edge_dim])
        ).to(cuda_device)
        
    
    def draw(self):
        graph = utils.to_networkx(self.graph, to_undirected=True, remove_self_loops = True)
        nx.draw(graph)
    
    
    def vectorise_input(self, x, input_data):
        mask = torch.zeros_like(x)
        
        mask[
                -(self.n_inputs+self.n_outputs):-self.n_outputs, :self.input_vector_size
            ] = self.input_vectorizer(input_data).squeeze(-1)
        
        x = x + mask

        return x
            
    def vectorize_output(self, x, input_data):
        mask = torch.zeros_like(x)
        
        mask[
                -self.n_outputs:, :self.input_vector_size
            ] = self.reverse_output_vectorizer(input_data).squeeze(-1)

        x = x + mask
        return x
    
    def forward(self, x, n_steps, data, plug_output_data = False, return_all = True, edge_attr = None):
        network_in = []
        network_out = []
        
        for idx, (problem_data_x, problem_data_y) in enumerate(data):
            last = idx == 1#len(data) - 1
            
            # network_in.append(problem_data_x.float().squeeze(0).numpy())
            # network_out.append(problem_data_y.float().squeeze(0).numpy())
            
            problem_data_y_ = problem_data_y.float().to(cuda_device).unsqueeze(-1) 
            problem_data_y_ = torch.cat((problem_data_y_, torch.ones_like(problem_data_y_)), dim = 2)
            
            # if idx != len(data) - 1:
            if not last:
                x = self.vectorize_output(x, problem_data_y_)
            else:
                x = self.vectorize_output(x, torch.zeros_like(problem_data_y_))
            
            # problem_data_x = problem_data_x.repeat(n_steps, 1)#.unsqueeze(0).transpose(1,2)
            input_data = problem_data_x.float().unsqueeze(-1).to(cuda_device)
            
            # if last:
            x = self.vectorise_input(x, input_data)
            
            
            
            for _ in range(n_steps):
                if not last:
                    x = torch.cat((torch.zeros([x.shape[0], 1]).to(cuda_device), x), dim = 1)
                else:
                    x = torch.cat((torch.ones([x.shape[0], 1]).to(cuda_device), x), dim = 1)
                x = self.step(x, edge_attr=edge_attr)
            
            if last:
                break
            # break
                
        # print("#################")    
        network_output = self.get_output(x)
        # loss = F.binary_cross_entropy_with_logits(network_output, problem_data_y.float().squeeze(0))
        
        #l2 loss
        loss = F.mse_loss(network_output, problem_data_y.float().squeeze(0))
        
        loss /= len(data)
        
        if return_all:
            return (
                x, 
                loss, 
                network_output.detach().numpy(),
                problem_data_y.float().squeeze(0).numpy(), 
                # problem_data_x.float().squeeze(0).numpy()
                # np.array(network_in),
                np.array(network_out)
            )
        
        return x
        
    
    def act(self, x, env_state, n_step = 1):
        input_data = env_state.float().unsqueeze(-1).to(cuda_device)
        x = self.vectorise_input(x, input_data)
        for _ in range(n_step):
            x = self.step(x, env_state.float(), None)
        
        return x
        
    
    def step(self, x, edge_attr = None):
        # skip = x[:, :-1]
        
        # print(x.shape)
        
        forgor = self.forgor1(x, self.edge_index)
        forgor = self.forgor2(forgor, self.edge_index).sigmoid()
        x = x * forgor
        
        
        update = self.update1(x, self.edge_index)
        update = self.update2(update, self.edge_index).sigmoid() 
        
        # x = torch.cat((x, self.initial_state()), dim = -1)
        # print(x.shape)
        
        
        updatet = self.conv1(x, self.edge_index)
        # updatet = self.layer_norm1(updatet)
        updatet = self.relu(updatet)
        
        
        updatet = self.conv2(updatet, self.edge_index)
        # updatet = self.layer_norm2(updatet)
        # updatet = self.relu(updatet)
        
        # x = self.conv3(x, self.edge_index)
        # x = self.layer_norm3(x)
        # x = self.relu(x)
        
        # x = self.conv4(x, self.edge_index)
        # x = self.layer_norm4(x)
        # x += skip
        # skip = x
        # x = self.relu(x)
        
        # x = self.conv5(x, self.edge_index)
        # x = self.layer_norm5(x)
        # x = self.relu(x)
            
        updatet = self.conv_out(updatet, self.edge_index).tanh()
        # updatet = self.layer_norm6(updatet)
        # x = PairNorm()(x)
        
        x = x[:, :-1] + updatet * update
        
        # forgor = self.forgor(skip).softmax(dim = -1)
        
        # x = x * forgor[:, 0].unsqueeze(-1) + skip * forgor[:, 1].unsqueeze(-1) 
        
        # x += skip 
        
        
        return x
    
    
    def get_output(self, x, softmax=True):
        """
        Returns last n_outputs nodes in x
        Args:
            x: Network state after rule application
        """
        
        output = self.output_vectorizer(x[-self.n_outputs:, :self.hidden_dim]).squeeze(-1)
        
        if softmax:
            output = output.softmax(-1)
        
        return output
        
    
    def reset(self):
        self.vectorized_input = None
        self.vectorized_output = None
        
