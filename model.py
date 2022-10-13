from locale import normalize
from turtle import forward
from torch_geometric.nn import GCNConv, Sequential, GATConv, GATv2Conv
from torch_geometric.nn.norm import GraphNorm
 
from torch_geometric_temporal.nn.attention import STConv
from torch_geometric_temporal.nn.recurrent import GConvGRU, A3TGCN


from torch.nn import ReLU, LeakyReLU
import torch
import torch.nn as nn
from utils import *
from torch_geometric.data import Data
import torch_geometric.utils as utils
import torch.nn.functional as F

# cuda_device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
cuda_device = torch.device("cpu")

class UpdateRule(torch.nn.Module):
    def __init__(self, 
                n_inputs, 
                n_outputs,
                hidden_dim,
                edge_dim,
                network_width = 80):
        
        super(UpdateRule, self).__init__()
        # torch.manual_seed(12312345)
        
        
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.hidden_dim = hidden_dim
        skip_size = hidden_dim
        self.total_hidden_dim = hidden_dim# + skip_size
        
        self.initial = nn.parameter.Parameter(
            torch.zeros([n_outputs + n_inputs, self.total_hidden_dim]), requires_grad=True
        ).to(cuda_device)
        # self.initial = nn.parameter.Parameter(torch.zeros([1, self.total_hidden_dim]), requires_grad=True).expand(n_outputs, -1)
        # self.initial = torch.zeros([n_outputs, self.total_hidden_dim])
       
        # self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.relu = nn.ReLU()
        
        # self.input_vectorizer = nn.Linear(n_inputs, self.total_hidden_dim, bias=True)
        self.input_vector_size = 2
        self.input_vectorizer = nn.Linear(1, self.input_vector_size, bias=True)
        
        # Vectorizes training targets
        self.reverse_output_vectorizer = nn.Linear(n_outputs, self.total_hidden_dim)
        self.output_vectorizer = nn.Linear(hidden_dim, 1)
        
        heads = 1
        self.conv1 = GATConv(self.total_hidden_dim, network_width, heads = heads, edge_dim = edge_dim)
        # self.conv1 = GCNConv(self.total_hidden_dim, network_width, aggr='max')
        
        
        # self.conv1 = A3TGCN(self.total_hidden_dim, hidden_dim, 5)
        
        self.n_hidden_rule_layers = 1
        # self.hidden_rule_layers = Sequential('x, edge_index', 
        #                                      [x for i in range(n_hidden_rule_layers) for x in 
        #                                       [(GCNConv(width, width), 'x, edge_index -> x'), LeakyReLU(0.1, True)]])
        self.hidden_rule_layers = GATConv(network_width, network_width, heads = heads, edge_dim = edge_dim)
        # self.hidden_rule_layers = A3TGCN(network_width, network_width, 5)
        # self.hidden_rule_layers = GCNConv(network_width, network_width, aggr='max')
        
        self.conv_out = GATConv(network_width, hidden_dim, heads = heads, edge_dim = edge_dim)
        # self.conv_out = GCNConv(network_width, hidden_dim, aggr='max')
        
        self.reset()


    def initial_state(self, height = None, width = None):
        if height is None:
            height = self.height
            
        if width is None:
            width = self.width
        
        return torch.concat(
            (torch.zeros([height*width, self.total_hidden_dim]).to(cuda_device),
            self.initial)
        )
    
    def build_graph(self, height, width):
        self.width = width
        self.height = height
        
        # Build initial state
        n_nodes = height*width + self.n_inputs + self.n_outputs
        # self.initial = nn.parameter.Parameter(torch.zeros([n_nodes, self.total_hidden_dim]), requires_grad=True)
        
        # Build graph
        edges = build_edges(self.n_inputs, self.n_outputs, height, width)
        
        self.graph = Data(edge_index=edges, x=torch.zeros(n_nodes, self.total_hidden_dim))
        self.edge_index = self.graph.edge_index.long().clone()
    
    def draw(self):
        graph = utils.to_networkx(self.graph, to_undirected=True, remove_self_loops = True)
        nx.draw(graph)
    
    
    def vectorise_input(self, x, input_data):
        x[
                -(self.n_inputs+self.n_outputs):-self.n_outputs, :self.input_vector_size
            ] = self.input_vectorizer(input_data).squeeze(-1)

        return x
            
    
    
    def forward(self, x, n_steps, data, plug_output_data = False, return_all = True, edge_attr = None):
        for idx, (problem_data_x, problem_data_y) in enumerate(data):
            output_data = None#torch.zeros_like(problem_data_y.float())
            # if idx == meta_set.get_set_size() - 1:
            #     output_data = None#problem_data_y.float()
            
            
            # problem_data_x = problem_data_x.repeat(n_steps, 1)#.unsqueeze(0).transpose(1,2)
            # print(problem_data_x.shape)
            input_data = problem_data_x.float().unsqueeze(-1).to(cuda_device)
            
            
            x = self.vectorise_input(x, input_data)
            
            
            for _ in range(n_steps):
                x = self.step(x, problem_data_x.float(), output_data, edge_attr=edge_attr)
                
            
        network_output = self.get_output(x)
        loss = F.binary_cross_entropy_with_logits(network_output, problem_data_y.float().squeeze(0))
        
        if return_all:
            return x, loss, network_output.detach().numpy(), problem_data_y.float().squeeze(0).numpy()
        
        return x
        
    
    def act(self, x, env_state, n_step = 1):
        input_data = env_state.float().unsqueeze(-1).to(cuda_device)
        x = self.vectorise_input(x, input_data)
        for _ in range(n_step):
            x = self.step(x, env_state.float(), None)
        
        return x
        
    
    def step(self, x, input_data, output_data = None, edge_attr = None):
        skip = x
        
        # x = self.relu(x)
        
        if output_data is not None:
            mask = torch.zeros(x.shape)
            mask[-self.n_outputs:, :self.hidden_dim] = 1
            x = x+((self.reverse_output_vectorizer(output_data)-x) * mask)
            
        # print(edge_attr.shape, self.edge_index.shape)
        
        x = self.conv1(x, self.edge_index, edge_attr=edge_attr)
        x = self.relu(x)
        for _ in range(self.n_hidden_rule_layers):
            # inner_skip = x
            x = self.hidden_rule_layers(x, self.edge_index, edge_attr=edge_attr)
            x = self.relu(x)
            # x = x + inner_skip
            
        
        x = self.conv_out(x, self.edge_index, edge_attr=edge_attr)
        
        x += skip
        
        
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
        
