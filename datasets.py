import torch
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
import random
import os
import json
import torch.nn.functional as F

class SelfOrganizeTest(torch.nn.Module):
    """
    If Rule can learn this data, then it can somewhat self organize
    """
    def __init__(self) :
        super().__init__()
        self.data = torch.tensor([[0,1], [1,0]])
        self.target = torch.tensor([[0,1], [1,0]])
        
    def __getitem__(self, index):
        return self.data[index], self.target[index]
    def __len__(self):
        return len(self.data)

class ANDDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = torch.tensor([[0,0], [0,1],[1,0], [1,1]])
        self.target = torch.tensor([[1,0],[1,0],[1,0],[0,1]])
        
        
    def __getitem__(self, index):
        return self.data[index], self.target[index]
    def __len__(self):
        return len(self.data)
    
    
class XORDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = torch.tensor([[0,0], [0,1],[1,0], [1,1]])
        self.target = torch.tensor([[1,0],[0,1],[0,1],[1,0]])
        
        
    def __getitem__(self, index):
        return self.data[index], self.target[index]
    def __len__(self):
        return len(self.data)
    
    
class Dataset(torch.utils.data.Dataset):
    """
    datset superclass
    """
    def __init__(self, data, target, metadata=None):
        cuda_device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
        # cuda_device = torch.device("cpu")
        
        self.data = torch.tensor(data).to(cuda_device)
        self.target = torch.tensor(target).to(cuda_device)
        self.metadata = metadata
        
    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return len(self.data)    
    
    def shuffle(self):
        """
        shuffle data and targets together
        """
        new_idx = torch.randperm(len(self.data))
        self.data = self.data[new_idx]
        self.target = self.target[new_idx]
        
        #return a copy of self
        return Dataset(self.data, self.target, self.metadata)        


class TranslateDataset(Dataset):
    def __init__(self, diff=1, drops = None, n=3) -> None:
        if drops is None:
            drops = [random.randint(1, n-1)]
        
        arr = torch.arange(n)
        # for drop in drops:
        #     arr = torch.cat([arr[:drop-1], arr[drop:]])
        
        data = one_hot(arr, n)
        targets = one_hot((arr+diff)%n, n)
        super().__init__(data, targets)


# class TranslateDataset2(Dataset):
#     def __init__(self) -> None:
#         arr = torch.arange(n)
#         if drop:
#             arr = torch.cat([arr[:drop-1], arr[drop:]])
    
#         super().__init__(one_hot(arr, n), one_hot((arr-1)%n, n))
        

class TimeDataset1(Dataset):
    def __init__(self):
        self.data = torch.tensor([[0,1], [0,1], [0,0]])
        self.target = torch.tensor([[0,0,0,0], [0,0,0,0],[1,0,0,0]])
     
class TimeDataset2(Dataset):
    def __init__(self):
        self.data = torch.tensor([[1,0], [1,0], [0,0]])
        self.target = torch.tensor([[0,0,0,0], [0,0,0,0],[0,1,0,0]])
        
    
class TimeDataset3(Dataset):
    def __init__(self):
        self.data = torch.tensor([[1,0], [0,1], [0,0]])
        self.target = torch.tensor([[0,0,0,0], [0,0,0,0],[0,0,1,0]])
        
class TimeDataset4(Dataset):
    def __init__(self):
        self.data = torch.tensor([[1,1], [0,1], [0,0]])
        self.target = torch.tensor([[0,0,0,0], [0,0,0,0],[0,0,0,1]])
        
class TDataset1(Dataset):
    def __init__(self):
        self.data = torch.tensor([[0,1,0,1], [1,0,1,0]])
        self.target = torch.tensor([[0,1], [1,0]])
class TDataset2(Dataset):
    def __init__(self):
        self.data = torch.tensor([[0,1,1,0], [1,0,0,1]])
        self.target = torch.tensor([[1,0], [0,1]])

     
class MetaDataset():
    """
    Dataset of datasets
    """
    def __init__(self):
        # self.datasets = [dataset() for dataset in [TranslateDataset, TranslateDataset2]]
        # self.datasets = [dataset() for dataset in [TranslateDataset]]
        # self.datasets = [dataset() for dataset in [TimeDataset1, TimeDataset2, TimeDataset3, TimeDataset4]]
        # self.datasets = [dataset() for dataset in [FTDataset2, FTDataset3]]
        self.init()
        
        
    def init(self):
        self.datasets = [
            # TranslateDataset(1),
            # TranslateDataset(0),
            # TranslateDataset(2),
            # TranslateDataset(3),
            # TranslateDataset(-1),
            # TranslateDataset(-2),
            TDataset1(),
            TDataset2(),
            ]
        
    def iterate(self):
        """
        returns a generator that gives an shuffled index
        """
        self.init()
        return iter(torch.randperm(len(self.datasets)))

    @property
    def n_inputs(self):
        return self.datasets[0].data.shape[1]

    @property
    def n_outputs(self):
        return self.datasets[0].target.shape[1]
    
    def get_set(self, n, batch_size=1):
        return DataLoader(self.datasets[n] , batch_size=batch_size, shuffle=True)
    
    def get_set_size(self):
        return len(self.datasets[0])
    
class D():
    def __init__(self, datasets, shuffle = True):
        self.datasets = datasets
        self.shuffle = shuffle
        
        self.batch_size = 1
    
    def __iter__(self):
        if self.shuffle:
            new_idx = torch.randperm(len(self.datasets))
            self.datasets = [self.datasets[i].shuffle() for i in new_idx]
        
        datasets_in_batch = [self.datasets[i % len(self.datasets)] for i in range(self.batch_size)]
        
        if self.shuffle:
            datasets_in_batch = [d.shuffle() for d in datasets_in_batch]
        
        
        metadatas = [d.metadata['shape'] for d in datasets_in_batch]
        
        #this is borked
        # for x, metadata in zip(zip(*([d.data for d in datasets_in_batch] + [d.target for d in datasets_in_batch])), [d.metadata for d in datasets_in_batch]):
        #     yield torch.stack(x[0:len(datasets_in_batch)]), torch.stack(x[len(datasets_in_batch):]), metadata
            
        # for x in zip(*([d.data for d in datasets_in_batch] + [d.target for d in datasets_in_batch])):
        #     yield torch.stack(x[0:len(datasets_in_batch)]), torch.stack(x[len(datasets_in_batch):]), None
            
        for x in zip(*([d.data for d in datasets_in_batch] + [d.target for d in datasets_in_batch])):
            yield torch.concat(x[0:len(datasets_in_batch)]), torch.concat(x[len(datasets_in_batch):]), metadatas
            
    def __len__(self):
        return len(self.datasets)
    
    # def iterate(self):
    #     """
    #     returns a generator that gives an shuffled index
    #     """
    #     self.init()
    #     if self.shuffle:
    #         return iter(torch.randperm(len(self.datasets)))

    #     return iter(range(len(self.datasets)))
    @property
    def n_inputs(self):
        return self.datasets[0].data.shape[1]

    @property
    def n_outputs(self):
        return self.datasets[0].target.shape[1]
    
# def load_arc(path = r"C:\Users\lmacl\Google Drive\GitHub\ARC\data\training/", print_filenames = False):
#     max_dim = 3
#     dataset = []

#     for filename in os.listdir(path):
#         json_data = json.loads(open(path + filename).read())
        
#         if len(json_data['train'][0]['input']) > max_dim or len(json_data['train'][0]['input'][0]) > max_dim:
#             continue
        
#         if len(json_data['train'][0]['output']) > max_dim or len(json_data['train'][0]['output'][0]) > max_dim:
#             continue
        
#         try:
#             if torch.tensor([x['input'] for x in json_data['train']]).shape[1] != max_dim:
#                 continue
            
#             if torch.tensor([x['output'] for x in json_data['train']]).shape[1] != max_dim:
#                 continue
            
#             if len(json_data['train']) + len(json_data['test']) < 4:
#                 continue
#         except:
#             continue
        
#         if print_filenames:
#             print(filename)
            
#         json_data['filename'] = filename
#         dataset.append(json_data)
    
#     return dataset
def load_arc(path = r"C:\Users\lmacl\Google Drive\GitHub\ARC\data\training/", print_filenames = False, max_dim = 3, max_length = 3):
    dataset = []

    for filename in os.listdir(path):
        json_data = json.loads(open(path + filename).read())
        
        if len(json_data['train'][0]['input']) > max_dim or len(json_data['train'][0]['input'][0]) > max_dim:
            continue
        
        if len(json_data['train'][0]['output']) > max_dim or len(json_data['train'][0]['output'][0]) > max_dim:
            continue
        
        try:
            a = torch.tensor([x['input'] for x in json_data['train'] + json_data['test']])#.shape[1]
            b = torch.tensor([x['output'] for x in json_data['train'] + json_data['test']])#.shape[1]
            torch.concat([a, b], dim =0)
        except:
            continue
        
        # if len(json_data['train']) + len(json_data['test']) != max_length:
        #     continue
            
        if print_filenames:
            print(filename)
            
        json_data['filename'] = filename
        dataset.append(json_data)
    
    return dataset
def process_arc_sample(sample, io, final_dim=3, pad = False, d_len = 4):
    sample = torch.tensor([x[io] for x in sample['train'] + sample['test']])
    if pad:
        sample = F.pad(sample, (0, final_dim - sample.shape[2], 0, final_dim - sample.shape[1]))
    sample = one_hot(sample.flatten(1,2), 10)[:d_len]
    return sample