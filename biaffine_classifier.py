import torch
import torch.nn as nn
class Biaffine(nn.Module):
    def __init__(self, network_kwargs):
        super(Biaffine,self).__init__()
        self.span_start = FFNN(network_kwargs['input_size'], network_kwargs['hidden_size'],
                              network_kwargs['dropout'])
        self.span_end = FFNN(network_kwargs['input_size'], network_kwargs['hidden_size'],
                                network_kwargs['dropout'])
        
    @classmethod
    def biaffine_mapping(v1,v2,numClasses=2):
        batch_size = v1.shape[0]
        bucket_size = v1.shape[1]
        v1 = torch.cat(v1,torch.ones([batch_size, bucket_size, 1]), axis=2)
        v2 = torch.cat(v2,torch.ones([batch_size, bucket_size, 1]), axis=2)
        
        v1_size = v1.shape[-1]
        v2_size = v2.shape[-1]
        
        bilinear_map = nn.Parameter(torch.FloatTensor(v1_size,numClasses,v2_size))
        
        v1 = v1.reshape((-1,v1_size))
        bilinear_map = bilinear_map.reshape((v1_size,-1))
        
        bilinear_mapping = torch.matmul(v1, bilinear_map)
        bilinear_mapping = bilinear_mapping.reshape((batch_size, bucket_size * numClasses, v2_size))
        
        bilinear_mapping = torch.matmul(bilinear_mapping, v2.transpose(1, -1))
        bilinear_mapping = bilinear_mapping.reshape((batch_size, bucket_size, numClasses, bucket_size))
        
        bilinear_mapping = bilinear_mapping.transpose(-2, -1)
        
        return bilinear_mapping  
    
    def forward(self, start_representations, end_representations):
        span_start = self.span_start(start_representations)
        span_end = self.span_end(end_representations)
        loss = self.biaffine_mapping(span_start,span_end)

        return loss 

class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        output=self.dropout(self.sigmoid(self.fc1(x)))
        output = self.dropout(self.sigmoid(self.fc2(output)))
        return output
