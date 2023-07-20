import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli,Categorical
from torch.distributions import RelaxedBernoulli

class BaseCNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(BaseCNN, self).__init__()
    
        self.conv= nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=5,stride=1)
        
        self.conv_pad_1_64 =  nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm1d(64,momentum=0.1),
            nn.ReLU()
        )
        self.conv_pad_2_64 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm1d(64,momentum=0.1),
            nn.ReLU()
        )
        
        self.conv_pad_1_128 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm1d(128,momentum=0.1),
            nn.ReLU()
        )
        self.conv_pad_2_128 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm1d(128,momentum=0.1),
            nn.ReLU()
        )
        
        self.conv_pad_1_256 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm1d(256,momentum=0.1),
            nn.ReLU()
        )
        self.conv_pad_2_256 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm1d(256,momentum=0.1),
            nn.ReLU()
        )
        self.conv_pad_3_256 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm1d(256,momentum=0.1),
            nn.ReLU()
        )
        
        self.conv_pad_1_512 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm1d(512,momentum=0.1),
            nn.ReLU()
        )
        self.conv_pad_2_512 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm1d(512,momentum=0.1),
            nn.ReLU()
        )
        self.conv_pad_3_512 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm1d(512,momentum=0.1),
            nn.ReLU()
        )
        self.conv_pad_4_512 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm1d(512,momentum=0.1),
            nn.ReLU()
        )
        self.conv_pad_5_512 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm1d(512,momentum=0.1),
            nn.ReLU()
        )
        self.conv_pad_6_512 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm1d(512,momentum=0.1),
            nn.ReLU()
        )
        
        self.maxpool_1 = nn.MaxPool1d(kernel_size=3,stride=3) 
        self.maxpool_2 = nn.MaxPool1d(kernel_size=3,stride=3) 
        self.maxpool_3 = nn.MaxPool1d(kernel_size=3,stride=3) 
        self.maxpool_4 = nn.MaxPool1d(kernel_size=3,stride=3) 
        self.maxpool_5 = nn.MaxPool1d(kernel_size=3,stride=3) 

        # *4 for the series length 1000
        # *2 for the series length 500
        self.dense1 = nn.Linear(512 * 4, 1024)
        self.dense2 = nn.Linear(1024, hidden_size)
        
        self.dense_final = nn.Linear(hidden_size, num_classes)
        #self.softmax= nn.LogSoftmax(dim=1)

    def forward(self, x):
        #print(x.shape)
        x = x.permute(0,2,1)
        x = self.conv_pad_1_64(x)
        x = self.conv_pad_2_64(x)
        x = self.maxpool_1(x)
        
        
        x = self.conv_pad_1_128(x)
        x = self.conv_pad_2_128(x)
        x = self.maxpool_2(x)
        
        x = self.conv_pad_1_256(x)
        x = self.conv_pad_2_256(x)
        x = self.conv_pad_3_256(x)
        x = self.maxpool_3(x)
        
        x = self.conv_pad_1_512(x)
        x = self.conv_pad_2_512(x)
        x = self.conv_pad_3_512(x)
        x = self.maxpool_4(x)
        
        x = self.conv_pad_4_512(x)
        x = self.conv_pad_5_512(x)
        x = self.conv_pad_6_512(x)
        x = self.maxpool_5(x)

        # *4 for the series length 1000
        # *2 for the series length 500
        x = x.view(-1, 512 * 4) #Reshape (current_dim, 32*2)
        #print(x.shape)
        x = self.dense1(x)
        #print(x.shape)
        x= self.dense2(x)
        
        return x

# class ReverseCNN(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(ReverseCNN, self).__init__()
#         
#         self.re_conv_pad_1_64 =  nn.Sequential(
#             nn.Conv1d(in_channels=64, out_channels=input_size, kernel_size=3,stride=1,padding=1),
#             nn.BatchNorm1d(input_size,momentum=0.01),
#             nn.ReLU()
#         )
#         self.re_conv_pad_2_64 = nn.Sequential(
#             nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride=1,padding=1),
#             nn.BatchNorm1d(64,momentum=0.01),
#             nn.ReLU()
#         )
#         
#         self.re_conv_pad_1_128 = nn.Sequential(
#             nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3,stride=1,padding=1),
#             nn.BatchNorm1d(64,momentum=0.01),
#             nn.ReLU()
#         )
#         self.re_conv_pad_2_128 = nn.Sequential(
#             nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride=1,padding=1),
#             nn.BatchNorm1d(128,momentum=0.01),
#             nn.ReLU()
#         )
#         
#         self.re_conv_pad_1_256 = nn.Sequential(
#             nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3,stride=1,padding=1),
#             nn.BatchNorm1d(128,momentum=0.01),
#             nn.ReLU()
#         )
#         self.re_conv_pad_2_256 = nn.Sequential(
#             nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride=1,padding=1),
#             nn.BatchNorm1d(256,momentum=0.01),
#             nn.ReLU()
#         )
#         self.re_conv_pad_3_256 = nn.Sequential(
#             nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride=1,padding=1),
#             nn.BatchNorm1d(256,momentum=0.01),
#             nn.ReLU()
#         )
#         
#         self.re_conv_pad_1_512 = nn.Sequential(
#             nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3,stride=1,padding=1),
#             nn.BatchNorm1d(256,momentum=0.01),
#             nn.ReLU()
#         )
#         self.re_conv_pad_2_512 = nn.Sequential(
#             nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride=1,padding=1),
#             nn.BatchNorm1d(512,momentum=0.01),
#             nn.ReLU()
#         )
#         self.re_conv_pad_3_512 = nn.Sequential(
#             nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride=1,padding=1),
#             nn.BatchNorm1d(512,momentum=0.01),
#             nn.ReLU()
#         )
#         self.re_conv_pad_4_512 = nn.Sequential(
#             nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride=1,padding=1),
#             nn.BatchNorm1d(512,momentum=0.01),
#             nn.ReLU()
#         )
#         self.re_conv_pad_5_512 = nn.Sequential(
#             nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride=1,padding=1),
#             nn.BatchNorm1d(512,momentum=0.01),
#             nn.ReLU()
#         )
#         self.re_conv_pad_6_512 = nn.Sequential(
#             nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride=1,padding=1),
#             nn.BatchNorm1d(512,momentum=0.01),
#             nn.ReLU()
#         )
#         
#         self.maxunpool_1 = nn.MaxUnpool1d(kernel_size=3,stride=3)
#         self.maxunpool_2 = nn.MaxUnpool1d(kernel_size=3,stride=3)
#         self.maxunpool_3 = nn.MaxUnpool1d(kernel_size=3,stride=3)
#         self.maxunpool_4 = nn.MaxUnpool1d(kernel_size=3,stride=3)
#         self.maxunpool_5 = nn.MaxUnpool1d(kernel_size=3,stride=3)
#         
#         self.re_dense1 = nn.Linear(1024, 512 * 4)
#         
#         self.re_dense2 = nn.Linear(hidden_size, 1024)
#         
#     def forward(self, x, indices):
#         
#         ''' reverse architecture '''
#         
#         y = self.re_dense2(x)
#         y = self.re_dense1(y)
#         y = y.view(-1,512,4)
#         
#         y = self.maxunpool_5(y,indices[4])
#         y = self.re_conv_pad_6_512(y)
#         y = self.re_conv_pad_5_512(y)
#         y = self.re_conv_pad_4_512(y)
#         
#         
#         y = self.maxunpool_4(y,indices[3],output_size=torch.Size([1,256, 37]))
#         y = self.re_conv_pad_3_512(y)
#         y = self.re_conv_pad_2_512(y)
#         y = self.re_conv_pad_1_512(y)
#         
#         y = self.maxunpool_3(y,indices[2],output_size=torch.Size([1,128, 111]))
#         y = self.re_conv_pad_3_256(y)
#         y = self.re_conv_pad_2_256(y)
#         y = self.re_conv_pad_1_256(y)
#         
#         y = self.maxunpool_2(y,indices[1],output_size=torch.Size([1,64, 333]))
#         y = self.re_conv_pad_2_128(y)
#         y = self.re_conv_pad_1_128(y)
#         
#         y = self.maxunpool_1(y,indices[0],output_size=torch.Size([1,12, 1000]))
#         y = self.re_conv_pad_2_64(y)
#         y = self.re_conv_pad_1_64(y)
#         y = y.permute(0,2,1)
#         
#         return y

class BaseRNN(nn.Module):

    def __init__(self,
                 N_FEATURES,
                 HIDDEN_DIM,
                 CELL_TYPE="LSTM",
                 N_LAYERS=1):
        super(BaseRNN, self).__init__()

        # --- Mappings ---
        if CELL_TYPE in ["RNN", "LSTM", "GRU"]:
            self.rnn = getattr(nn, CELL_TYPE)(N_FEATURES,
                                              HIDDEN_DIM,
                                              N_LAYERS)
        else:
            try: 
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[CELL_TYPE]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was
                                 supplied, options are ['LSTM', 'GRU',
                                 'RNN_TANH' or 'RNN_RELU']""")

            self.rnn = nn.RNN(N_FEATURES,
                              HIDDEN_DIM,
                              N_LAYERS,
                              nonlinearity=nonlinearity)
        self.tanh = nn.Tanh()

    def forward(self, x_t, hidden):
        output, h_t = self.rnn(x_t, hidden)
        return output, h_t

class Controller(nn.Module):

    def __init__(self, input_size, output_size, isCuda = True):
        super(Controller, self).__init__()
        self.isCuda = isCuda
        self.fc = nn.Linear(input_size, output_size)
        '''
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.Linear(128, output_size),
        )
        '''
        self.softmax = nn.Softmax(dim=1)
        self.log_sig = nn.LogSigmoid()
        
    def forward(self, h_t, eps=0.05, train=False):

        probs = torch.sigmoid(self.fc(h_t.detach())) # Compute halting-probability
                
        if(self.isCuda): 
            if(train): 
                probs = (1-eps) * probs + eps * torch.FloatTensor([1]).cuda() # Add randomness according to eps
            else:
                probs = (1-eps) * probs + eps * torch.FloatTensor([1]).cuda()
                      
        else:
            probs = (1-eps) * probs + eps * torch.FloatTensor([0.5]) # Add randomness according to eps
        
        '''
        m = Categorical(probs=probs) # Define bernoulli distribution parameterized with predicted probability
        
        halt = m.sample() # Sample action
        
        log_pi = m.log_prob(halt).unsqueeze(1) # Compute log probability for optimization
        '''
        m = Bernoulli(probs=probs) # Define bernoulli distribution parameterized with predicted probability
        
        halt = m.sample() # Sample action
        
        log_pi = m.log_prob(halt) # Compute log probability for optimization
        
        return halt, log_pi, -torch.log(probs), probs

class Controller_New(nn.Module):
    """
    A network that chooses whether or not enough information
    has been seen to predict a label of a time series.
    """
    def __init__(self, ninp, nout):
        super(Controller_New, self).__init__()

        # --- Mappings ---
        self.fc = nn.Linear(ninp, nout)  # Optimized w.r.t. reward

    def forward(self, x):
        probs = torch.sigmoid(self.fc(x))
        probs = (1-self._epsilon)*probs + self._epsilon*torch.FloatTensor([0.05])  # Explore/exploit
        m = Bernoulli(probs=probs)
        action = m.sample() # sample an action
        
        log_pi = m.log_prob(action) # compute log probability of sampled action
        return action.squeeze(0), log_pi.squeeze(0), -torch.log(probs).squeeze(0)


class RelaxedController(nn.Module):

    def __init__(self, input_size, output_size, isCuda = True):
        super(RelaxedController, self).__init__()
        self.isCuda = isCuda
        self.fc = nn.Linear(input_size, output_size)
        self.softmax = nn.Softmax(dim=1)
        self.log_sig = nn.LogSigmoid()

    def forward(self, h_t, eps=0.):

        probs = torch.sigmoid(self.fc(h_t.detach())) # Compute halting-probability
        
        #probs = (1-eps) * probs + eps * torch.FloatTensor([0.5]).cuda() # Add randomness according to eps
        
        temperature = torch.FloatTensor([15]).cuda()
        m = RelaxedBernoulli(temperature = temperature ,probs= probs) # Define bernoulli distribution parameterized with predicted probability
        
        halt = m.sample() # Sample action
        
        log_pi = -torch.log(halt) # Compute log probability for optimization

        return halt, log_pi, -torch.log(probs), probs

class ControllerCPU(nn.Module):

    def __init__(self, input_size, output_size):
        super(ControllerCPU, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.softmax = nn.Softmax(dim=1)
        self.log_sig = nn.LogSigmoid()

    def forward(self, h_t, decade = 2, eps=0.):

        probs = torch.sigmoid(self.fc(h_t)*-decade) # Compute halting-probability
        probs = (1-eps) * probs + eps * torch.FloatTensor([0.5]) # Add randomness according to eps
        
        m = Bernoulli(probs=probs) # Define bernoulli distribution parameterized with predicted probability
        halt = m.sample() # Sample action
        log_pi = m.log_prob(halt) # Compute log probability for optimization
        return halt, log_pi, -torch.log(probs), probs

class Discriminator(nn.Module):

    def __init__(self, input_size, output_size):
        super(Discriminator, self).__init__()
        
        self.fc = nn.Linear(input_size, output_size)
        '''
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.Linear(128, output_size),
        )
        '''
        self.softmax = nn.LogSoftmax(dim=1) #test sofmax

    def forward(self, h_t):

        y_hat = self.fc(h_t)
        y_hat = self.softmax(y_hat)
        
        return y_hat

class BaselineNetwork(nn.Module):

    def __init__(self, input_size, output_size):
        super(BaselineNetwork, self).__init__()

        # --- Mappings ---
        self.fc = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, h_t):
        b_t = self.fc(h_t.detach())
        #b_t = self.relu(b_t)
        return b_t 

class Attention(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.
    **Thank you** to IBM for their initial implementation of :class:`Attention`. Here is
    their `License
    <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__.
    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:
            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`
    Example:
         >>> attention = Attention(256)
         >>> query = torch.randn(5, 1, 256)
         >>> context = torch.randn(5, 5, 256)
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([5, 1, 256])
         >>> weights.size()
         torch.Size([5, 1, 5])
    """

    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.
        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.view(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.view(batch_size, output_len, dimensions)

        # TODO: Include mask on PADDING_INDEX?

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)

        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights
