import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 


#Parameters of the model
class parameters():
    

  def __init__(self,num_features,k_h,conv_o,rnn_o,skip,skip_o,ar_window,ar_features,dropout,n_out,device):
    '''
    Parameters:
    num_features : the number of the input features 
    k_h : the kernel height of the convolutional layer
    conv_o : the number of the convolutional output chennels
    rnn_o : the number of the outputs of the recurrent layer
    skip : the number of skipped steps
    skip_o : the number of the outputs for the recurrent skip layer
    ar_window : the number of samples to be passed directly to the autoregressive component
    ar_features : the selected features to be passed to the auto regressive component
    n_out : the number of the final outputs
    dropout : probability for dropout
    device : to specify current work environment 'cuda' or 'cpu'
    '''
    self.num_features = num_features
    self.k_h = k_h
    self.conv_o = conv_o

    self.rnn_o = rnn_o
    
    self.skip = skip
    self.skip_o = skip_o
    self.ar_window = ar_window
    self.ar_features = ar_features
    self.dropout = dropout
    self.n_out = n_out 

    self.device = device 

# The LSTNet Model
class Model(nn.Module):
    
    def __init__(self,params):
        super(Model, self).__init__()
        self.num_features = params.num_features #Number of input features
        
        self.conv_o = params.conv_o # Number of the convolutional layer's output channels 10
        self.conv_k_h = params.k_h # the kerenl hight of the convolutional layer 10

        self.rnn_o = params.rnn_o #Output channels of the recurrent units 64 

        self.skip = params.skip # number of skipped steps [4, 24] 
        self.skip_o = params.skip_o #Output Channels of the recurrent skip layer[4, 4] 

        self.n_out  = params.n_out # number of output features

        self.ar_window = params.ar_window # length of the sequence to be used by the autoregressor 
        self.ar_features = params.ar_features # selected features to be used // // // 

        self.dropout = nn.Dropout(p = params.dropout)

        self.device = params.device
       
        # Initiation of the Convolutional Layer

        self.conv = nn.Conv2d(1, self.conv_o, 
                               kernel_size=(self.conv_k_h, self.num_features))

        # Initiation of the recurrent unit rnn 
        self.GRU = nn.GRU(self.conv_o, self.rnn_o, batch_first=True)

        # Recurrent Skip Part 
        self.skip_GRU = {}
        for i in range(len(self.skip)):
            self.skip_GRU[i] = nn.GRU(self.conv_o, self.skip_o[i], batch_first=True).to(self.device)
        
        # number of the learned features from the output of the rnn and the skipped rnn 
        self.out_features = self.rnn_o + np.dot(self.skip, self.skip_o)

        # Output layer
        self.output = nn.Linear(self.out_features, self.n_out)

        #The autoregressive which is a linear layer
        self.ar = nn.Linear(self.ar_window, self.n_out)
        
    def forward(self, X):
        """
        Parameters:
        X (tensor) [batch_size, time_steps, num_features]
        """
        batch_size = X.size(0)
        
        # Pass the input to the convolutional layer Convolutional Layer
        C = X.unsqueeze(1) # add addisional dimension to X to fed it into the conv-layer 
        # X will have a number of channels =1
        C = F.relu(self.conv(C)) 
        C = self.dropout(C)
        C = torch.squeeze(C, 3) #remove the 4th dimension which is 1
        # Shape of C is (batch_size,conv_out_channels,sequence_length)

        # Pass the output of the conv. to the Recurrent Layer
        R = C.permute(0, 2, 1) 
        out, hidden = self.GRU(R) 
        out= out.to(self.device)
        hidden= hidden.to(self.device)
        R = out[:, -1, :] 
        R = self.dropout(R)
        
        # Skip rnn layers 
        # Take the output of the conv. C and pass it to recurrent skip
        shrinked_time_steps = C.size(2)
        for i in range(len(self.skip)):
            skip_step = self.skip[i]
            skip_sequence_len = shrinked_time_steps // skip_step
            
            S = C[:, :, -skip_sequence_len*skip_step:] 
            S = S.view(S.size(0), S.size(1), skip_sequence_len, skip_step) 
            
            S = S.permute(0, 3, 2, 1).contiguous() 
            S = S.view(S.size(0)*S.size(1), S.size(2), S.size(3))

            out, hidden = self.skip_GRU[i](S) 
            S = out[:, -1, :] # [batch_size*num_skip_components, skip_reccs_out_channels[i]]
            S = S.view(batch_size, skip_step*S.size(1)) # [batch_size, num_skip_components*skip_reccs_out_channels[i]]
            S = self.dropout(S)
            R = torch.cat((R, S), 1) # [batch_size, recc_out_channels + skip_reccs_out_channels * num_skip_components]
            
        
        # Output Layer
        O = F.relu(self.output(R)) # [batch_size, output_out_features=1]
        
        if self.ar_window  > 0:
            # set dim3 based on output_out_features
            AR = X[:, -self.ar_window:, self.ar_features] # [batch_size, ar_window_size, output_out_features=1]
            AR = AR.permute(0, 2, 1).contiguous() # [batch_size, output_out_features, ar_window_size]
            AR = self.ar(AR) # [batch_size, output_out_features, 1]
            AR = AR.squeeze(2) # [batch_size, output_out_features]
            O = O + AR
        
        return O