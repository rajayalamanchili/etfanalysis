import torch
import torch.nn as nn
import torch.optim as optim

class SimpleSequenceLSTM(nn.Module):
    
    def __init__(self, numInputFeatures, numberHiddenDimensions, sequenceLength, numberLayers=2):
        super(SimpleSequenceLSTM, self).__init__()
        
        self.numberHiddenDimensions = numberHiddenDimensions
        self.numberLayers = numberLayers
        self.sequenceLength = sequenceLength
        

        self.hiddenState = (torch.zeros(numberLayers, sequenceLength, numberHiddenDimensions),
                           torch.zeros(numberLayers, sequenceLength, numberHiddenDimensions))        

        self.lstm = nn.LSTM(numInputFeatures, 
                            numberHiddenDimensions, 
                            numberLayers, 
                            batch_first=True,
                            dropout=0.5)
                
        self.fc = nn.Linear(in_features=numberHiddenDimensions, out_features=1)
    
    def forward(self, x):
        
        output, hiddenState = self.lstm(x.view(len(x), self.sequenceLength, -1))
        #output = output.view(self.sequenceLength, len(x), self.numberHiddenDimensions)[-1]
        output = self.fc(output[:,-1,:])
        
        return output