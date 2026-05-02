import torch
import torch.nn as nn
from utils import NUMBER_OF_JOINTS

N_LAYERS = 4
VECTOR_SIZE  = 128
N_HEADS  = 4
DROPOUT  = 0.1
        
class PoseTransformer(nn.Module):

    def __init__(self, n_layers, vector_size, n_heads, dropout):
        super().__init__()
        self.d_model = vector_size
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.encoding = self.SinPosEnc()

        #embedding
        self.input_layer = nn.Linear(2, vector_size)

        #encoding pos
        self.encoding = self.SinPosEnc()

        # enc layer/block
        encoder_layer = nn.TransformerEncoderLayer(d_model= vector_size, nhead = n_heads, dropout= dropout, batch_first= True, dim_feedforward = 4*vector_size)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers= n_layers)
        
        #head
        self.output_layer = nn.Linear(vector_size, 3)
    

    def SinPosEnc(self):
        encoding = torch.zeros(NUMBER_OF_JOINTS, self.d_model)
        for pos in range(NUMBER_OF_JOINTS):
            even = torch.arange(0, self.d_model, 2)

            encoding[pos, even] = torch.sin(pos/(10000**(2*even/self.d_model)))
            encoding[pos, even + 1 ] = torch.cos(pos/(10000**(2*even/self.d_model)))

        return encoding

    def forward(self, x):

        x = self.input_layer(x)
        x = x + self.encoding
        x = self.transformer_encoder(x)
        x = self.output_layer(x)

        return x

if __name__ == "__main__":
    batch = 32
    model = PoseTransformer(N_LAYERS, VECTOR_SIZE, N_HEADS, DROPOUT)
    print(model)
    dummy_input = torch.rand(32, NUMBER_OF_JOINTS, 2)
    output = model(dummy_input)
    print(output.shape)