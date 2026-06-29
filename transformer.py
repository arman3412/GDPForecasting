import torch
import torch.nn as nn



class Transformer(nn.Module):
    def __init__(self, input_dim, layer_dim, output_dim, num_countries, embedding_dim=4, nhead=2, dim_feedforward=28):
        super(Transformer, self).__init__()

        self.layer_dim = layer_dim
        self.countryEmbedding = nn.Embedding(num_countries, embedding_dim)
        self.positionEmbedding = nn.Embedding(8, input_dim + embedding_dim)
        self.mask = nn.Transformer.generate_square_subsequent_mask(8)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=input_dim + embedding_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=layer_dim)
        self.fc = nn.Linear(input_dim + embedding_dim, output_dim)
        

    def forward(self, x, country_idx):
        countries_vec = self.countryEmbedding(country_idx)
        countries_with_timestep = countries_vec.unsqueeze(1).repeat(1, x.size(1), 1)

        positions_vec = self.positionEmbedding(torch.arange(x.size(1)))
        

        x = torch.cat((x, countries_with_timestep), dim=2)
        x = x + positions_vec.unsqueeze(0)  # Add positional encoding
    
        out = self.transformer_encoder(x, mask=self.mask)
        out = out[:, -1, :] 
        out = self.fc(out)
        return out