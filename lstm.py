import torch
import torch.nn as nn





class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, num_countries, embedding_dim=4):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim + embedding_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.embedding = nn.Embedding(num_countries, embedding_dim)

    def forward(self, x, country_idx, h0=None, c0=None):
        countries_vec = self.embedding(country_idx)
        countries_with_timestep = countries_vec.unsqueeze(1).repeat(1, x.size(1), 1)
        x = torch.cat((x, countries_with_timestep), dim=2)
        if h0 is None:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        if c0 is None:
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out