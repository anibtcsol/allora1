import torch
import torch.nn as nn
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler

# Define the Split BiRNN model with LSTM cells
class SplitBiRNNModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, num_layers, dropout):
        super(SplitBiRNNModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.forward_lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.backward_lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size * 2, output_size)

    def forward(self, input_seq):
        h_0_forward = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size)
        c_0_forward = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size)
        h_0_backward = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size)
        c_0_backward = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size)
        forward_lstm_out, _ = self.forward_lstm(input_seq, (h_0_forward, c_0_forward))
        backward_lstm_out, _ = self.backward_lstm(torch.flip(input_seq, [1]), (h_0_backward, c_0_backward))
        backward_lstm_out = torch.flip(backward_lstm_out, [1])
        lstm_out = torch.cat((forward_lstm_out, backward_lstm_out), dim=2)
        predictions = self.linear(lstm_out[:, -1])
        return predictions

# Define the Attention Mechanism
class AttentionMechanism(nn.Module):
    def __init__(self, hidden_layer_size):
        super(AttentionMechanism, self).__init__()
        self.linear = nn.Linear(hidden_layer_size, hidden_layer_size)

    def forward(self, lstm_out):
        attention_weights = torch.tanh(self.linear(lstm_out))
        attention_weights = attention_weights.sum(dim=2, keepdim=True)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context_vector = attention_weights * lstm_out
        context_vector = context_vector.sum(dim=1)
        return context_vector

# Define the model
class Model(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, num_layers, dropout, window_size):
        super(Model, self).__init__()
        self.split_birnn = SplitBiRNNModel(input_size, hidden_layer_size, hidden_layer_size, num_layers, dropout)
        self.attention_mechanism = AttentionMechanism(hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.window_size = window_size

    def forward(self, input_seq):
        lstm_out = self.split_birnn(input_seq)
        context_vector = self.attention_mechanism(lstm_out)
        predictions = self.linear(context_vector)
        moving_avg = self.moving_average(predictions, self.window_size)
        return moving_avg

    def moving_average(self, predictions, window_size):
        moving_avg = torch.zeros_like(predictions)
        for i in range(window_size, predictions.size(1)):
            moving_avg[:, i] = predictions[:, i - window_size:i].mean(dim=1)
        return moving_avg

# Load the dataset
df = pd.read_csv('your_data.csv')

# Preprocess the data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

# Create the dataset and data loader
dataset = torch.utils.data.TensorDataset(torch.tensor(df_scaled))
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the model, loss function, and optimizer
model = Model(input_size=1, hidden_layer_size=128, output_size=1, num_layers=2, dropout=0.2, window_size=10)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(100):
    for batch in data_loader:
        input_seq = batch[0].unsqueeze(1)
        optimizer.zero_grad()
        predictions = model(input_seq)
        loss = criterion(predictions, input_seq)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Evaluate the model
model.eval()
predictions = model(torch.tensor(df_scaled).unsqueeze(1))
mse = nn.MSELoss()(predictions, torch.tensor(df_scaled).unsqueeze(1))
print(f'MSE: {mse.item()}')
