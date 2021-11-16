class LSTM(nn.Module):
    def __init__(self, input_size=20, hidden_layer_size=800, output_size=20):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear1 = nn.Linear(hidden_layer_size, math.ceil(1.25*hidden_layer_size))
        self.linear2 = nn.Linear(math.ceil(1.25*hidden_layer_size), output_size)
        
        self.dropout = nn.Dropout(p=0.2)
        
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        drop_out = self.dropout(lstm_out)
        predictions = self.linear1(drop_out.view(len(input_seq), -1))
        #predictions = self.dropout(predictions)
        predictions = self.linear2(predictions)
        return predictions[-1]