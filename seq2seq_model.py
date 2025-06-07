import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Embedding for input tokenas
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # Bahdanau Attention implementation - Bidirectional LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)

    def forward(self, x, lengths):
        embedded = self.embedding(x)

        # Pack padded sequence for LSTM - Bahdanau
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        batch_size = x.size(0)
        # Reshaping hidden
        hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_dim)
        hidden = torch.cat((hidden[:, 0, :, :], hidden[:, 1, :, :]), dim=2)

        cell = cell.view(self.num_layers, 2, batch_size, self.hidden_dim)
        cell = torch.cat((cell[:, 0, :, :], cell[:, 1, :, :]), dim=2)

        return output, (hidden, cell)

class Attention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim - decoder_hidden_dim

        self.W_encoder = nn.Linear(encoder_hidden_dim, decoder_hidden_dim, bias = False)
        self.W_decoder = nn.Linear(decoder_hidden_dim, decoder_hidden_dim, bias = False)
        self.v = nn.Linear(decoder_hidden_dim, 1, bias = False)

        self.softmax = nn.Softmax(dim = 1)

    def forward(self, encoder_outputs, decoder_hidden, encoder_lengths):
        batch_size = encoder_outputs.size(0)
        sequence_length = encoder_outputs.size(1)
        
        # For every time step, repeat decoder hidden layer
        # print(decoder_hidden.shape)
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, sequence_length, 1)
        # print(decoder_hidden.shape)

        # Additive attention - v*tanh(Wencoder + Wdecoder)
        encoder_projection = self.W_encoder(encoder_outputs)
        decoder_projection = self.W_decoder(decoder_hidden)
        # shape - [batch_size, N] 
        energies = self.v(torch.tanh(encoder_projection + decoder_projection)).squeeze(2)
        # print(energies.shape)

        # Masking <PAD> tokens prior to softmax - 0 attention weight
        mask = torch.arange(sequence_length).unsqueeze(0).repeat(batch_size, 1).to(encoder_outputs.device)
        mask = mask < encoder_lengths.unsqueeze(1)
        energies = energies.masked_fill(~mask, -float('inf')) 
        attention_weights = self.softmax(energies)

        # print(encoder_outputs.shape)
        # print(attention_weights.shape)
        
        # weighted sum
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context_vector, attention_weights

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, encoder_hidden_dim, num_layers = 2, use_attention = False):
        super().__init__()
        self.vocab_size = vocab_size
        self.use_attention = use_attention
        self.embedding_dim = nn.Embedding(vocab_size, embedding_dim, padding_idx = 0)

        # IN Bahdanau attention input to LSTM is context vector + embedding
        lstm_input_dim = embedding_dim
        # Attention flag
        if use_attention:
            lstm_input_dim += encoder_hidden_dim
            self.attention = Attention(encoder_hidden_dim, hidden_dim)
        # print(lstm_input_dim.shape)
        self.lstm = nn.LSTM(lstm_input_dim, hidden_dim, num_layers, batch_first = True)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_token, hidden, encoder_outputs = None, encoder_lengths = None):
        embedded = self.embedding(input_token)

        if self.use_attention and encoder_outputs is not None:
            # Last layer of hidden state of LSTM
            context_vector, attention_weights = self.attention(encoder_outputs, hidden[0][-1], encoder_lengths)
            lstm_input = torch.cat([embedded, context_vector.unsqueeze(1)], dim = 2)
        else:
            lstm_input = embedded
            attention_weight = None
        output, hidden = self.lstm(lstm_input, hidden)
        output = self.output_projection(output)

        return output, hidden, attention_weights

class Seq2SeqModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim = 256, hidden_dim = 512, num_layers = 2, use_attention = False):
        super().__init__()
        self.use_attention = use_attention
        # Bi directional LSTM in Bahdanau Attention paper
        encoder_hidden_dim = hidden_dim * 2
        self.encoder = Encoder(vocab_size, embedding_dim, hidden_dim, num_layers)
        self.decoder = Decoder(vocab_size, embedding_dim, hidden_dim, num_layers, use_attention)

    def forward(self, source, target, source_lengths, target_lengths):
        batch_size, target_len = target.size()
        vocab_size = self.decoder.vocab_size

        encoder_outputs, encoder_hidden = self.encoder(source, source_lengths)
        
        # First deocder hidden state is encoder hidden state
        decoder_hidden = encoder_hidden

        outputs = torch.zeros(batch_size, target_len, vocab_size).to(source.device)
        # Accounting for <SOS> tag
        decoder_input = target[:, 0].unsqueeze(1)

        for i in range(1, target_len):
            output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs, source_lengths)
            outputs[:, i, :] = output.squeeze(1)
            # Using ground truth next workd from dataset instead of argmax of outputs
            decoder_input = target[:, i].unsqueeze(1)

        return outputs
        
            
        
        
