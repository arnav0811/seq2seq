import torch
import torch.nn as nn
import json
from collections import Counter
import numpy as np
from torch.utils.data import Dataset
import pickle

class ConalaDataset(Dataset):
    def __init__(self, data_path, vocab_path=None, max_length=100):
        self.data = self.load_data(data_path, max_length)
        self.vocab = self.build_vocab()
    
    def load_data(self, data_path, max_length):
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # Filter out extremely long snippets
        filtered = []
        for i in data:
            if len(i['intent'].split()) <= max_length and len(i['snippet'].split()) <= max_length:
                filtered.append(i)
        print(f"Loaded {len(filtered)}")
        return filtered

    def build_vocab(self):
        all_tokens = []
        for i in self.data:
            all_tokens.extend(i['intent'].split())
            all_tokens.extend(i['snippet'].split())
        
        # Tokens - padding, start, end, unknown
        vocab = {
            '<PAD>': 0,
            '<SOS>': 1,
            '<EOS>': 2,
            '<UNK>': 3,
        }

        token_count = Counter(all_tokens)
        for token, _ in token_count.most_common(8000):
            if token not in vocab:
                vocab[token] = len(vocab)
        print(f"Vocab Size: {len(vocab)}")
        return vocab
    
    def text_to_seq(self, text):
        tokens = text.split()
        # string to list of token indices
        return [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        i = self.data[index]
        intent_seq = self.text_to_seq(i['intent']) + [self.vocab['<EOS>']]
        snippet_seq = [self.vocab['<SOS>']] + self.text_to_seq(i['snippet']) + [self.vocab['<EOS>']]

        return {
            'intent': torch.tensor(intent_seq),
            'snippet': torch.tensor(snippet_seq),
        }

def collate(batch):
    # padding for batching
    intents = [i['intent'] for i in batch]
    snippets = [i['snippet'] for i in batch]
    intent_lengths = [len(i) for i in intents]
    snippet_lengths = [len(i) for i in snippets]

    max_intent_lenght = max(intent_lengths)
    max_snippet_length = max(snippet_lengths)

    intent_padded = torch.zeros(len(batch), max_intent_lenght, dtype=torch.long)
    snippet_padded = torch.zeros(len(batch), max_snippet_length, dtype=torch.long)

    for i, (intent, snippet) in enumerate(zip(intents, snippets)):
        intent_padded[i, :len(intent)] = intent
        snippet_padded[i, :len(snippet)] = snippet
    
    return {
        'intents': intent_padded,
        'snippets': snippet_padded,
        'intent_lengths': torch.tensor(intent_lengths),
        'snippet_lengths': torch.tensor(snippet_lengths),
    }

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
        self.v = nn.Linear(decocder_hidden_dim, 1, bias = False)

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
        decoder_projection = self.W_decoder(decode_hidden)
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
        
