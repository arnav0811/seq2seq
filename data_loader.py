import torch
import os
from torch.utils.data import Dataset
import pickle
import json
from collections import Counter

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
