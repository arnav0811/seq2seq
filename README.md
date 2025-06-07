# Sequence-to-Sequence with Bahdanau Attention
Implemented the [Sequence to Sequence](https://arxiv.org/pdf/1409.3215) paper using [Bahdanau Attention](https://arxiv.org/pdf/1409.0473). 
Trained a sequence-to-sequence model with and without attention to translate natural language to Python snippets (code generation) using the [CoNaLa](https://conala-corpus.github.io/) dataset..
The papers I drew inspiration from perform the WMT’14 English to French translation task. 

## Model Architecture
2-layer Encoder, 2-layer Decoder, Bahdanau Attention, Embedding Dimension - 256, Hidden Dimension - 256. 

## Training
Used SGD Optimizer as pointed out in the [Sequence to Sequence](https://arxiv.org/pdf/1409.3215) paper. Closely followed the training loop displayed in said paper with an initial learning rate of 0.7 and the learnign rate was halved at every half epoch after the 5th epoch. First trained the mdoel for 8 epochs, and then for 15 epochs both wth and without the Bahdanau Attention mechanism. Used Cross-Entropy loss and achieved small differences between the two training mechanisms (with and without Bahdanau Attention). 

## Setup
pip install -r requirements.txt
python train.py --data_path data/conala-train.json --use_attention
