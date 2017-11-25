# Chatbot
Included are supervised and unsupervised sentence generation models.

## Seq2Seq Chatbot
The biggest component of the program is a Sequence to Sequence model for generating responses to user inputs.
Supported models include GRU or LSTM encoder, decoder as well as a decoder with a global attention mechanism.
The Jupyter Notebook chatbot.ipynb goes through the data downloading, generation, and processing, as well as the model definitions and training. An interface for chatting with model is provided via ipywidgets.
The implementation include best practices from Neural Machine Translation such as input and output embedding weight tying, unknown token replacement, and a bidirectional encoder.

## Variational Autoencoder for Sentence Generation
The second part of chatbot.ipynb is a Variational Autoencoder (VAE) for unsupervised sentence generation. The encoder and decoder for the VAE are RNNs, following Bowman et al. Generating Sentences from a Continous Space. We also include word dropout in the decoder, and KL Divergence Annealing as in the aforementioned paper. The Variational Autoencoder is implemented with the Pyro Deep Probabilistic Programming Language, and follows the VAE implementation provided in the tutorial for Pyro very closely.

## Requirements
```
Python 3
Jupyter Notebook
PyTorch 0.2
PyTorch Text
Pyro-ppl
NumPy
Matplotlib
tqdm
```