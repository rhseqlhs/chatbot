import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import weight_norm


class RNN(nn.Module):
    """
    Base RNN class
    """

    def __init__(self, input_size, hidden_size, nlayers, embed_dim,
                 rnn_type, pad_idx, use_cuda, dropout, bidirect):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.embed_dim = embed_dim
        self.ndirect = 2 if bidirect else 1

        self.embedding = nn.Embedding(input_size, embed_dim, padding_idx=pad_idx)
        if rnn_type in ['GRU', 'LSTM']:
            self.rnn = getattr(nn, rnn_type)(embed_dim,
                                             hidden_size // self.ndirect,
                                             num_layers=nlayers,
                                             batch_first=True, dropout=dropout,
                                             bidirectional=bidirect)
            if use_cuda:
                self.rnn.cuda()  # turn on cuda before applying weight_norm
        else:
            raise ValueError("Please choose rnn type from: GRU or LSTM")
        self.rnn_type = rnn_type

    def forward(self, input):
        """
        Override default forward function in torch.nn.Module
        """
        pass

    def init_hidden(self, batch_size):
        # Get Tensor type from first parameter of model (e.g. cuda.FloatTensor)
        # to see if we should initialize cuda tensor or not
        weight = next(self.parameters()).data
        h_0 = Variable(weight.new(self.nlayers * self.ndirect,
                                  batch_size,
                                  self.hidden_size // self.ndirect).zero_(),
                       requires_grad=False)
        if self.rnn_type == 'LSTM':
            return (h_0,
                    Variable(weight.new(self.nlayers * self.ndirect,
                                        batch_size,
                                        self.hidden_size // self.ndirect).zero_(),
                             requires_grad=False))
        else:
            return h_0

    def init_weights(self):
        """
        Initialize weights, including internal weights of RNN. From:
        gist.github.com/thomwolf/eea8989cab5ac49919df95f6f1309d80
        Apply weight normalization to internal weights of RNN.
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            nn.init.xavier_uniform(t)
        for t in hh:
            nn.init.orthogonal(t)
        for t in b:
            nn.init.constant(t, 0)
        self.embedding.weight.data.uniform_(-0.05, 0.05)
        # Apply Weight Normalization
        l = [name for name, _ in list(self.rnn.named_parameters()) if 'weight' in name]
        for name in l:
            weight_norm(self.rnn, name)

    def is_cuda(self):
        """
        Return boolean value of whether model is cuda enabled.
        """
        param_type = str(type(next(self.parameters()).data))
        return 'cuda' in param_type


class DecoderRNN(RNN):
    """
    Basic Decoder without attentional mechanism
    """

    def __init__(self, input_size, hidden_size, nlayers, embed_dim,
                 rnn_type, pad_idx, use_cuda, dropout, bidirect=False):
        super().__init__(input_size, hidden_size, nlayers, embed_dim,
                         rnn_type, pad_idx, use_cuda, dropout, False)  # unidirectional
        self.linear = nn.Linear(hidden_size, input_size)
        self.softmax = nn.LogSoftmax()
        self.init_weights()

    def init_weights(self):
        super().init_weights()
        self.linear.weight.data.uniform_(-0.05, 0.05)

    def forward(self, input, hidden):
        batch_size = input.size()[0]
        embedded = self.embedding(input).unsqueeze(1)
        output, hidden = self.rnn(embedded, hidden)
        output = self.linear(output[:, 0, :])
        output = self.softmax(output)
        return output, hidden
