import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Gumbel
from utils.data import flatten_padded_sequences
from collections import namedtuple
import numpy as np
from progress.bar import Bar
from PoPMAG_RNN .config import device
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PoPMAG_RNN(nn.Module):
    def __init__(self, init_dim, event_dim, hidden_dim,
                 rnn_layers=2, dropout=0.5):
        super().__init__()

        self.event_dim = event_dim
        self.init_dim = init_dim
        self.hidden_dim = hidden_dim
        self.rnn_layers = rnn_layers
        self.output_dim = event_dim

        self.primary_event = self.event_dim - 1
        self.inithid_fc = nn.Linear(init_dim, rnn_layers * hidden_dim)
        self.inithid_fc_activation = nn.Tanh()

        self.event_embedding = nn.Embedding(event_dim, event_dim)

        self.encoder = nn.GRU(self.event_dim, self.hidden_dim,
                          num_layers=rnn_layers, dropout=dropout)

        self.decoder = nn.GRU(self.event_dim, self.hidden_dim,
                              num_layers=rnn_layers, dropout=dropout)
        #self.output_fc = nn.Linear(hidden_dim * rnn_layers, self.output_dim)
        self.output_fc = [nn.Linear(hidden_dim, self.output_dim) for i in range(3)]

        self.output_fc_activation = nn.Softmax(dim=-1)

    def forward(self, event, hidden=None):
        # One step forward
        # event  [1, batch*beam0]
        # hidden [grus, batch*beam, hid]
        assert len(event.shape) == 2
        assert event.shape[0] == 1
        batch_size = event.shape[1]
        # print()
        input = self.event_embedding(event) #[1, batch*beam0, dim]
        # print(f'input.shape={input.shape}')
        _, hidden = self.rnn(input, hidden)
        output = hidden.permute(1, 0, 2).contiguous()
        output = output.view(batch_size, -1).unsqueeze(0)
        # print(f'output.shape={output.shape}')
        output = self.output_fc(output)
        return output, hidden

    def gen_forward(self, event, hidden=None):
        # One step forward
        assert len(event.shape) == 2
        assert event.shape[0] == 1
        batch_size = event.shape[1]
        input = self.event_embedding(event)
        output, hidden = self.rnn(input, hidden)
        output = output.permute(1, 0, 2).contiguous()
        output = output.view(batch_size, -1).unsqueeze(0)
        output = self.output_fc(output)
        return output, hidden

    def SeqForward(self, events, hidden=None, lengths=None):
        batch_size = events.shape[1]
        event = self.get_primary_event(batch_size)
        # print(f'event.shape={event.shape}')
        one, hidden = self.gen_forward(event, hidden)
        # print(f'one.shape={one.shape}')
        # One step forward
        input = self.event_embedding(events)# (step, batch_size, dim)
        if lengths is not None:
            input = nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=True)
        # packed_output, self.state = self.encoder(embedding_packed, state)  # output, (h, c)
        output, _ = self.rnn(input, hidden)#(step, batch, dim)

        if lengths is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        # output = flatten_padded_sequences(output.permute(1,0,2), lengths)
        # print(f'output.shape={output.shape}')
        # output = hidden.permute(1, 0, 2).contiguous()#(seqlen, batch, dim)
        # output = output.view(batch_size, -1).unsqueeze(0)
        output = self.output_fc(output)#(tot_len, dim)
        return  torch.cat((one,output) , 0)


    def get_primary_event(self, batch_size):
        return torch.LongTensor([[self.primary_event] * batch_size]).to(device)

    def _sample_event(self, output, greedy=True, temperature=1.0):
        if greedy:
            return output.argmax(-1)
        else:
            output = output / temperature
            probs = self.output_fc_activation(output)
            return Categorical(probs).sample()

    def init_to_hidden(self, init):
        # [batch_size, init_dim]
        batch_size = init.shape[0]
        out = self.inithid_fc(init)
        out = self.inithid_fc_activation(out)
        out = out.view(self.rnn_layers, batch_size, self.hidden_dim)
        return out

    # model.generate(init, window_size, events=events[:-1],
    #                              teacher_forcing_ratio=teacher_forcing_ratio, output_type='logit')

    def Train(self, init, events, lengths=None):
        # init [batch_size, init_dim]
        # events [steps, batch_size] indeces
        hidden = self.init_to_hidden(init)

        output = self.SeqForward(events, hidden, lengths) #forward one step

        return output


    def generate(self, init, steps, events=None, greedy=1.0,
                 temperature=1.0, teacher_forcing_ratio=1.0, output_type='index', verbose=False):
        # init [batch_size, init_dim]
        # events [steps, batch_size] indeces
        # controls [1 or steps, batch_size, control_dim]

        batch_size = init.shape[0]
        assert init.shape[1] == self.init_dim
        assert steps > 0

        use_teacher_forcing = events is not None
        if use_teacher_forcing:
            assert len(events.shape) == 2
            assert events.shape[0] >= steps - 1
            events = events[:steps - 1]

        event = self.get_primary_event(batch_size)

        hidden = self.init_to_hidden(init)

        outputs = []
        step_iter = range(steps)
        if verbose:
            step_iter = Bar('Generating').iter(step_iter)

        for step in step_iter:
            output, hidden = self.gen_forward(event, hidden) #forward one step

            use_greedy = np.random.random() < greedy
            event = self._sample_event(output, greedy=use_greedy,
                                       temperature=temperature)

            if output_type == 'index':
                outputs.append(event)
            elif output_type == 'softmax':
                outputs.append(self.output_fc_activation(output))
            elif output_type == 'logit':
                outputs.append(output)
            else:
                assert False

            if use_teacher_forcing and step < steps - 1:  # avoid last one
                if np.random.random() <= teacher_forcing_ratio:
                    event = events[step].unsqueeze(0)

        return torch.cat(outputs, 0)
