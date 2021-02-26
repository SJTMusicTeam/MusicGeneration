import os
import torch
import itertools, os
import numpy as np
from progress.bar import Bar
import utils.shared as utils
from utils.sequence import EventSeq, ControlSeq

def flatten_padded_sequences(outs, lengths):
    batch, mx_length, vocab_size = outs.shape
    if lengths is None:
        return outs.contiguous().view(-1, vocab_size)
    res = []
    for i in range(batch):
        res.append(outs[i, :lengths[i] - 1, :].squeeze(0))
    return torch.cat(res, 0)

def SeqBatchify(inputs):
    #print(inputs)
    inputs = sorted(inputs, key=lambda i: len(i), reverse=True)
    lengths = np.array([len(item) for item in inputs])
    mx_length = np.max(lengths)
    X = np.zeros((len(inputs), mx_length),dtype=np.int16)
    for i in range(len(inputs)):
        mx = lengths[i]
        X[i, :mx] = np.array(inputs[i])
    labels = []
    for i in range(len(inputs)):
        labels.append( np.array(X[i])[1:lengths[i]] )
    Y = np.concatenate(labels)
    return X ,Y, lengths


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, seqs):
        self.seqs = seqs

    def __getitem__(self, index):
        return self.seqs[index]

    def __len__(self):
        return len(self.seqs)



class Event_Dataset:
    def __init__(self, root, limlen=None, verbose=False):
        assert os.path.isdir(root), root
        paths = utils.find_files_by_extensions(root, ['.data'])
        # print(paths)
        self.root = root
        self.samples = []
        self.seqlens = []
        if verbose:
            paths = Bar(root).iter(list(paths))
        for path in paths:
            eventseq = torch.load(path)
            if len(eventseq) >= limlen:
                self.samples.append(eventseq)
                self.seqlens.append(len(eventseq))
        self.avglen = np.mean(self.seqlens)

    def count(self, v):
        a = sorted(self.seqlens, reverse=False)
        #print(a[:100])
        x = np.searchsorted(a, v, side='left', sorter=None)
        print(f'{x}/{len(a)}')
        print(f'the ratio of length of events less than {v} is {100*x/len(a)}%')
        return 100*x / len(a)

    def batches(self, batch_size, window_size, stride_size):
        indeces = [(i, (j, j + window_size) )
                   for i, seqlen in enumerate(self.seqlens)
                   for j in range(0, seqlen - window_size, stride_size)]
        return indeces
        #print(len(indeces)) #2,452,443
        # idx = np.random.permutation(len(indeces))
        # eventseq_batch = []
        # for k in range(len(indeces)//batch_size):
        #     for ii in range(k*batch_size, k*batch_size+batch_size):
        #         i, (start, end) = indeces[idx[ii]]
        #         eventseq = self.samples[i]
        #         eventseq = eventseq[start:end]
        #         eventseq_batch.append(eventseq)
        #     yield np.stack(eventseq_batch, axis=1)
        #     eventseq_batch.clear()
        # while True:
        #     eventseq_batch = []
        #     n = 0
        #     for ii in np.random.permutation(len(indeces)):
        #         i, (start, end) = indeces[ii]
        #         eventseq = self.samples[i]
        #         eventseq = eventseq[start:end]
        #         eventseq_batch.append(eventseq)
        #         n += 1
        #         if n == batch_size:
        #             yield np.stack(eventseq_batch, axis=1)
        #             eventseq_batch.clear()
        #             n = 0

    def SegBatchify(self,data):
        eventseq_batch = []
        #labels = []
        for i, (start, end) in data:
            eventseq = self.samples[i]
            eventseq = eventseq[start:end]
            eventseq_batch.append(eventseq)
            #labels.append(eventseq[1:])
            # print(eventseq_batch[-1][:10],labels[-1][:10])

        return np.stack(eventseq_batch, axis=1)#, np.stack(labels,axis=0)
        # return np.stack(eventseq_batch, axis=0), np.stack(labels,axis=0)

    def Batchify(self,data):
        eventseq_batch = []
        for i, (start, end) in data:
            eventseq = self.samples[i]
            eventseq = eventseq[start:end]
            eventseq_batch.append(eventseq)
        return np.stack(eventseq_batch, axis=1)

    def __repr__(self):
        return (f'Dataset(root="{self.root}", '
                f'samples={len(self.samples)}, '
                f'avglen={self.avglen})')


class Event_Control_Dataset:
    def __init__(self, root, verbose=False):
        assert os.path.isdir(root), root
        paths = utils.find_files_by_extensions(root, ['.data'])
        self.root = root
        self.samples = []
        self.seqlens = []
        if verbose:
            paths = Bar(root).iter(list(paths))
        for path in paths:
            eventseq, controlseq = torch.load(path)
            controlseq = ControlSeq.recover_compressed_array(controlseq)
            assert len(eventseq) == len(controlseq)
            self.samples.append((eventseq, controlseq))
            self.seqlens.append(len(eventseq))
        self.avglen = np.mean(self.seqlens)

    def batches(self, batch_size, window_size, stride_size):
        indeces = [(i, range(j, j + window_size))
                   for i, seqlen in enumerate(self.seqlens)
                   for j in range(0, seqlen - window_size, stride_size)]
        while True:
            eventseq_batch = []
            controlseq_batch = []
            n = 0
            for ii in np.random.permutation(len(indeces)):
                i, r = indeces[ii]
                eventseq, controlseq = self.samples[i]
                eventseq = eventseq[r.start:r.stop]
                controlseq = controlseq[r.start:r.stop]
                eventseq_batch.append(eventseq)
                controlseq_batch.append(controlseq)
                n += 1
                if n == batch_size:
                    yield (np.stack(eventseq_batch, axis=1),
                           np.stack(controlseq_batch, axis=1))
                    eventseq_batch.clear()
                    controlseq_batch.clear()
                    n = 0

    def __repr__(self):
        return (f'Dataset(root="{self.root}", '
                f'samples={len(self.samples)}, '
                f'avglen={self.avglen})')

