import os
import torch
import itertools, os
import numpy as np
from progress.bar import Bar
import utils.shared as utils
import pickle
import sys
sys.path.append('/data2/qt/MusicGeneration/mg/model/')

from utils.MuMIDI import MuMIDI_EventSeq
from utils.shared import find_files_by_extensions

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


class Melody_Arrangement_Dataset:
    def __init__(self, root=None, paths=None, limlen=0, verbose=False):
        if root is None:
            return
        assert os.path.isdir(root), root

        # print(paths)
        self.root = root
        self.melody_seqs = []
        self.arrange_seqs = []
        if verbose:
            paths = Bar(root).iter(list(paths))
        for path in paths:
            eventseq = torch.load(path)
            # print()
            # print(eventseq)
            if MuMIDI_EventSeq.filter_melody(eventseq['melody']):
            #if len(eventseq['melody']) >= limlen and len(eventseq['arrangement']) >= limlen:
                self.melody_seqs.append(eventseq['melody'])
                self.arrange_seqs.append(eventseq['arrangement'])
        self.avg_melody_len = np.mean([len(item) for item in self.melody_seqs])
        self.avg_arrange_len = np.mean([len(item) for item in self.arrange_seqs])

    @staticmethod
    def save_file(obj, path):

        with open(path, 'wb') as output_hal:
            sp = pickle.dumps(obj)
            output_hal.write(sp)
        print(f'save dataset to: {path}')

    @staticmethod
    def load_file(path):
        with open(path,'rb') as file:
            obj = pickle.loads(file.read())
        print(f'load dataset from: {path}')
        return obj

    def __getitem__(self, index):
        return self.melody_seqs[index], self.arrange_seqs[index]

    def __len__(self):
        return len(self.melody_seqs)

    def count_bar(self):
        seq = []
        for item in self.melody_seqs:
            seq.append(MuMIDI_EventSeq.count_bar(item))
        return seq

    def SegBatchify(self, data):
        eventseq_batch = []
        #labels = []
        for melody_seq, arrange_seq in data:
            melody_seq_bar = segmentation(melody_seq)
            arrange_seq_bar = segmentation(arrange_seq)
            eventseq_batch.append((melody_seq_bar,arrange_seq_bar))
            #labels.append(eventseq[1:])
            # print(eventseq_batch[-1][:10],labels[-1][:10])

        return np.stack(eventseq_batch, axis=1)#, np.stack(labels,axis=0)
        # return np.stack(eventseq_batch, axis=0), np.stack(labels,axis=0)


    def __repr__(self):
        return (f'Dataset(root="{self.root}", '
                f'melody_seq={len(self.melody_seqs)}, '
                f'arrange_seq={len(self.arrange_seqs)}, '
                f'avg_melody_len={self.avg_melody_len}, '
                f'avg_arrange_len={self.avg_arrange_len})')


if __name__ == '__main__':
    pp = '/data2/qt/MusicGeneration/egs/dataset/lmd_matched_split/'
    pt = '/data2/qt/MusicGeneration/egs/dataset/lmd_matched_split/train.pth'
    pv = '/data2/qt/MusicGeneration/egs/dataset/lmd_matched_split/vaild.pth'
    paths = list(find_files_by_extensions(pp, ['.data']))

    # dataset = Melody_Arrangement_Dataset(pp, paths=paths[:-200], verbose=True)
    # Melody_Arrangement_Dataset.save_file(dataset, pt)
    re_dataset = Melody_Arrangement_Dataset()
    re_dataset = Melody_Arrangement_Dataset.load_file(pv)
    seq = re_dataset.count_bar()
    print(max(seq))
    print(seq)

    # dataset = Melody_Arrangement_Dataset(pp, paths=paths[-200:], verbose=True)
    # Melody_Arrangement_Dataset.save_file(dataset, pv)
    # print(dataset.melody_seqs)
    # print(dataset.__repr__())
    # re_dataset = Melody_Arrangement_Dataset.load_file(pv)
    # print(re_dataset.melody_seqs)
    # print(re_dataset.__repr__())
