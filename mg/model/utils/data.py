import os
import torch
import itertools, os
import numpy as np
from progress.bar import Bar
import utils.shared as utils
import pickle
import sys
sys.path.append('/data2/qt/MusicGeneration/mg/model/')
# from mg.model.PoPMAG_RNN.config import device
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
        with open(path, 'rb') as file:
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
        s = []
        t = []
        for melody_seq, arrange_seq in data:
            melody_seq_bar = MuMIDI_EventSeq.segmentation(melody_seq)
            arrange_seq_bar = MuMIDI_EventSeq.segmentation(arrange_seq)
            s.append(melody_seq_bar)
            t.append(arrange_seq_bar)
        return s, t


    @staticmethod
    def bar_id(n_bar):
        from mg.model.PoPMAG_RNN.config import model
        if n_bar < model['bar_dim']:
            return model['event_dim'] + n_bar
        return model['event_dim'] + model['bar_dim'] - 1
    @staticmethod
    def pos_id(num):
        from mg.model.utils.MuMIDI import MuMIDI_EventSeq as MU
        feat_range = MU.feat_ranges()
        return feat_range['position'][num]

    @staticmethod
    def bar():
        from mg.model.utils.MuMIDI import MuMIDI_EventSeq as MU
        feat_range = MU.feat_ranges()
        return feat_range['bar'][0]


    @staticmethod
    def event_dim():
        from mg.model.PoPMAG_RNN.config import model
        return model['event_dim'] + model['bar_dim']


    @staticmethod
    def get_mask(input, delta = 0):#place_hodler for embedding
        """
        :param input: batch of unprocessed sequence
        :return: processed matrix id for embedding [batch, mx_bar_num , mx_bar_len, 7]
        (bar_embed, pos_embed, tempo_cls, tempo_value, token1, token2, token3)
        """
        batch = len(input)
        batch_seqs = []
        mx_bar_num, mx_bar_len = 0, 0
        for bar_seqs in input:
            n_bar = 0
            one_bars = []
            for bar_items in bar_seqs:
                bar_embed = Melody_Arrangement_Dataset.bar_id(n_bar)
                n_bar += 1

                i = 0
                pos_embed = 0
                tempo_cls = 0
                tempo_val = 0
                pitch = 0
                duration = 0
                velocity = 0

                bar_seq = []

                while i < len(bar_items):
                    # item = np.zeros(7)
                    if MuMIDI_EventSeq.check('bar', bar_items[i]):
                        item = torch.LongTensor([Melody_Arrangement_Dataset.bar_id(n_bar), \
                                                 Melody_Arrangement_Dataset.pos_id(0),\
                                                 0, 0, Melody_Arrangement_Dataset.bar(), 0, 0])
                        bar_seq.append(item)
                        i += 1
                    elif MuMIDI_EventSeq.check('position', bar_items[i]):
                        n_pos = bar_items[i]
                        pos_embed = n_pos
                        i += 1
                        bar_seq.append(torch.LongTensor([bar_embed, pos_embed, 0, 0, 0, 0, 0]))
                    elif i < len(bar_items) and MuMIDI_EventSeq.check('tempo_class', bar_items[i]) \
                            and MuMIDI_EventSeq.check('tempo_value', bar_items[i + 1]):
                        tempo_cls = bar_items[i]
                        tempo_val = bar_items[i + 1]
                        i += 2
                        bar_seq.append(torch.LongTensor([bar_embed, pos_embed, tempo_cls, 0, 0, 0, 0]))
                        bar_seq.append(torch.LongTensor([bar_embed, pos_embed, 0, tempo_val, 0, 0, 0]))
                    elif i+2 < len(bar_items) and MuMIDI_EventSeq.check('note_velocity', bar_items[i]) \
                            and MuMIDI_EventSeq.check('note_on', bar_items[i + 1]) \
                            and MuMIDI_EventSeq.check('note_duration', bar_items[i + 2]):
                        velocity = bar_items[i]
                        pitch = bar_items[i + 1]
                        duration = bar_items[i + 2]
                        bar_seq.append(torch.LongTensor([bar_embed, pos_embed, tempo_cls, tempo_val, pitch, duration, velocity]))
                        i += 3
                    else:
                        pitch = bar_items[i]
                        i += 1
                        bar_seq.append(torch.LongTensor([bar_embed, pos_embed, tempo_cls, tempo_val, pitch, 0, 0]))
                    # (bar_embed, pos_embed, tempo_cls, tempo_value, token1, token2, token3)
                    # item[0] = bar_embed
                    # item[1] = pos_embed
                    # item[2] = tempo_cls
                    # item[3] = tempo_val
                    # item[4] = pitch
                    # item[5] = duration
                    # item[6] = velocity
                    # item = [bar_embed, pos_embed, tempo_cls, tempo_val, pitch, duration, velocity]
                    # # print(item)
                    # item = torch.LongTensor(item)
                    # bar_seq.append(item)

                if delta != 0:
                    bar_seq.pop(-1)
                mx_bar_len = max(mx_bar_len, len(bar_seq))
                bar_seq = torch.stack(bar_seq)
                one_bars.append(bar_seq)
                # one_bars = [bar_num(vary) * bar_len(vary) * embedding]
            mx_bar_num = max(mx_bar_num, len(one_bars))
            batch_seqs.append(one_bars)

        # mx_bar_num -= 1
        # mx_bar_len -= 1

        pad_data = torch.zeros((batch, mx_bar_num, mx_bar_len, 7))
        pad_data_len = torch.ones((batch, mx_bar_num))
        for batch_id in range(batch):
            one_bars = batch_seqs[batch_id]
            # print(f'len_one_bar={len(one_bars)}')
            for bar_num in range(len(one_bars)):
                # print(f'shape_bar_seq={one_bars[bar_num].shape}')
                bar_seq = one_bars[bar_num]
                pad_data[batch_id, bar_num, :len(bar_seq), :] = bar_seq
                pad_data_len[batch_id, bar_num] = len(bar_seq)
        # print(pad_data.shape)
        return pad_data, pad_data_len

    @staticmethod
    def label_mask(input):#place holder for label & label mask
        """
        :param input: batch of unprocessed sequence
        :return: processed matrix id for embedding [batch, mx_bar_num , mx_bar_len, 3]
        (type:)
        (duartion:)
        (velocity:)
        """
        feat_dim = MuMIDI_EventSeq.feat_dims()
        shift = [1 + feat_dim['note_on'] + feat_dim['note_duration'], 1, 1 + feat_dim['note_on'] ]
        bar_idx = MuMIDI_EventSeq.feat_ranges()['bar'][0]
        batch = len(input)
        batch_seqs = []
        batch_masks = []
        mx_bar_num, mx_bar_len = 0, 0
        for bar_seqs in input:
            n_bar = 0
            one_bars = []
            one_bars_masks = []
            for bar_items in bar_seqs:
                n_bar += 1
                i = 1
                bar_seq = []
                bar_seq_mask = []
                while i < len(bar_items):
                    # item = np.zeros(4)
                    # mask = np.zeros(4)
                    if MuMIDI_EventSeq.check('bar', bar_items[i]):
                        item  = torch.LongTensor([bar_idx - shift[0], 0, 0])
                        mask = torch.LongTensor([1, 0, 0])
                        bar_seq.append(item)
                        bar_seq_mask.append(mask)
                        i += 1
                    elif MuMIDI_EventSeq.check('position', bar_items[i]):
                        n_pos = bar_items[i]
                        pos_embed = n_pos
                        item = torch.LongTensor([pos_embed - shift[0], 0, 0])
                        mask = torch.LongTensor([1, 0, 0])
                        bar_seq.append(item)
                        bar_seq_mask.append(mask)
                        i += 1
                    elif i < len(bar_items) and MuMIDI_EventSeq.check('tempo_class', bar_items[i]) \
                            and MuMIDI_EventSeq.check('tempo_value', bar_items[i + 1]):
                        tempo_cls = bar_items[i] - shift[0]
                        tempo_val = bar_items[i + 1] - shift[0]
                        item = torch.LongTensor([tempo_cls, 0, 0])
                        mask = torch.LongTensor([1, 0, 0])
                        bar_seq.append(item)
                        bar_seq_mask.append(mask)

                        item = torch.LongTensor([tempo_val, 0, 0])
                        mask = torch.LongTensor([1, 0, 0])
                        bar_seq.append(item)
                        bar_seq_mask.append(mask)
                        i += 2
                    elif i+2 < len(bar_items) and MuMIDI_EventSeq.check('note_velocity', bar_items[i]) \
                            and MuMIDI_EventSeq.check('note_on', bar_items[i + 1]) \
                            and MuMIDI_EventSeq.check('note_duration', bar_items[i + 2]):
                        velocity = bar_items[i] - shift[0]
                        pitch = bar_items[i + 1] - shift[1]
                        duration = bar_items[i + 2] - shift[2]
                        item = torch.LongTensor([pitch, duration, velocity])
                        mask = torch.LongTensor([1, 1, 1])

                        bar_seq.append(item)
                        bar_seq_mask.append(mask)
                        i += 3
                    else:
                        pitch = bar_items[i] - shift[0]
                        item = torch.LongTensor([pitch, 0, 0])
                        mask = torch.LongTensor([1, 0, 0])
                        bar_seq.append(item)
                        bar_seq_mask.append(mask)
                        i += 1

                mx_bar_len = max(mx_bar_len, len(bar_seq))
                bar_seq = torch.stack(bar_seq)
                bar_seq_mask = torch.stack(bar_seq_mask)
                one_bars.append(bar_seq)
                one_bars_masks.append(bar_seq_mask)
                # one_bars = [bar_num(vary) * bar_len(vary) * embedding]
            mx_bar_num = max(mx_bar_num, len(one_bars))
            batch_seqs.append(one_bars)
            batch_masks.append(one_bars_masks)

        # mx_bar_len -= 1

        pad_data = torch.zeros((batch, mx_bar_num, mx_bar_len, 3))
        pad_data_mask = torch.zeros((batch, mx_bar_num, mx_bar_len, 3))
        # pad_data_len = torch.zeros((batch, mx_bar_num))
        for batch_id in range(batch):
            one_bars = batch_seqs[batch_id]
            one_bars_masks = batch_masks[batch_id]
            # print(f'len_one_bar={len(one_bars)}')
            for bar_num in range(len(one_bars)):
                # print(f'shape_bar_seq={one_bars[bar_num].shape}')
                bar_seq = one_bars[bar_num]
                bar_seq_mask = one_bars_masks[bar_num]
                pad_data[batch_id, bar_num, :len(bar_seq), :] = bar_seq
                pad_data_mask[batch_id, bar_num, :len(bar_seq), :] = bar_seq_mask
                # pad_data_len[batch_id, bar_num] = len(bar_seq)
        # print(pad_data.shape)
        return pad_data, pad_data_mask#, pad_data_len

    @staticmethod
    def get_tar_bar_mask(batch ,n_bar):#place_hodler for embedding
        """
        :param id: index of bar
        :return: bar mask for get bar embedding
        [batch , 1, 7]
        """
        pad_data = torch.zeros((batch, 7))
        for batch_id in range(batch):
            pad_data[batch_id, : ] = torch.LongTensor([Melody_Arrangement_Dataset.bar_id(n_bar), \
                                                            Melody_Arrangement_Dataset.pos_id(0), \
                                                            0, 0, Melody_Arrangement_Dataset.bar(), 0, 0])
        return pad_data.reshape(batch, 1, 7).long()

    @staticmethod
    def get_next_mask(batch, seq):
        # [batch , 1, 7]
        pad_data = torch.zeros((batch, 7))
        for i in range(batch):
            pad_data[i, : ] = torch.LongTensor(seq[i])
        return pad_data.reshape(batch, 1, 7).long()

    def FastBatchify(self, data):
        s = []
        t = []
        for melody_seq, arrange_seq in data:
            melody_seq_bar = MuMIDI_EventSeq.segmentation(melody_seq)
            arrange_seq_bar = MuMIDI_EventSeq.segmentation(arrange_seq)
            for i in range(len(arrange_seq_bar)):
                arrange_seq_bar[i] = np.append(arrange_seq_bar[i], MuMIDI_EventSeq.feat_ranges()['bar'][0])
            s.append(melody_seq_bar)
            t.append(arrange_seq_bar)
        src, src_mask = Melody_Arrangement_Dataset.get_mask(s, 0)
        tar, tar_mask = Melody_Arrangement_Dataset.get_mask(t, -1)
        label, label_mask = Melody_Arrangement_Dataset.label_mask(t)

        src = src.long()
        src_mask = src_mask.long()
        tar = tar.long()
        tar_mask = tar_mask.long()
        label = label.long()
        label_mask = label_mask.long()

        return src, src_mask, tar, tar_mask, label, label_mask
        # return s,t, torch.tensor(src, dtype=torch.long, device=device), \
        #        torch.tensor(src_mask, dtype=torch.long, device=device), \
        #        torch.tensor(tar, dtype=torch.long, device=device), \
        #        torch.tensor(tar_mask, dtype=torch.long, device=device), \
        #        torch.tensor(label, dtype=torch.long, device=device), \
        #        torch.tensor(label_mask, dtype=torch.long, device=device)


    def Batchify(self, data):
        s = []
        t = []
        for melody_seq, arrange_seq in data:
            melody_seq_bar = MuMIDI_EventSeq.segmentation(melody_seq)
            arrange_seq_bar = MuMIDI_EventSeq.segmentation(arrange_seq)
            s.append(melody_seq_bar)
            t.append(arrange_seq_bar)
        return s, t

    def __repr__(self):
        return (f'Dataset(root="{self.root}", '
                f'melody_seq={len(self.melody_seqs)}, '
                f'arrange_seq={len(self.arrange_seqs)}, '
                f'avg_melody_len={self.avg_melody_len}, '
                f'avg_arrange_len={self.avg_arrange_len})')


if __name__ == '__main__':
    pp = '/data2/qt/MusicGeneration/egs/dataset/lmd_matched_MuMIDI/'
    pt = '/data2/qt/MusicGeneration/egs/dataset/lmd_matched_MuMIDI/train.pth'
    pv = '/data2/qt/MusicGeneration/egs/dataset/lmd_matched_MuMIDI/valid.pth'
    paths = list(find_files_by_extensions(pp, ['.data']))

    dataset = Melody_Arrangement_Dataset(pp, paths=paths[:-200], verbose=True)
    Melody_Arrangement_Dataset.save_file(dataset, pt)
    # re_dataset = Melody_Arrangement_Dataset()
    # re_dataset = Melody_Arrangement_Dataset.load_file(pv)
    # seq = re_dataset.melody_seqs
    # print(max(seq))
    # seq = sorted(seq, key = lambda i:len(i))
    # print(f'melody_seq={[len(i) for i in seq]}')
    # seq = re_dataset.arrange_seqs
    # print(max(seq))
    # seq = sorted(seq, key = lambda i:len(i))
    # print(f'arrange_seq={[len(i) for i in seq]}')

    dataset = Melody_Arrangement_Dataset(pp, paths=paths[-200:], verbose=True)
    Melody_Arrangement_Dataset.save_file(dataset, pv)
    # print(dataset.melody_seqs)
    # print(dataset.__repr__())
    # re_dataset = Melody_Arrangement_Dataset.load_file(pv)
    # print(re_dataset.melody_seqs)
    # print(re_dataset.__repr__())
