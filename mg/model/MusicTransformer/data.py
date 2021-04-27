import utils
import random
import pickle
import numpy as np
import torch

import config


class Data:
    def __init__(self, dir_path, max_length):
        # self.files = list(utils.find_files_by_extensions(dir_path, ['.pickle']))
        # self.file_dict = {
        #     'train': self.file_filter(self.files[:int(len(self.files) * 0.8)], max_length),
        #     'eval': self.file_filter(self.files[int(len(self.files) * 0.8): int(len(self.files) * 0.9)], max_length),
        #     'test': self.file_filter(self.files[int(len(self.files) * 0.9):], max_length),
        # }
        self.file_dict = {}
        for item in ['train', 'valid', 'test']:
            # print(dir_path+item+'_processed/')
            self.files = list(utils.find_files_by_extensions(dir_path+'processed_'+item+'/', ['.data']))
            # print(self.files[:5])
            self.file_dict[item] = self.file_filter(self.files, max_length)
        self._seq_file_name_idx = 0
        self._seq_idx = 0

        pass

    def __repr__(self):
        return f"<class Data has train: {len(self.file_dict['train'])}, val: {len(self.file_dict['valid'])},test: {len(self.file_dict['test'])} files>"


    def file_filter(self, files, max_length):
        filtered_data = []
        for fname in files:
            # print(fname)
            data = torch.load(fname)
            if max_length <= len(data):
                filtered_data.append(fname)
        return filtered_data

    def batch(self, batch_size, length, mode='train'):
        batch_files = random.sample(self.file_dict[mode], k=batch_size)

        batch_data = [
            self._get_seq(file, length)
            for file in batch_files
        ]
        return np.array(batch_data, dtype=np.int16)  # batch_size, seq_len

    def seq2seq_batch(self, batch_size, length, mode='train'):
        data = self.batch(batch_size, length * 2, mode)
        x = data[:, :length]
        y = data[:, length:]
        return x, y

    def smallest_encoder_batch(self, batch_size, length, mode='train'):
        data = self.batch(batch_size, length * 2, mode)
        x = data[:, :length//100]
        y = data[:, length//100:length//100+length]
        return x, y

    def slide_seq2seq_batch(self, batch_size, length, mode='train'):
        data = self.batch(batch_size, length+1, mode)
        x = data[:, :-1]
        y = data[:, 1:]
        return x, y

    def random_sequential_batch(self, batch_size, length):
        batch_files = random.sample(self.files, k=batch_size)
        batch_data = []
        for i in range(batch_size):
            data = self._get_seq(batch_files[i])
            for j in range(len(data) - length):
                batch_data.append(data[j:j+length])
                if len(batch_data) == batch_size:
                    return batch_data

    def sequential_batch(self, batch_size, length):
        batch_data = []
        data = self._get_seq(self.files[self._seq_file_name_idx])

        while len(batch_data) < batch_size:
            while self._seq_idx < len(data) - length:
                batch_data.append(data[self._seq_idx: self._seq_idx + length])
                self._seq_idx += 1
                if len(batch_data) == batch_size:
                    return batch_data

            self._seq_idx = 0
            self._seq_file_name_idx = self._seq_file_name_idx + 1
            if self._seq_file_name_idx == len(self.files):
                self._seq_file_name_idx = 0
                print('iter intialized')

    def _get_seq(self, fname, max_length=None):
        data = torch.load(fname)
        if max_length is not None:
            if max_length <= len(data):
                start = random.randrange(0,len(data) - max_length)
                data = data[start:start + max_length]
            else:
                raise IndexError
                # data = np.append(data, config.token_eos)
                # while len(data) < max_length:
                #     data = np.append(data, config.pad_token)
        return data


class PositionalY:
    def __init__(self, data, idx):
        self.data = data
        self.idx = idx

    def position(self):
        return self.idx

    def data(self):
        return self.data

    def __repr__(self):
        return '<Label located in {} position.>'.format(self.idx)


def add_noise(inputs: np.array, rate:float = 0.01): # input's dim is 2
    seq_length = np.shape(inputs)[-1]

    num_mask = int(rate * seq_length)
    for inp in inputs:
        rand_idx = random.sample(range(seq_length), num_mask)
        inp[rand_idx] = random.randrange(0, config.pad_token)

    return inputs


if __name__ == '__main__':
    import pprint
    def count_dict(max_length, data):
        cnt_arr = [0] * max_length
        cnt_dict = {}
        # print(cnt_arr)
        for batch in data:
            for index in batch:
                try:
                    cnt_arr[int(index)] += 1

                except:
                    print(index)
                try:
                    cnt_dict['index-'+str(index)] += 1
                except KeyError:
                    cnt_dict['index-'+str(index)] = 1
        return cnt_arr
