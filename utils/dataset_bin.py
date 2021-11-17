import torch
from torchvision.datasets.folder import default_loader
from torch.utils import data
from tqdm import tqdm
import sys
import numpy as np
import struct


class DatasetBin(data.Dataset):
    def __init__(self, meta_filename, bin_filename, meta_columns, transform=None, targets_transform=None, loader=default_loader):
        self.loader = loader
        self.transform = (lambda x: x) if transform is None else transform
        self.targets_transform = lambda x: x if targets_transform is None else targets_transform
        self.bin_filename = bin_filename

        self.meta_filename = meta_filename
        self.samples = []
        self.num_subjects = 0

        def convert_type(str_value, key):
            # print(str_value, type(str_value))
            if 'NaN' == str_value:
                return -9999

            if key == 'SUBJECT_ID':
                return int(str_value)
            elif key == 'RACE':
                return int(str_value)
            else:  # key == 'PR_MALE' or anything else
                return float(str_value)

        with open(self.meta_filename, 'r') as f:
            keys = f.readline().strip().split(',')

            assert all(k in keys for k in meta_columns)

            for idx, line in tqdm(enumerate(f), file=sys.stdout, ncols=0, desc=f'Parsing {self.meta_filename}'):
                d = dict(zip(keys, line.strip().split(',')))

                targets = [convert_type(d[k], k) for k in meta_columns]
                if (-9999 not in targets):
                    self.samples.append((targets, idx))
                    self.num_subjects = max(self.num_subjects, int(d['SUBJECT_ID']) + 1)

        with open(self.bin_filename, 'rb') as fin:
            self.bin_rows = struct.unpack('i', fin.read(4))[0]
            self.bin_cols = struct.unpack('i', fin.read(4))[0]
            self.bin_header_offset = fin.tell()

        print('Num Subjects: ', self.num_subjects)

    def __getitem__(self, index):
        targets, bin_index = self.samples[index]
        assert bin_index < self.bin_rows
        targets = self.targets_transform(targets)
        float32_sz = np.dtype(np.float32).itemsize
        with open(self.bin_filename, 'rb') as fin:
            fin.seek(self.bin_header_offset + (float32_sz * self.bin_cols * bin_index))
            sample = torch.from_numpy(
                np.fromfile(fin, dtype=np.float32, count=self.bin_cols))

        return sample, targets

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Ultraface File: {}\n'.format(self.meta_filename)
        fmt_str += '    Bin File: {}\n'.format(self.bin_filename)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Targets Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.targets_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
