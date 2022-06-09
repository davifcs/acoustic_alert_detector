import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import pandas as pd
import torchaudio
from torch.nn import functional as F

import soundfile as sf

SEED = 42
random.seed(SEED)


class BaseDataset(Dataset):
    def __init__(self, train, transforms, target_sr, target_size, model, patch_size, mixup):
        super().__init__()
        self.train = train
        self.transforms = transforms
        self.target_sr = target_sr
        self.target_size = int(target_size * target_sr)
        self.model = model
        self.patch_size = patch_size
        self.mixup = mixup

    def _map_target_classes(self, map_class_to_id):
        self.annotations.target = self.annotations.category.apply(
            lambda name: map_class_to_id[name] if name in map_class_to_id.keys() else 0)

    def _crop(self, signal):
        cropped_signal = torch.empty(0, self.target_size)
        for start in range(0, signal.shape[1], self.target_size):
            cropped_signal = torch.vstack([cropped_signal, signal[:, start: start + self.target_size]])
        return cropped_signal

    def _random_crop(self, signal, label):
        cropped_rms = 0
        while cropped_rms < 0.0001:
            start = random.randint(0, signal.shape[1] - self.target_size)
            cropped_signal = signal[:, start: start + self.target_size]
            cropped_rms = torch.sqrt(torch.mean(cropped_signal ** 2))
            if label == 0:
                break
        return cropped_signal

    def _img_to_patch(self, signal, patch_size):
        b, h, w = signal.shape

        signal = signal.reshape(h // patch_size, patch_size, w // patch_size, patch_size)
        signal = signal.permute(0, 2, 1, 3)  # [h', w', p_h, p_w]
        signal = signal.flatten(0, 1)  # [h'*w', c, p_h, p_w]
        signal = signal.flatten(1, 2)  # [h'*w', c*p_h*p_h]
        return signal

    def _load_signal(self, audio_sample_path, label):
        signal, sr = torchaudio.load(audio_sample_path)
        if sr != self.target_sr:
            signal = torchaudio.functional.resample(signal, sr, self.target_sr)
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        if signal.shape[1] < self.target_size:
            signal = F.pad(signal, (int((self.target_size / 2 - signal.shape[1] / 2) + 0.5),
                                    int((self.target_size / 2 - signal.shape[1] / 2) + 0.5)), "constant", 0)
        elif self.train:
            signal = self._random_crop(signal, label)
        else:
            signal = self._crop(signal)
        return signal

    def _mix_up(self, signal, mixup_signal, label, mixup_label):
        gain_signal = torch.max(torchaudio.transforms.AmplitudeToDB(top_db=80)(signal))
        gain_mixup_signal = torch.max(torchaudio.transforms.AmplitudeToDB(top_db=80)(mixup_signal))

        ratio = random.random()

        p = 1.0 / (1 + np.power(10, (gain_signal - gain_mixup_signal) / 20.) * (1 - ratio) / ratio)
        signal = ((signal * p + mixup_signal * (1 - p)) / np.sqrt(p ** 2 + (1 - p) ** 2))

        eye = torch.eye(2)
        label = (eye[label] * ratio + eye[mixup_label] * (1 - ratio))

        return signal, label

    def _pre_processing(self, signal):
        if self.transforms:
            for transform in self.transforms:
                if isinstance(signal, list):
                    for i, s in enumerate(signal):
                        signal[i] = transform(s)
                else:
                    signal = transform(signal)
        if self.model == 'transformer':
            if isinstance(signal, list):
                for i, s in enumerate(signal):
                    signal[i] = self._img_to_patch(s, self.patch_size)
        return signal


class ESC50(BaseDataset):
    def __init__(self, train, annotations_file, audio_dir, folds, transforms, target_sr, target_size, model, patch_size,
                 mixup=False):
        super().__init__(train, transforms, target_sr, target_size, model, patch_size, mixup)
        self.annotations = pd.read_csv(annotations_file)
        self.annotations = self.annotations[self.annotations.fold.isin(folds)]
        self.annotations.reset_index(drop=True, inplace=True)
        self.audio_dir = audio_dir
        self._map_target_classes(map_class_to_id={'siren': 1})

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = os.path.join(self.audio_dir, self.annotations.filename[index])
        label = self.annotations.target[index]
        signal = self._load_signal(audio_sample_path, label)

        if self.train and self.mixup and bool(random.getrandbits(1)):
            mixup_index = random.randint(0, len(self.annotations) - 1)
            mixup_audio_sample_path = os.path.join(self.audio_dir, self.annotations.filename[mixup_index])
            mixup_label = self.annotations.target[mixup_index]
            mixup_signal = self._load_signal(mixup_audio_sample_path, mixup_label)
            signal, label = self._mix_up(signal, mixup_signal, label, mixup_label)
        else:
            eye = torch.eye(2)
            label = eye[label]
        signal = self._pre_processing(signal)
        return signal, label


class UrbanSound8K(BaseDataset):
    def __init__(self, train, annotations_file, audio_dir, folds, transforms, target_sr, target_size, model, patch_size,
                 mixup=False):
        super().__init__(train, transforms, target_sr, target_size, model, patch_size, mixup)
        self.annotations = pd.read_csv(annotations_file)
        self.annotations = self.annotations[self.annotations.fold.isin(folds)]
        self.annotations.reset_index(drop=True, inplace=True)
        self.audio_dir = audio_dir
        self._map_target_classes(map_class_to_id={'siren': 1})

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = os.path.join(self.audio_dir, "fold"+str(self.annotations.fold[index]),
                                         self.annotations.slice_file_name[index])
        label = self.annotations['classID'][index]
        signal = self._get_item(audio_sample_path, label)

        if self.train and self.mixup and bool(random.getrandbits(1)):
            mixup_index = random.randint(0, len(self.annotations) - 1)
            mixup_audio_sample_path = os.path.join(self.audio_dir, "fold"+str(self.annotations.fold[mixup_index]),
                                         self.annotations.slice_file_name[mixup_index])
            mixup_label = self.annotations['classID'][mixup_index]
            mixup_signal = self._load_signal(mixup_audio_sample_path, mixup_label)
            signal, label = self._mix_up(signal, mixup_signal, label, mixup_label)
        else:
            eye = torch.eye(2)
            label = eye[label]
        signal = self._pre_processing(signal)
        return signal, label


class AudioSet(BaseDataset):
    def __init__(self, train, annotations_file, audio_dir, transforms, target_sr, target_size, model, patch_size,
                 mixup):
        super().__init__(train, transforms, target_sr, target_size, model, patch_size, mixup)
        self.annotations = pd.read_csv(annotations_file, delimiter=',', names=list(range(10)), dtype=object)
        self.annotations.reset_index(drop=True, inplace=True)
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = os.path.join(self.audio_dir, (self.annotations.iloc[index, 0] + "_" +
                                                          str(self.annotations.iloc[index, 1][1:]))+".wav")
        signal = self._load_signal(audio_sample_path, label=1)

        if self.train and self.mixup and bool(random.getrandbits(1)):
            mixup_index = random.randint(0, len(self.annotations) - 1)
            mixup_audio_sample_path = os.path.join(self.audio_dir, (self.annotations.iloc[mixup_index, 0] + "_" +
                                                          str(self.annotations.iloc[mixup_index, 1][1:]))+".wav")
            mixup_label = 1
            mixup_signal = self._load_signal(mixup_audio_sample_path, mixup_label)
            signal, label = self._mix_up(signal, mixup_signal, 1, mixup_label)
        else:
            eye = torch.eye(2)
            label = eye[1]
        signal = self._pre_processing(signal)
        return signal, label


def build_weighted_random_sampler(targets):
    targets_unique, counts = np.unique(targets, return_counts=True)
    class_weights = [sum(counts) / c for c in counts]
    weights = [class_weights[e] for e in targets]
    return WeightedRandomSampler(weights, len(targets))


def collate_fn(batch):
    x, y = batch[0]
    if len(x.shape) < 3:
        x = x.reshape(x.shape[0], 1, x.shape[1])
    else:
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
    return x, y.reshape(-1, 2)


if __name__ == "__main__":
    esc50_dataset = ESC50(
                        train=False,
                        annotations_file='../data/ESC-50-master/meta/esc50.csv',
                        audio_dir='../data/ESC-50-master/audio/',
                        folds=[1, 2],
                        transforms=None,
                        target_sr=22050,
                        target_size=1,
                        model='cnn',
                        patch_size=4,
                        mixup=True)

