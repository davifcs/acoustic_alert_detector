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
    def __init__(self, train, transforms, target_sr, target_size, model, patch_size, mixup_alpha=0.2):
        super().__init__()
        self.train = train
        self.transforms = transforms
        self.target_sr = target_sr
        self.target_size = int(target_size * target_sr)
        self.model = model
        self.patch_size = patch_size
        self.mixup_alpha = mixup_alpha

    def _map_target_classes(self, map_class_to_id):
        self.annotations.target = self.annotations.category.apply(
            lambda name: map_class_to_id[name] if name in map_class_to_id.keys() else 0)

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
        else:
            signal = self._random_crop(signal, label)

        return signal

    def _mix_up(self, signal, mixup_signal, label, mixup_label):
        one_hot_label = torch.zeros(2)
        one_hot_label[label] = 1.
        mixup_one_hot_label = torch.zeros(2)
        mixup_one_hot_label[mixup_label] = 1.

        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        signal = lam * signal + (1 - lam) * mixup_signal
        label = lam * one_hot_label + (1 - lam) * mixup_one_hot_label

        return signal, label


class ESC50(BaseDataset):
    def __init__(self, train, annotations_file, audio_dir, folds, transforms, target_sr, target_size, model,
                 patch_size):
        super().__init__(train, transforms, target_sr, target_size, model, patch_size)
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

        if self.train:
            mixup_index = random.randint(0, len(self.annotations) - 1)
            mixup_audio_sample_path = os.path.join(self.audio_dir, self.annotations.filename[mixup_index])
            mixup_label = self.annotations.target[mixup_index]
            mixup_signal = self._load_signal(mixup_audio_sample_path, mixup_label)
            signal, label = self._mix_up(signal, mixup_signal, label, mixup_label)
        else:
            one_hot_label = torch.zeros(2)
            one_hot_label[label] = 1.
            label = one_hot_label
        if self.transforms:
            for transform in self.transforms:
                signal = transform(signal)
        if self.model == 'transformer':
            signal = self._img_to_patch(signal, self.patch_size)
        return signal, label


class UrbanSound8K(BaseDataset):
    def __init__(self, train, annotations_file, audio_dir, folds, transforms, target_sr, target_size, model, patch_size):
        super().__init__(train, transforms, target_sr, target_size, model, patch_size)
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

        if self.train:
            mixup_index = random.randint(0, len(self.annotations) - 1)
            mixup_audio_sample_path = os.path.join(self.audio_dir, "fold"+str(self.annotations.fold[index]),
                                         self.annotations.slice_file_name[index])
            mixup_label = self.annotations['classID'][mixup_index]
            mixup_signal = self._load_signal(mixup_audio_sample_path, mixup_label)
            signal, label = self._mix_up(signal, mixup_signal, label, mixup_label)
        else:
            one_hot_label = torch.zeros(2)
            one_hot_label[label] = 1.
            label = one_hot_label

        if self.transforms:
            for transform in self.transforms:
                signal = transform(signal)
        if self.model == 'transformer':
            signal = self._img_to_patch(signal, self.patch_size)
        return signal, label


class AudioSet(BaseDataset):
    def __init__(self, train, annotations_file, audio_dir, transforms, target_sr, target_size, model, patch_size):
        super().__init__(train, transforms, target_sr, target_size, model, patch_size)
        self.annotations = pd.read_csv(annotations_file, delimiter=',', names=list(range(10)), dtype=object)
        self.annotations.reset_index(drop=True, inplace=True)
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = os.path.join(self.audio_dir, (self.annotations.iloc[index, 0] + "_" +
                                                          str(self.annotations.iloc[index, 1][1:]))+".wav")
        signal = self._load_signal(audio_sample_path, label=1)

        label = torch.zeros(2)
        label[1] = 1.

        if self.transforms:
            for transform in self.transforms:
                signal = transform(signal)
        if self.model == 'transformer':
            signal = self._img_to_patch(signal, self.patch_size)

        return signal, label


def build_weighted_random_sampler(targets):
    targets_unique, counts = np.unique(targets, return_counts=True)
    class_weights = [sum(counts) / c for c in counts]
    weights = [class_weights[e] for e in targets]
    return WeightedRandomSampler(weights, len(targets))


if __name__ == "__main__":
    mel_spectrogram = [torchaudio.transforms.MelSpectrogram(sample_rate=1600, n_fft=1024, hop_length=512, n_mels=64)]

    audioset = UrbanSound8K(
        annotations_file='../data/UrbanSound8K/metadata/UrbanSound8K.csv',
        audio_dir='../data/UrbanSound8K/audio/',
        folds=[1,2],
        transforms=mel_spectrogram,
        target_sr=16000,
        target_size=1,
        model='cnn',
        patch_size=4)

    next(iter(audioset))
