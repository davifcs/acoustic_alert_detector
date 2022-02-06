import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import pandas as pd
import torchaudio


class ESC50Dataset(Dataset):
    def __init__(self, annotations_file, audio_dir, folds, transforms, target_sr, target_size, device):
        self.annotations = pd.read_csv(annotations_file)
        self.annotations = self.annotations[self.annotations.fold.isin(folds)]
        self.annotations.reset_index(drop=True, inplace=True)
        self.audio_dir = audio_dir
        self.transforms = transforms.to(device, non_blocking=True)
        self.target_sr = target_sr
        self.target_size = int(target_size * target_sr)
        self.device = device
        self._map_target_classes()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = os.path.join(self.audio_dir, self.annotations.filename[index])
        label = self.annotations.target[index]
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device, non_blocking=True)
        if sr != self.target_sr:
            signal = torchaudio.functional.resample(signal, sr, self.target_sr)
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        signal = self._random_crop(signal)
        signal = self.transforms(signal)
        return signal, label

    def _map_target_classes(self):
        map_class_to_id = {'car_horn': 1, 'siren': 1}
        self.annotations.target = self.annotations.category.apply(
            lambda name: map_class_to_id[name] if name in map_class_to_id.keys() else 0)

    def _random_crop(self, signal):
        start = random.randint(0, signal.shape[1] - self.target_size)
        return signal[:, start: start + self.target_size]


def build_weighted_random_sampler(targets):
    targets_unique, counts = np.unique(targets, return_counts=True)
    class_weights = [sum(counts) / c for c in counts]
    weights = [class_weights[e] for e in targets]
    return WeightedRandomSampler(weights, len(targets))


if __name__ == "__main__":
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=1600, n_fft=1024, hop_length=512, n_mels=64)

    esc50ds = ESC50Dataset(annotations_file='../data/ESC-50-master/meta/esc50.csv',
                           audio_dir='../data/ESC-50-master/audio/',
                           transforms=mel_spectrogram,
                           target_sr=16000,
                           target_size=1,
                           device='cuda')
