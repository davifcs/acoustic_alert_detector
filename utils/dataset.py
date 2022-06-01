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
    def __init__(self, transforms, target_sr, target_size, model, patch_size, device):
        super().__init__()
        if transforms:
            self.transforms = [transform.to(device, non_blocking=True) for transform in transforms]
        else:
            self.transforms = transforms
        self.target_sr = target_sr
        self.target_size = int(target_size * target_sr)
        self.model = model
        self.patch_size = patch_size
        self.device = device

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


class ESC50(BaseDataset):
    def __init__(self, annotations_file, audio_dir, folds, transforms, target_sr, target_size, model, patch_size, device):
        super().__init__(transforms, target_sr, target_size, model, patch_size, device)
        self.annotations = pd.read_csv(annotations_file)
        self.annotations = self.annotations[self.annotations.fold.isin(folds)]
        self.annotations.reset_index(drop=True, inplace=True)
        self.audio_dir = audio_dir
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
        signal = self._random_crop(signal, label)        #
        # if label:
        #     sf.write(file='./pos_samples/'+str(index)+'.wav', data=np.squeeze(signal.cpu().numpy()),
        #              samplerate=self.target_sr, format='WAV')
        if self.transforms:
            for transform in self.transforms:
                signal = transform(signal)
        if self.model == 'transformer':
            signal = self._img_to_patch(signal, self.patch_size)
        return signal, label

    def _map_target_classes(self):
        map_class_to_id = {'siren': 1}
        self.annotations.target = self.annotations.category.apply(
            lambda name: map_class_to_id[name] if name in map_class_to_id.keys() else 0)


class UrbanSound8K(BaseDataset):
    def __init__(self, annotations_file, audio_dir, folds, transforms, target_sr, target_size, model, patch_size, device):
        super().__init__(transforms, target_sr, target_size, model, patch_size, device)
        self.annotations = pd.read_csv(annotations_file)
        self.annotations = self.annotations[self.annotations.fold.isin(folds)]
        self.annotations.reset_index(drop=True, inplace=True)
        self.audio_dir = audio_dir
        self._map_target_classes()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = os.path.join(self.audio_dir, "fold"+str(self.annotations.fold[index]),
                                         self.annotations.slice_file_name[index])
        label = self.annotations['classID'][index]
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device, non_blocking=True)
        if sr != self.target_sr:
            signal = torchaudio.functional.resample(signal, sr, self.target_sr)
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        if signal.shape[1] < self.target_size:
            signal = F.pad(signal, (int((self.target_size/2 - signal.shape[1]/2) + 0.5),
                                    int((self.target_size/2 - signal.shape[1]/2) + 0.5)), "constant", 0)
        else:
            signal = self._random_crop(signal, label)
        # if label:
        #     sf.write(file='./pos_samples/'+str(index)+'.wav', data=np.squeeze(signal.cpu().numpy()),
        #              samplerate=self.target_sr, format='WAV')
        if self.transforms:
            for transform in self.transforms:
                signal = transform(signal)
        if self.model == 'transformer':
            signal = self._img_to_patch(signal, self.patch_size)
        return signal, label

    def _map_target_classes(self):
        map_class_to_id = {'siren': 1}
        self.annotations['classID'] = self.annotations['class'].apply(
            lambda name: map_class_to_id[name] if name in map_class_to_id.keys() else 0)


class AudioSet(BaseDataset):
    def __init__(self, annotations_file, audio_dir, transforms, target_sr, target_size, model, patch_size, device):
        super().__init__(transforms, target_sr, target_size, model, patch_size, device)
        self.annotations = pd.read_csv(annotations_file, delimiter=',', names=list(range(10)), dtype=object)
        self.annotations.reset_index(drop=True, inplace=True)
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = os.path.join(self.audio_dir, (self.annotations.iloc[index, 0] + "_" +
                                                          str(self.annotations.iloc[index, 1][1:]))+".wav")
        label = 1
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device, non_blocking=True)
        if sr != self.target_sr:
            signal = torchaudio.functional.resample(signal, sr, self.target_sr)
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        signal = self._random_crop(signal, label)
        # if label:
        #     sf.write(file='./pos_samples/'+str(index)+'.wav', data=np.squeeze(signal.cpu().numpy()),
        #              samplerate=self.target_sr, format='WAV')
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
        patch_size=4,
        device='cuda')

    next(iter(audioset))
