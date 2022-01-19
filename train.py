import argparse

import torch
import torchaudio
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

from models.spectrum import AcousticAlertDetector
from utils.dataset import ESC50Dataset


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--annotations_file', type=str)
    parser.add_argument('--audio_dir', type=str)
    parser.add_argument('--target_sample_rate', type=int, default=16000)



    return parser.parse_args()


def main(_args):
    transforms = torchaudio.transforms.MelSpectrogram(sample_rate=1600, n_fft=1024, hop_length=512, n_mels=64)

    device = 'cuda' if _args.gpus > 0 else 'cpu'
    dataset_train = ESC50Dataset( _args.annotations_file, _args.audio_dir, transforms, _args.target_sample_rate, device)

    train_size = int(len(dataset_train) * (1 - 0.2))
    val_size = len(dataset_train) - train_size
    dataset_train, dataset_val = torch.utils.data.random_split(dataset_train, [train_size, val_size])

    dataloader_train = DataLoader(dataset=dataset_train, batch_size=_args.batch_size, drop_last=True)
    dataloader_val = DataLoader(dataset=dataset_train, batch_size=_args.batch_size, drop_last=True)

    model = AcousticAlertDetector(learning_rate=_args.learning_rate)

    trainer = Trainer(max_epochs=_args.epochs, gpus=_args.gpus)
    trainer.fit(model=model, train_dataloader=dataloader_train, val_dataloaders=dataloader_val)


if __name__ == "__main__":
    _args = parse_opt()
    main(_args)
