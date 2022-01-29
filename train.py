import argparse
import yaml

import torch
import torchaudio
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from utils.general import increment_path
from utils.dataset import ESC50Dataset
from models.spectrum import AcousticAlertDetector


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_file', type=str, default='./config.yaml')
    return parser.parse_args()


def main(opt):
    with open(opt.yaml_file, errors='ignore') as f:
        config = yaml.safe_load(f)

    gpus, workers, learning_rate, weight_decay, epochs, batch_size, annotations_file, audio_dir, runs_dir, target_size,\
    target_sr = config['gpus'], config['workers'], config['learning_rate'], config['weight_decay'], config['epochs'], \
                config['batch_size'], config['annotations_file'], config['audio_dir'], config['runs_dir'], \
                config['target_size'], config['target_sr']

    save_dir = increment_path(runs_dir)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=save_dir,
        filename="{epoch:02d}-{val_loss:.2f}-{val_accuracy:.2f}",
        mode='min',
    )

    transforms = torchaudio.transforms.MelSpectrogram(sample_rate=1600, n_fft=256, hop_length=128, n_mels=64)

    device = 'cuda' if gpus > 0 else 'cpu'
    dataset_train = ESC50Dataset(annotations_file, audio_dir, transforms, target_sr,
                                 target_size, device)

    train_size = int(len(dataset_train) * (1 - 0.2))
    val_size = len(dataset_train) - train_size
    dataset_train, dataset_val = torch.utils.data.random_split(dataset_train, [train_size, val_size])

    dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, drop_last=True, num_workers=workers)
    dataloader_val = DataLoader(dataset=dataset_train, batch_size=batch_size, drop_last=True, num_workers=workers)

    model = AcousticAlertDetector(learning_rate=learning_rate)

    trainer = Trainer(max_epochs=epochs, gpus=gpus, callbacks=checkpoint_callback)
    trainer.fit(model=model, train_dataloader=dataloader_train, val_dataloaders=dataloader_val)


if __name__ == "__main__":
    _args = parse_opt()
    main(_args)
