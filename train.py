import argparse
import yaml
import warnings
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import DataLoader, ConcatDataset
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from utils.general import increment_path
from utils.dataset import ESC50Dataset, AudioSetDataset
from models.convolutional import CNN2D, CNN1D
from models.transformer import ViT

SEED = 42
seed_everything(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_file', type=str, default='./config.yaml')
    return parser.parse_args()


def main(opt):
    with open(opt.yaml_file, errors='ignore') as f:
        config = yaml.safe_load(f)

    save_dir = increment_path(config['runs_dir'])

    with open(save_dir / 'config.yaml', 'w') as f:
        yaml.safe_dump(config, f, sort_keys=False)

    gpus, workers, model, patch_size, learning_rate, weight_decay, epochs, batch_size, annotations_file, audio_dir, \
    train_folds, test_folds, target_size, target_sr, transforms = config['gpus'], config['workers'], config['model'],\
                                                                  config['patch_size'], config['learning_rate'], \
                                                                  config['weight_decay'], config['epochs'],\
                                                                  config['batch_size'], config['annotations_file'],\
                                                                  config['audio_dir'], config['train']['folds'], \
                                                                  config['test']['folds'], config['target_size'], \
                                                                  config['target_sr'], config['transforms']

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=Path(f"{save_dir}/trained_models"),
        filename="{epoch:02d}-{val_avg_loss:.2f}-{val_avg_f1:.2f}",
        mode='min',
    )

    if transforms['type'] == "mel_spectrogram":
        transforms = [torchaudio.transforms.MelSpectrogram(sample_rate=target_sr,
                                                          f_min=0,
                                                          n_fft=transforms['mel_spectrogram']['n_fft'],
                                                          win_length=transforms['mel_spectrogram']['n_fft'],
                                                          hop_length=transforms['mel_spectrogram']['hop_length'],
                                                          center=transforms['mel_spectrogram']['center'],
                                                          normalized=transforms['mel_spectrogram']['normalized'],
                                                          mel_scale="slaney",
                                                          n_mels=transforms['mel_spectrogram']['n_mels'],
                                                          power=transforms['mel_spectrogram']['power']),
                      torchaudio.transforms.AmplitudeToDB(top_db=80.0)]

    elif transforms['type'] == "mfcc":
        transforms = [torchaudio.transforms.MFCC(sample_rate=target_sr, n_mfcc=transforms["mfcc"]["n_mfcc"],
                                                 dct_type=transforms["mfcc"]["dct_type"],
                                                 norm=transforms["mfcc"]["norm"],
                                                 melkwargs={"f_min": 0,
                                                            "n_fft": transforms['mel_spectrogram']['n_fft'],
                                                            "win_length": transforms['mel_spectrogram']['n_fft'],
                                                            "hop_length": transforms['mel_spectrogram']['hop_length'],
                                                            "center": transforms['mel_spectrogram']['center'],
                                                            "normalized": transforms['mel_spectrogram']['normalized'],
                                                            "mel_scale": "slaney",
                                                            "n_mels": transforms['mel_spectrogram']['n_mels'],
                                                            "power": transforms['mel_spectrogram']['power']})]
    else:
        transforms = None

    device = 'cuda' if gpus > 0 else 'cpu'
    esc50_dataset = ESC50Dataset(annotations_file[0], audio_dir[0], train_folds, transforms, target_sr, target_size,
                                 model['type'], patch_size, device)

    audioset_dataset = AudioSetDataset(annotations_file[1], audio_dir[1], transforms, target_sr, target_size,
                                       model['type'], patch_size, device)

    dataset_train = ConcatDataset([esc50_dataset, audioset_dataset])

    train_size = int(len(dataset_train) * (1 - 0.2))
    val_size = len(dataset_train) - train_size
    dataset_train, dataset_val = torch.utils.data.random_split(dataset_train, [train_size, val_size],
                                                               generator=torch.Generator().manual_seed(42))

    dataset_test = ESC50Dataset(annotations_file[0], audio_dir[0], test_folds, transforms, target_sr, target_size,
                                model['type'], patch_size, device)

    dataloader_train = DataLoader(dataset=dataset_train, batch_size=1, drop_last=True, num_workers=workers)
    dataloader_val = DataLoader(dataset=dataset_train, batch_size=1, drop_last=True, num_workers=workers)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=1, drop_last=True, num_workers=workers)

    if model['type'] == 'convolutional':
        if model['cnn']['dim'] == 2:
            model = CNN2D(learning_rate=learning_rate, weight_decay=weight_decay)
        elif model['cnn']['dim'] == 1:
            model = CNN1D(learning_rate=learning_rate, weight_decay=weight_decay)
    elif model['type'] == 'transformer':
        model = ViT(embed_dim=model['transformer']['embed_dim'], hidden_dim=model['transformer']['hidden_dim'],
                    num_heads=model['transformer']['num_heads'], patch_size=model['transformer']['patch_size'],
                    num_channels=model['transformer']['num_channels'], num_patches=model['transformer']['num_patches'],
                    num_classes=model['transformer']['num_classes'], dropout=model['transformer']['dropout'],
                    learning_rate=learning_rate)
    model.to(device)

    trainer = Trainer(max_epochs=epochs, gpus=gpus, callbacks=checkpoint_callback,
                      log_every_n_steps=len(dataset_train)/batch_size/4)
    trainer.fit(model=model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)
    trainer.test(ckpt_path='best', test_dataloaders=dataloader_test)


if __name__ == "__main__":
    _args = parse_opt()
    main(_args)
