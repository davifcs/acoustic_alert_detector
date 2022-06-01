import argparse
import yaml
import warnings
import glob

import torch
import torchaudio
from torch.utils.data import DataLoader, ConcatDataset
from pytorch_lightning import Trainer, seed_everything, loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint


from utils.general import increment_path
from utils.dataset import ESC50, UrbanSound8K, AudioSet
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
    parser.add_argument('--pre_trained_exp_path', type=str)
    return parser.parse_args()


def main(opt):
    if not opt.pre_trained_exp_path:
        with open(opt.yaml_file, errors='ignore') as f:
            config = yaml.safe_load(f)

        save_dir = increment_path(config['runs_dir'])

        with open(save_dir / 'config.yaml', 'w') as f:
            yaml.safe_dump(config, f, sort_keys=False)
    else:
        with open(f"{opt.pre_trained_exp_path}/{opt.yaml_file}", errors='ignore') as f:
            config = yaml.safe_load(f)

        save_dir = opt.pre_trained_exp_path

    gpus, workers, model, learning_rate, weight_decay, epochs, batch_size, datasets, target_size, target_sr, \
    transforms = config['gpus'], config['workers'], config['model'], config['learning_rate'], config['weight_decay'], \
                 config['epochs'], config['batch_size'], config['datasets'], config['target_size'], \
                 config['target_sr'], config['transforms']

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

    audioset_dataset = AudioSet(datasets['audioset']['annotations_file'], datasets['audioset']['audio_dir'],
                                transforms, target_sr, target_size, model['type'],
                                model['transformer']['patch_size'], device)

    for fold in datasets[datasets['main']]['folds']:
        log_path = f"{save_dir}/{datasets['main']}-{fold}/"
        if datasets['main'] == 'esc50':
            main_dataset_train = ESC50(datasets['esc50']['annotations_file'], datasets['esc50']['audio_dir'],
                                          datasets['esc50']['folds'][:fold-1] + datasets['esc50']['folds'][fold:],
                                          transforms, target_sr, target_size,
                                          model['type'], model['transformer']['patch_size'], device)
            dataset_test = ESC50(datasets['esc50']['annotations_file'], datasets['esc50']['audio_dir'],
                                         [fold], transforms, target_sr, target_size,
                                         model['type'], model['transformer']['patch_size'], device)
        elif datasets['main'] == 'urbansound8k':
            main_dataset_train = UrbanSound8K(datasets['urbansound8k']['annotations_file'],
                                                 datasets['urbansound8k']['audio_dir'],
                                                 datasets['urbansound8k']['folds'][:fold-1] +
                                                 datasets['urbansound8k']['folds'][fold:],
                                                 transforms, target_sr, target_size, model['type'],
                                                 model['transformer']['patch_size'], device)
            dataset_test = UrbanSound8K(datasets['urbansound8k']['annotations_file'],
                                                 datasets['urbansound8k']['audio_dir'],
                                                 [fold],
                                                 transforms, target_sr, target_size, model['type'],
                                                 model['transformer']['patch_size'], device)

        dataset_train = ConcatDataset([main_dataset_train, audioset_dataset])

        train_size = int(len(dataset_train) * (1 - 0.2))
        val_size = len(dataset_train) - train_size
        dataset_train, dataset_val = torch.utils.data.random_split(dataset_train, [train_size, val_size],
                                                                   generator=torch.Generator().manual_seed(42))

        dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, drop_last=True, num_workers=workers)
        dataloader_val = DataLoader(dataset=dataset_train, batch_size=batch_size, drop_last=True, num_workers=workers)
        dataloader_test = DataLoader(dataset=dataset_test, drop_last=True, num_workers=workers)

        if model['type'] == 'convolutional':
            if model['cnn']['dim'] == 2:
                pl_model = CNN2D(learning_rate=learning_rate, log_path=log_path)
            elif model['cnn']['dim'] == 1:
                pl_model = CNN1D(learning_rate=learning_rate, log_path=log_path)
        elif model['type'] == 'transformer':
            pl_model = ViT(embed_dim=model['transformer']['embed_dim'], hidden_dim=model['transformer']['hidden_dim'],
                        num_heads=model['transformer']['num_heads'], patch_size=model['transformer']['patch_size'],
                        num_channels=model['transformer']['num_channels'], num_patches=model['transformer']['num_patches'],
                        num_classes=model['transformer']['num_classes'], dropout=model['transformer']['dropout'],
                        learning_rate=learning_rate, weight_decay=weight_decay, log_path=log_path)
        pl_model.to(device)

        if not opt.pre_trained_exp_path:
            checkpoint_callback = ModelCheckpoint(
                monitor='val_loss',
                dirpath=f"{log_path}/trained_models",
                filename="{epoch:02d}-{val_loss:.4f}",
                mode='min',
            )
            tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_path)

            trainer = Trainer(max_epochs=epochs, gpus=gpus, callbacks=checkpoint_callback,
                              log_every_n_steps=len(dataset_train)/batch_size/4, logger=tb_logger)
            trainer.fit(model=pl_model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)

            trainer.test(ckpt_path=glob.glob(f"{log_path}/trained_models/*.ckpt")[0],
                         test_dataloaders=dataloader_test)
        else:
            trainer = Trainer(gpus=gpus)
            trainer.test(model=pl_model, ckpt_path=glob.glob(f"{log_path}/trained_models/*.ckpt")[0],
                         dataloaders=dataloader_test)


if __name__ == "__main__":
    _args = parse_opt()
    main(_args)

