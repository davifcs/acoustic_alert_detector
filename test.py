import argparse
import yaml

import torchaudio
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

from models.spectrum import AcousticAlertDetector
from utils.dataset import ESC50Dataset


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_path', type=str)
    parser.add_argument('--model_path', type=str)
    return parser.parse_args()


def main(opt):
    with open(opt.exp_path + '/config.yaml', errors='ignore') as f:
        config = yaml.safe_load(f)

    gpus, workers, annotations_file, audio_dir, test_folds, \
    target_size, target_sr, transforms = config['gpus'], config['workers'], config['annotations_file'], \
                                         config['audio_dir'], config['test']['folds'], config['target_size'], \
                                         config['target_sr'], config['transforms']

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

    device = 'cuda' if gpus > 0 else 'cpu'
    dataset_test = ESC50Dataset(annotations_file, audio_dir, test_folds, transforms, target_sr, target_size, device)
    dataloader_test = DataLoader(dataset=dataset_test, drop_last=True, num_workers=workers)

    trainer = Trainer(gpus=gpus)
    trainer.test(model=AcousticAlertDetector(), ckpt_path=opt.exp_path + "/trained_models/" + opt.model_path,
                 dataloaders=dataloader_test)


if __name__ == "__main__":
    _args = parse_opt()
    main(_args)