import torch
from torch import nn
import pytorch_lightning as pl

from sklearn.metrics import confusion_matrix, classification_report


class SpectrumNetwork(pl.LightningModule):
    def __init__(self, learning_rate):
        super().__init__()

        self.learning_rate = learning_rate
        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 5 * 4, 1),
        )

    def forward(self, x):
        logits = self.linear(self.conv(x))
        preds = torch.argmax(logits, dim=1)
        return logits, preds

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits, preds = self.forward(x)
        loss = self.criterion(logits, y)
        accuracy = (preds == y)
        self.log(f'{stage}_loss', loss, prog_bar=True)
        self.log(f'{stage}_acc', accuracy, prog_bar=True)
        return {f'{stage}_loss': loss, f'{stage}_acc': accuracy, 'predictions': preds, 'label': y}

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, 'val')

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, 'test')

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([output['val_losses'] for output in outputs]).mean()
        accuracy = torch.stack([output['val_avg_accuracy'] for output in outputs]).float().mean()

        self.log('val_loss', avg_loss.item(), prog_bar=True)
        self.log('val_accuracy', accuracy.item(), prog_bar=True)

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([output['test_losses'] for output in outputs]).mean()
        accuracy = torch.stack([output['test_avg_accuracy'] for output in outputs]).float().mean()
        preds = torch.stack([output['predictions'] for output in outputs]).float()
        labels = torch.stack([output['labels'] for output in outputs]).float()

        report = classification_report(labels.cpu(), preds.cpu())
        cm = confusion_matrix(labels.cpu(), preds.cpu())

        print('\n', report)
        print('Confusion Matrix \n', cm)
        self.log('test_loss', avg_loss.item())
        self.log('test_accuracy', accuracy.item())
