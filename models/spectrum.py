import torch
from torch import nn
from torchmetrics.functional import f1_score
import pytorch_lightning as pl

from sklearn.metrics import confusion_matrix, classification_report


class AcousticAlertDetector(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, weight_decay=5e-4):
        super().__init__()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.2),

        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 1, 128),
            nn.Dropout(p=0.5),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.conv(x)
        logits = self.linear(x)
        preds = torch.argmax(logits, dim=1)
        return logits.squeeze(), preds

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 0.95)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.linear(self.conv(x)).squeeze()
        loss = self.criterion(logits, y.float())
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits, preds = self.forward(x)
        loss = self.criterion(logits, y.float())
        f1 = f1_score(preds, y, average='weighted', num_classes=2)

        self.log(f'{stage}_loss', loss.item(), prog_bar=True)
        self.log(f'{stage}_f1', f1.item(), prog_bar=True)
        return {f'{stage}_loss': loss, f'{stage}_f1': f1, 'predictions': preds, 'label': y}

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self.evaluate(batch, 'test')

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([output['val_loss'] for output in outputs]).mean()
        avg_f1 = torch.stack([output['val_f1'] for output in outputs]).float().mean()

        self.log('val_avg_loss', avg_loss.item(), prog_bar=True)
        self.log('val_avg_f1', avg_f1.item(), prog_bar=True)

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([output['test_avg_loss'] for output in outputs]).mean()
        avg_f1 = torch.stack([output['test_avg_f1'] for output in outputs]).float().mean()
        preds = torch.cat([output['predictions'] for output in outputs]).float()
        labels = torch.cat([output['label'] for output in outputs]).float()

        report = classification_report(labels.cpu(), preds.cpu())
        cm = confusion_matrix(labels.cpu(), preds.cpu())

        print('\n', report)
        print('Confusion Matrix \n', cm)
        self.log('test_avg_loss', avg_loss.item())
        self.log('test_avg_f1', avg_f1.item())
