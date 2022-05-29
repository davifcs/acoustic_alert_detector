import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics.functional import f1_score, accuracy
from pytorch_lightning import LightningModule

from sklearn.metrics import confusion_matrix, classification_report


class CNN(LightningModule):
    def __init__(self, learning_rate, weight_decay):
        super().__init__()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 0.95)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss.item(), prog_bar=True)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f'{stage}_acc', acc.item(), prog_bar=True)
            self.log(f'{stage}_loss', loss.item(), prog_bar=True)

        return {f'{stage}_loss': loss, f'{stage}_acc': acc, 'predictions': preds, 'label': y}

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self.evaluate(batch, 'test')

    def test_epoch_end(self, outputs):
        preds = torch.cat([output['predictions'] for output in outputs])
        labels = torch.cat([output['label'] for output in outputs])
        f1 = f1_score(preds, labels, average='macro', num_classes=2)

        report = classification_report(labels.cpu(), preds.cpu())
        cm = confusion_matrix(labels.cpu(), preds.cpu(), normalize='true')
        print('\n', report)
        print('Confusion Matrix \n', cm)

        self.log(f'test_f1', f1.item(), prog_bar=True)


class CNN2D(CNN):
    def __init__(self, learning_rate=1e-3, weight_decay=5e-4):
        super().__init__(learning_rate, weight_decay)
        self.criterion = nn.NLLLoss()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=0),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.2),
            # nn.Conv2d(in_channels=64, out_channels=2, kernel_size=(1, 1), stride=(1, 1), padding=0),
            # nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 13 * 12, 512),
            nn.Linear(512, 64),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.conv(x)
        logits = self.linear(x)
        return F.log_softmax(logits, dim=1)


class CNN1D(CNN):
    def __init__(self, learning_rate=1e-3, weight_decay=5e-4):
        super().__init__(learning_rate, weight_decay)
        self.criterion = nn.NLLLoss()

        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=8, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=8, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=32)
        )
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.2),
            # nn.Conv2d(in_channels=64, out_channels=2, kernel_size=(1, 1), stride=(1, 1), padding=0),
            # nn.AvgPool2d(kernel_size=(6, 2), stride=(6, 2))
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 83 * 5, 512),
            nn.Linear(512, 64),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.conv1d(x)
        x = x.unsqueeze(3)
        x = x.permute(0, 3, 2, 1)
        x = self.conv2d(x)
        logits = self.linear(x)
        return F.log_softmax(logits, dim=1)