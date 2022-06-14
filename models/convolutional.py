import torch
from torch import nn
from torchmetrics.functional import f1_score, accuracy
from pytorch_lightning import LightningModule
from torchvision.ops.focal_loss import sigmoid_focal_loss

from sklearn.metrics import confusion_matrix, classification_report


class CNN(LightningModule):
    def __init__(self, learning_rate, log_path, patience):
        super().__init__()

        self.learning_rate = learning_rate
        self.patience = patience
        self.log_path = log_path

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.patience)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, preds = self.forward(x)
        loss = self.criterion(logits, y, reduction='mean')
        self.log('train_loss', loss.item(), prog_bar=True)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits, preds = self.forward(x)
        if stage == 'test':
            logits = logits.mean(dim=0).reshape(-1, 2)
            preds = torch.argmax(logits, dim=1)
        loss = self.criterion(logits, y, reduction='mean')
        acc = accuracy(preds, y.argmax(dim=1))

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
        labels = torch.cat([output['label'] for output in outputs]).argmax(dim=1)
        f1 = f1_score(preds, labels, average='macro', num_classes=2)

        report = classification_report(labels.cpu(), preds.cpu())
        cm = confusion_matrix(labels.cpu(), preds.cpu())

        with open(self.log_path+'results.txt', 'a') as f:
            f.write(report + '\n')
            f.write('Confusion Matrix \n' + str(cm))

        self.log(f'test_f1', f1.item(), prog_bar=True)


class CNN2D(CNN):
    def __init__(self, learning_rate=1e-3, log_path='./', patience=20):
        super().__init__(learning_rate, log_path, patience)
        self.criterion = sigmoid_focal_loss

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
            nn.Linear(32 * 5 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.conv(x)
        logits = self.linear(x)
        preds = torch.argmax(logits, dim=1)
        return logits, preds


class CNN1D(CNN):
    def __init__(self, learning_rate=1e-3, log_path='./', patience=20):
        super().__init__(learning_rate, log_path, patience)
        self.criterion = sigmoid_focal_loss

        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=64, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=32, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=16, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8, stride=2, padding=0),
            nn.ReLU()
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.conv1d(x)
        logits = self.linear(x)
        preds = torch.argmax(logits, dim=1)
        return logits, preds


class DSCNN(CNN):
    def __init__(self, learning_rate=1e-3, log_path='./', patience=20):
        super().__init__(learning_rate, log_path, patience)
        self.criterion = sigmoid_focal_loss

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), padding=1, groups=1),
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, groups=16),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, groups=16),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, groups=32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, groups=32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, groups=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.2),
            # nn.Conv2d(in_channels=64, out_channels=2, kernel_size=(1, 1), stride=(1, 1), padding=0),
            # nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 5, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.conv(x)
        logits = self.linear(x)
        preds = torch.argmax(logits, dim=1)
        return logits, preds
