import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule
from torchmetrics.functional import f1_score

from sklearn.metrics import confusion_matrix, classification_report


class ViT(LightningModule):
    def __init__(self, embed_dim, hidden_dim, num_channels, num_heads, num_classes, patch_size,
                 num_patches, dropout, lr):
        super().__init__()

        # Input
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.criterion = nn.CrossEntropyLoss()

        # Layers
        self.input_layer = nn.Linear(num_channels * (patch_size ** 2), embed_dim)
        # Attention Block
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))
        self.dropout = nn.Dropout(dropout)

        # Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))

        self.lr = lr

    def forward(self, x):
        x = self.input_layer(x)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(x.shape[0], 1, 1)
        x = torch.cat((x, cls_token), dim=1)
        x = x + self.pos_embedding[:, : self.num_patches + 1]

        # Apply Transformer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))

        # Perform classification prediction
        cls = x[0]
        logits = self.mlp_head(cls)
        preds = F.softmax(logits)
        return logits, preds

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch):
        imgs, labels = batch
        logits, preds = self.forward(imgs)
        loss = self.criterion(logits, labels)
        self.log("train_loss", loss.item(), prog_bar=True)
        return loss

    def evaluate(self, batch, stage=None):
        imgs, labels = batch
        logits, preds = self.forward(imgs)
        loss = self.criterion(logits, labels)
        f1 = f1_score(preds, labels, average='macro', num_classes=2)

        self.log(f'{stage}_loss', loss.item(), prog_bar=True)
        self.log(f'{stage}_f1', f1.item(), prog_bar=True)
        return {f'{stage}_loss': loss, f'{stage}_f1': f1, 'predictions': preds, 'label': labels}

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
        avg_loss = torch.stack([output['test_loss'] for output in outputs]).mean()
        avg_f1 = torch.stack([output['test_f1'] for output in outputs]).float().mean()
        preds = torch.cat([output['predictions'] for output in outputs]).float()
        labels = torch.cat([output['label'] for output in outputs]).float()

        report = classification_report(labels.cpu(), preds.cpu())
        cm = confusion_matrix(labels.cpu(), preds.cpu())

        print('\n', report)
        print('Confusion Matrix \n', cm)
        self.log('test_avg_loss', avg_loss.item())
        self.log('test_avg_f1', avg_f1.item())
