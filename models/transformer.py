import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics.functional import f1_score, accuracy
from pytorch_lightning import LightningModule

from sklearn.metrics import confusion_matrix, classification_report


class ViT(LightningModule):
    def __init__(self, embed_dim, hidden_dim, num_channels, num_heads, num_classes, patch_size,
                 num_patches, dropout, learning_rate, weight_decay):
        super().__init__()

        self.weight_decay = weight_decay

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

        self.learning_rate = learning_rate

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
        return F.log_softmax(logits, dim=1)

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
