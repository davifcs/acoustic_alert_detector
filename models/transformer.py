import torch
from torch import nn
from torchmetrics.functional import f1_score, accuracy
from pytorch_lightning import LightningModule
from torchvision.ops.focal_loss import sigmoid_focal_loss
from sklearn.metrics import confusion_matrix, classification_report


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        super().__init__()

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

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x


class ViT(LightningModule):
    def __init__(self, embed_dim, hidden_dim, num_channels, num_heads, num_layers, num_classes, patch_size,
                 num_patches, dropout, learning_rate, log_path):
        super().__init__()

        self.patch_size = patch_size

        # Layers/Networks
        self.input_layer = nn.Linear(num_channels * (patch_size**2), embed_dim)
        self.transformer = nn.Sequential(
            *(AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers))
        )
        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))

        self.criterion = sigmoid_focal_loss
        self.learning_rate = learning_rate
        self.log_path = log_path

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def forward(self, x):
        B, T, _ = x.shape
        x = self.input_layer(x)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:, : T + 1]

        # Apply Transforrmer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)

        # Perform classification prediction
        cls = x[0]
        logits = self.mlp_head(cls)
        preds = torch.argmax(logits, dim=1)
        return logits, preds

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