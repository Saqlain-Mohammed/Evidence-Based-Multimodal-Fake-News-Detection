"""Independent randomly initialized networks. No pretrained weights or API models."""
import torch
from torch import nn

class ImageCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        channels = [3, 24, 48, 96]
        for a, b in zip(channels, channels[1:]):
            layers += [nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.GELU(), nn.MaxPool2d(2)]
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(.2), nn.Linear(96, 1))

    def forward(self, x):
        return self.head(self.pool(self.features(x))).squeeze(-1)

class AudioCNN(nn.Module):
    """Log-magnitude STFT input, trained only on bona-fide/spoof labels."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 24, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(96, 1)

    def forward(self, x):
        return self.head(self.features(x).flatten(1)).squeeze(-1)

class VideoCNN(nn.Module):
    """Frame CNN plus temporal GRU. One label per clip, never per-frame split."""
    def __init__(self):
        super().__init__()
        self.encoder = ImageCNN()
        self.temporal = nn.GRU(96, 64, batch_first=True, bidirectional=True)
        self.head = nn.Linear(128, 1)

    def forward(self, x):
        b, t, c, h, w = x.shape
        frames = self.encoder.pool(self.encoder.features(x.reshape(b*t, c, h, w))).flatten(1)
        sequence, _ = self.temporal(frames.reshape(b, t, -1))
        return self.head(sequence.mean(1)).squeeze(-1)

class TextTransformer(nn.Module):
    """Optional scratch neural alternative to the CPU TF-IDF baseline."""
    def __init__(self, vocab_size=30000, length=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(1, length, 128) * .02)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(128, 4, 384, .15, batch_first=True), 3)
        self.head = nn.Linear(128, 1)

    def forward(self, x):
        mask = x.eq(0)
        y = self.encoder(self.embed(x) + self.pos[:, :x.shape[1]], src_key_padding_mask=mask)
        valid = (~mask).unsqueeze(-1)
        return self.head((y * valid).sum(1) / valid.sum(1).clamp(min=1)).squeeze(-1)
