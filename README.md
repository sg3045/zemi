## Core Implementation

本プロジェクトの主要な実装コードの抜粋です。
Librosaを用いたダウンサンプリングと、PyTorchを用いたResNetベースの学習を行っています。

### 1. Data Preprocessing (Downsampling)
元音源(22,050Hz)を、実験条件である16,000Hzと3,000Hzに変換します。

```python
import librosa
import torchaudio

def preprocess_audio(file_path, target_sr):
    # 音声の読み込みとリサンプリング
    waveform, sr = librosa.load(file_path, sr=target_sr)
    
    # 2D-CNN用：メルスペクトログラムへの変換
    # 3,000Hzの場合は高域がカットされ、低域のパターンが強調される
    mel_spec = librosa.feature.melspectrogram(y=waveform, sr=target_sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return waveform, mel_spec_db

import torch.nn as nn
from torchvision import models

class GenreClassifier2D(nn.Module):
    def __init__(self, num_classes=10):
        super(GenreClassifier2D, self).__init__()
        # Pre-trained ResNet18を使用
        self.model = models.resnet18(pretrained=True)
        
        # 入力チャンネルを1(グレースケール)に変更
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 出力層を10ジャンルに合わせて変更
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)
        
