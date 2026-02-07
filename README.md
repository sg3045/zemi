import torch
import torch.nn as nn
import librosa
import numpy as np
from torchvision import models

# ==========================================
# 1. データの前処理 (Sampling Rate変換 & Spectrogram)
# ==========================================
def preprocess_audio(file_path, target_sr=3000):
    # 音声を指定のサンプリングレートで読み込み
    # 16,000Hz vs 3,000Hz の比較に使用
    waveform, sr = librosa.load(file_path, sr=target_sr)
    
    # 2D-CNN用：メルスペクトログラム変換
    # 3,000Hzだと高域がカットされ、リズム(縦線)が際立つ
    mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return waveform, mel_spec_db

# ==========================================
# 2. モデル定義 (2D-CNN: ResNet18)
# ==========================================
class GenreClassifier2D(nn.Module):
    def __init__(self, num_classes=10):
        super(GenreClassifier2D, self).__init__()
        # 学習済みResNet18をベースに使用
        self.model = models.resnet18(pretrained=True)
        
        # 入力をスペクトログラム(1ch)に合わせるため第1層を修正
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 出力を10ジャンル分類用に修正
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

# ==========================================
# 3. 学習設定 (Hyperparameters)
# ==========================================
# 今回の実験で最も安定した設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GenreClassifier2D().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005) # Adamを使用
criterion = nn.CrossEntropyLoss()
batch_size = 32
