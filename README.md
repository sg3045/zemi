# zemi
import librosa
import torch.nn as nn
from torchvision import models

# --- データの前処理 ---
# 元音源をターゲットのサンプリングレート(3,000Hz or 16,000Hz)に変換
def get_spectrogram(file_path, target_sr):
    waveform, _ = librosa.load(file_path, sr=target_sr)
    # 2D-CNN用：メルスペクトログラム生成
    mel_spec = librosa.feature.melspectrogram(y=waveform, sr=target_sr, n_mels=128)
    return librosa.power_to_db(mel_spec)

# --- モデル構成 (2D-CNN: ResNet18) ---
# 画像認識モデルを音声識別に応用
class GenreClassifier2D(nn.Module):
    def __init__(self):
        super(GenreClassifier2D, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) # 1ch入力へ
        self.model.fc = nn.Linear(self.model.fc.in_features, 10) # 10ジャンル分類

# --- 学習設定 (Hyperparameters) ---
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005) # 学習の安定化のため0.0005に設定
batch_size = 32
