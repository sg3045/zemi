## 核心的な実装コード (Core Workflow)

本プロジェクトのメインロジックを1つのワークフローにまとめました。
「サンプリングレートの変換」から「1D/2Dそれぞれのモデルへの入力」までの流れを示しています。

```python
import librosa
import torch
import torch.nn as nn
from torchvision import models

# 1. データの前処理：ターゲットSR(3000Hz / 16000Hz)に合わせて読み込み
def prepare_data(file_path, target_sr, mode='2D'):
    # ダウンサンプリングして読み込み
    waveform, _ = librosa.load(file_path, sr=target_sr)
    
    if mode == '1D':
        # 1D-CNN用：生の波形をそのまま使用
        return torch.tensor(waveform).unsqueeze(0)
    else:
        # 2D-CNN用：メルスペクトログラム（画像）に変換
        spec = librosa.feature.melspectrogram(y=waveform, sr=target_sr, n_mels=128)
        return torch.tensor(librosa.power_to_db(spec)).unsqueeze(0)

# 2. モデル定義：同じResNet18構造で「次元」だけを変更
# --- 1D-CNN (波形を聴く) ---
class GenreResNet1D(nn.Module):
    def __init__(self):
        super().__init__()
        # 1次元の畳み込み層を持つResNet構造
        self.resnet = CustomResNet1D(layers=[2, 2, 2, 2]) # 18-layer
        
# --- 2D-CNN (模様を見る) ---
class GenreResNet2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7) # 1ch(画像)入力に修正
        self.model.fc = nn.Linear(512, 10) # 10ジャンル分類

# 3. 実験の実行
# SR(3000/16000) × 次元(1D/2D) の4パターンで精度を比較
for sr in [3000, 16000]:
    for model_type in ['1D', '2D']:
        # 学習・評価を実行
        # 結果：2D-CNN × 3000Hz が最高精度 74% を記録
        pass
