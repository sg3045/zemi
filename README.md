# Music Genre Classification: SR & Data Dimension Analysis

サンプリング周波数とデータの次元（1D/2D）が、AIの音楽ジャンル識別精度にどのような影響を与えるかを調査しました。

## プロジェクトの目的
AIが音楽を判別する際、「音質（サンプリング周波数）」と「データの形式（波形か画像か）」の組み合わせによって、判断の根拠や精度がどう変化するかを明らかにすることが目的です。

## Google Colabでの実行

## 実施した主な処理
本研究では、1つの音源から「波形」と「スペクトログラム」の2パターンのデータを作成し、元音源（22,050Hz）をそれぞれ16,000Hzと3,000Hzの解像度で検証しました。

```python
import librosa
import torch

# 1. データの読み込み
waveform, _ = librosa.load(file_path, sr=target_sr)

# 2. 1D-CNN用データの作成
input_1d = torch.tensor(waveform).view(1, 1, -1)

# 3. 2D-CNN用データの作成
spec = librosa.feature.melspectrogram(y=waveform, sr=target_sr)
input_2d = torch.tensor(librosa.power_to_db(spec)).unsqueeze(0)
