# zemi
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
