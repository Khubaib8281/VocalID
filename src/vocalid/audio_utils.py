import torch 
import torchaudio
import sounddevice as sd

from .config import SAMPLE_RATE

def load_audio(path, target_sr=SAMPLE_RATE):
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform

def record_audio(seconds=4, fs=SAMPLE_RATE):
    print(f"Recording {seconds} seconds...")
    audio = sd.rec(int(seconds*fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    print("Recording complete")
    return (torch.tensor(audio.T))