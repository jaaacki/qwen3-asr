import soundfile as sf
import librosa
import numpy as np
import torch
import os

AUDIO_FILE = "test01_20s.wav"

def check_audio():
    print(f"Checking {AUDIO_FILE}...")
    if not os.path.exists(AUDIO_FILE):
        print("File not found.")
        return

    try:
        # Load with soundfile
        audio, sr = sf.read(AUDIO_FILE)
        print(f"Original SR: {sr}")
        print(f"Shape: {audio.shape}")
        print(f"Dtype: {audio.dtype}")
        print(f"Min: {np.min(audio)}, Max: {np.max(audio)}, Mean: {np.mean(audio)}")
        
        if np.max(np.abs(audio)) < 1e-4:
            print("WARNING: Audio seems to be silent!")
        
        # Convert to mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
            print("Converted to mono.")
        
        # Resample
        if sr != 16000:
            print("Resampling to 16000Hz...")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            print(f"New Shape: {audio.shape}")
            print(f"New Min: {np.min(audio)}, New Max: {np.max(audio)}")
            
        return audio
        
    except Exception as e:
        print(f"Error processing audio: {e}")

if __name__ == "__main__":
    check_audio()
