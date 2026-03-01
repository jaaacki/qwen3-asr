from logger import log
import soundfile as sf
import numpy as np
import os

AUDIO_FILE = "test01_20s.wav"

def check_audio():
    log.info(f"Checking {AUDIO_FILE}...")
    if not os.path.exists(AUDIO_FILE):
        log.info("File not found.")
        return

    try:
        audio, sr = sf.read(AUDIO_FILE)
        log.info(f"Original SR: {sr}")
        log.info(f"Shape: {audio.shape}")
        log.info(f"Dtype: {audio.dtype}")
        log.info(f"Min: {np.min(audio)}, Max: {np.max(audio)}, Mean: {np.mean(audio)}")

        if np.max(np.abs(audio)) < 1e-4:
            log.warning("WARNING: Audio seems to be silent!")

        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
            log.info("Converted to mono.")

        if sr != 16000:
            log.info(f"Resampling from {sr}Hz to 16000Hz...")
            import torchaudio
            import torch
            tensor = torch.from_numpy(audio).unsqueeze(0).float()
            audio = torchaudio.functional.resample(tensor, sr, 16000).squeeze(0).numpy()
            log.info(f"New Shape: {audio.shape}")
            log.info(f"New Min: {np.min(audio)}, New Max: {np.max(audio)}")

        return audio

    except Exception as e:
        log.error(f"Error processing audio: {e}")

if __name__ == "__main__":
    check_audio()
