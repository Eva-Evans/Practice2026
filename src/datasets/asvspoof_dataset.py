import os

import torch
import torchaudio
import librosa
import numpy as np
from torch.utils.data import DataLoader, Dataset


class ASVSpoofDataset(Dataset):
    # LFCC-LCNN: LFCC were extracted similar to baseline system with 20 ms window length, 512 number of FFT
    # bins and 20 filters.
    # Only the first 600 features for each file were used as LCNN input in all single systems.
    def __init__(
        self, protocol_path, audio_dir, n_fft=512, hop_length=160, max_len=600, **kwargs
    ):
        super().__init__()
        self.samples = []
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_len = max_len

        with open(protocol_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                file_id = parts[1]
                label = 1 if parts[-1] == "bonafide" else 0
                audio_path = os.path.join(audio_dir, file_id + ".flac")
                self.samples.append((audio_path, label))
        

        

    def __getitem__(self, idx: int):
        if isinstance(idx, str):
            return self
        audio_path, label = self.samples[idx]

        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        

        audio_np = waveform.squeeze(0).numpy()

        spec = librosa.stft(
            audio_np,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window='blackman'
        )
        spectrogram = np.abs(spec)
        spectrogram = np.log(spectrogram + 1e-9)
        spectrogram = torch.tensor(spectrogram, dtype=torch.float32)


        if spectrogram.shape[1] > self.max_len:
            spectrogram = spectrogram[:, :self.max_len]
        elif spectrogram.shape[1] < self.max_len:
            pad = self.max_len - spectrogram.shape[1]
            spectrogram = torch.nn.functional.pad(spectrogram, (0, pad))

        # spectrogram = spectrogram.unsqueeze(0)

        # if spectrogram.dim() == 5:
        #     spectrogram = spectrogram.squeeze(2)

        return {"data_object": spectrogram, "labels": label}

    def __len__(self):
        return len(self.samples)
