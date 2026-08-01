import os
import random

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
        self,
        protocol_path,
        audio_dir,
        n_fft=512,
        hop_length=160,
        max_len=600,
        crop_mode="random",
        **kwargs
    ):
        super().__init__()
        self.samples = []
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_len = max_len
        self.crop_mode = crop_mode

        with open(protocol_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                file_id = parts[1]
                label = 0 if parts[-1] == "bonafide" else 1
                audio_path = os.path.join(audio_dir, file_id + ".flac")
                self.samples.append((audio_path, label, file_id))

    def _pad_or_crop(self, spectrogram):
        t = spectrogram.shape[1]
        if t < self.max_len:
            pad = self.max_len - t
            spectrogram = torch.nn.functional.pad(spectrogram, (0, pad))
        elif t > self.max_len:
            if self.crop_mode == "random":
                start = random.randint(0, t - self.max_len)
            else:
                start = (t - self.max_len) // 2
            spectrogram = spectrogram[:, start : start + self.max_len]
        return spectrogram

    def __getitem__(self, idx: int):
        if isinstance(idx, str):
            return self
        audio_path, label, file_id = self.samples[idx]

        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        audio_np = waveform.squeeze(0).numpy()

        spec = librosa.stft(
            audio_np,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window="blackman",
        )
        spectrogram = np.abs(spec) ** 2
        spectrogram = np.log(spectrogram + 1e-6)
        spectrogram = torch.tensor(spectrogram, dtype=torch.float32)

        spectrogram = self._pad_or_crop(spectrogram)

        # spectrogram = spectrogram.unsqueeze(0)

        # if spectrogram.dim() == 5:
        #     spectrogram = spectrogram.squeeze(2)

        return {"data_object": spectrogram, "labels": label, "file_id": file_id}

    def __len__(self):
        return len(self.samples)
