from torch.utils.data import DataLoader

from src.datasets.asvspoof_dataset import ASVSpoofDataset

dataset = ASVSpoofDataset(
    protocol_path="../Practice2026dataset/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
    audio_dir="../Practice2026dataset/LA/LA/ASVspoof2019_LA_train/flac/",
)

print(f"Samples: {len(dataset)}")
spec, label = dataset[0]
print(f"Spectrogram: {spec.shape}, Label: {label}")


train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
batch = next(iter(train_loader))
print(f"Batch spectrograms: {batch[0].shape}")
print(f"Batch labels: {batch[1].shape}")
