import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
import numpy as np
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader

from src.datasets.data_utils import get_dataloaders

# from src.model.lcnn import AngularSoftmax
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)


    train_dataset = dataloaders["train"].dataset
    labels = []
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        if "labels" in sample:
            labels.append(sample["labels"])
        elif "label" in sample:
            labels.append(sample["label"])
        else:
            raise KeyError("sososos no key")
    
    labels = np.array(labels)
    class_count = np.bincount(labels, minlength=2)
    class_weights = class_count.sum() / (2.0 * class_count)
    sample_weights = class_weights[labels]
    
    logger.info(f"Class counts [bonafide, spoof]: {class_count.tolist()}")
    logger.info(f"Class weights [bonafide, spoof]: {class_weights.tolist()}")
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True
    )
    

    old_loader = dataloaders["train"]
    dataloaders["train"] = DataLoader(
        dataset=train_dataset,
        batch_size=old_loader.batch_size,
        sampler=sampler,
        num_workers=old_loader.num_workers,
        pin_memory=old_loader.pin_memory,
        drop_last=old_loader.drop_last if hasattr(old_loader, 'drop_last') else False,
    )


    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    logger.info(model)

    # Добавить логирование количества параметров
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # get function handles of loss and metrics
    loss_function = torch.nn.CrossEntropyLoss().to(device)

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights_tensor).to(device)

    # criterion_angular = AngularSoftmax(in_features=80, out_features=2, m=4).to(device)
    # criterion_angular = None

    metrics = instantiate(config.metrics)

    # build optimizer, learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, params=trainable_params)
    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")


    sample = train_dataset[0]
    print(f"Sample shape: {sample['data_object'].shape}")
    print(f"Sample dtype: {sample['data_object'].dtype}")
    print(f"Sample min: {sample['data_object'].min():.3f}, max: {sample['data_object'].max():.3f}")

    # Проверяем батч
    sample_batch = next(iter(train_loader))
    print(f"Batch shape: {sample_batch['data_object'].shape}")

    trainer = Trainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
        # criterion_angular=criterion_angular,
        # criterion_angular=None,
    )

    trainer.train()


if __name__ == "__main__":
    main()