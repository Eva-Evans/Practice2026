from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def __init__(self, model, criterion, optimizer, device, **kwargs):
        self.criterion_angular = kwargs.pop('criterion_angular', None)
        metrics = kwargs.pop('metrics', None)

        super().__init__(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            metrics=metrics,
            **kwargs
        )
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self._batch_idx = 0

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        if self.criterion_angular is not None:
            embeddings = self.model.get_features(batch["data_object"])
            logits = self.criterion_angular(embeddings, batch["labels"])
            batch["logits"] = logits
            all_losses = self.criterion(logits, batch["labels"])
        else:
            logits = self.model(batch["data_object"])
            batch["logits"] = logits
            all_losses = self.criterion(logits, batch["labels"])

        batch["loss"] = all_losses

        # --- Логирование (каждый батч) ---
        print(f"Loss requires_grad: {batch['loss'].requires_grad}")
        print(f"Loss grad_fn: {batch['loss'].grad_fn}")

        if self.is_train:
            batch["loss"].backward()
            
            # Проверка градиентов у каждого слоя (каждые 10 батчей)
            if self._batch_idx % 10 == 0:
                print("\n" + "="*60)
                print("GRADIENT CHECK (every 10 batches):")
                has_grad = False
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        if grad_norm > 0:
                            print(f"  {name}: grad_norm={grad_norm:.6f}")
                            has_grad = True
                        else:
                            print(f"  {name}: grad_norm=0.000000 (zero gradient)")
                    else:
                        print(f"  {name}: grad is None")
                if not has_grad:
                    print("  ⚠️  WARNING: No non-zero gradients found!")
                print("="*60 + "\n")
            
            # Проверка общей нормы градиентов (каждые 100 батчей)
            if self._batch_idx % 100 == 0:
                total_norm = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                print(f"  Total gradient norm: {total_norm:.4f}")
                
                if total_norm > 100:
                    print("  WARNING: Gradient explosion detected!")
                elif total_norm == 0:
                    print("  WARNING: Total gradient norm is ZERO!")
            
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        
        self._batch_idx += 1

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            # Log Stuff
            pass
        else:
            # Log Stuff
            pass