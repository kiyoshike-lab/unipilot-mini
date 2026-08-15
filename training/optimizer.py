import torch


def create_optimizer(model, learning_rate: float, weight_decay: float = 0.1):
    decay, no_decay = [], []
    for parameter_name, parameter in model.named_parameters():
        (decay if parameter.dim() >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=learning_rate,
        betas=(0.9, 0.95),
    )
