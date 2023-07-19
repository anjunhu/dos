import torch


class BaseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        raise NotImplementedError()

    def get_metrics_dict(self, model_outputs, batch):
        raise NotImplementedError()

    def get_loss_dict(self, model_outputs, batch, metrics_dict):
        # TODO: check what nerfstudio does with the metrics_dict
        raise NotImplementedError()

    def get_visuals_dict(self, model_outputs, batch):
        raise NotImplementedError()
