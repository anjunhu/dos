from dataclasses import dataclass


@dataclass
class VanillaPipeline(object):
    """"""

    dataloader: object  # TODO: type, check nerfstudio for the difference between dataloader and datamanager
    model: object  # TODO: type

    def get_train_loss_dict(self, step: int):
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        return model_outputs, loss_dict, metrics_dict
