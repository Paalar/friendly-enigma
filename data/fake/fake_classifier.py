import torch

class Fake_Classifier():
    def predict(self, batch):
        targets = torch.tensor([
            1 if get_sum_with_max_contribution_per_value(tensor) > 20 else 0
            for tensor in batch
        ])
        return targets

def get_sum_with_max_contribution_per_value(tensor: torch.Tensor):
        values = tensor.tolist()
        target_val = 0
        for value in values:
            positive_capped = value if value < 20 else 20
            negative_capped = positive_capped if positive_capped > -20 else -20
            target_val += negative_capped
        return target_val
