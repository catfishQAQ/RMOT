
import torch
from typing import Optional
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from models.structures import Instances
import numpy as np

# --------------------------------------------------------------------------- #
#  PGD‑style attacker
# --------------------------------------------------------------------------- #

np.random.seed(2020)
torch.manual_seed(2020)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

loss_weights = {
        "ce": 1.0 * 1.0,
        "conf": 0.0565 * 1.0, 
        "bbox": 0.000424 * 1.0,
        "giou": 0.8695 * 1.0,
        "matching": 0.00651 * 1.0,
        "daedalus": 1.0,
        "coolshrink": 1.0,
        "reborn": 1.0,
        }

def where(cond, x, y):
    """
    code from :
        https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
    """
    cond = cond.float()
    return (cond*x) + ((1-cond)*y)

def hook(g, step):
    print(f"step {step}: "
          f"min {g.min().item():.4e}  "
          f"max {g.max().item():.4e}  "
          f"mean {g.mean().item():.4e}  "
          f"L∞ {g.abs().max().item():.4e}")
    return g        # keep the grad unchanged

class PGDAttackerDaedalus:
    def __init__(
        self,
        model,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        eps: float = 4 / 255,
        alpha: float = 1 / 255,
        steps: int = 40,
        attack_type: str = "daedalus", # "ff", "hijack", "dedalus", "ours_1", "ours_2"
        attack_vector: Optional[str] = "digital",
        device: str = "cuda",
    ) -> None:
        self.model = model
        self.steps = steps
        self.device = device
        self.mean = torch.tensor(mean, device=device).view(3, 1, 1)
        self.std = torch.tensor(std, device=device).view(3, 1, 1)
        
        self.min_bound = (0 - self.mean) / self.std
        self.max_bound = (1 - self.mean) / self.std
        
        self.attack_type = attack_type
        print(f"PGD attack type: {self.attack_type}")

        self.loss_weights = loss_weights

        # Freeze model weights (saves memory, gradients only for inputs)
        for p in self.model.parameters():
            p.requires_grad = False

        self.eps = (eps / torch.tensor(std)).to(device).view(3, 1, 1)
        self.alpha = (alpha / torch.tensor(std)).to(device).view(3, 1, 1)

    def _forward(self, img, img_hw, track_instances):
        """Forward **with** gradients enabled so PGD can back‑prop."""
        return self.model.inference_single_image(img, img_hw, track_instances)

    def perturb(self, fid: int, img: torch.Tensor, img_hw: tuple, track_instances: Optional[Instances] = None):

        image_init = img.to(self.device).detach()
        delta = torch.zeros_like(image_init, device=self.device)

        for step in range(self.steps):
            delta = delta.detach().requires_grad_()
            adv_img = torch.clamp(image_init + delta, min=self.min_bound, max=self.max_bound)

            if hasattr(self.model, "criterion") and hasattr(self.model.criterion, "losses_dict"):
                self.model.criterion.losses_dict.clear()

            frame = self.model.criterion._current_frame_idx
            out = self.model.inference_single_image(adv_img, img_hw, track_instances)

            loss_bbox = out["track_losses"][f"frame_{frame}_loss_bbox"] * self.loss_weights['bbox']
            loss_ce = out["track_losses"][f"frame_{frame}_loss_ce"] * self.loss_weights['ce']
            loss_giou = out["track_losses"][f"frame_{frame}_loss_giou"] * self.loss_weights['giou']
            loss_matching = out["track_losses"][f"frame_{frame}_loss_matching"] * self.loss_weights['matching']
            loss_conf = out["track_losses"][f"frame_{frame}_loss_conf"] * self.loss_weights['conf']
            loss_dedalus = out["track_losses"][f"frame_{frame}_loss_daedalus"] * self.loss_weights['daedalus']
            loss_shrink = out["track_losses"][f"frame_{frame}_loss_coolshrink"] * self.loss_weights['coolshrink']
            loss_reborn = out["track_losses"][f"frame_{frame}_loss_reborn"] * self.loss_weights['reborn']

            loss =  - loss_dedalus

            print(f"Step {step}: bbox {loss_bbox:.3f}, ce {loss_ce:.3f}, giou {loss_giou:.3f}, "
                  f"matching {loss_matching:.3f}, conf {loss_conf:.3f}, daedalus {loss_dedalus:.3f}, "
                  f"coolshrink {loss_shrink:.3f}, reborn {loss_reborn:.3f}")

            self.model.zero_grad(set_to_none=True) # (sets each parameter’s .grad to None instead of zeroing it)

            if delta.grad is not None:
                delta.grad.zero_()
            loss.backward()

            grad = delta.grad
            if grad is None:
                print("Warning: delta.grad is None. Check preprocessing and model input.")
                break

            delta = delta + self.alpha * torch.sign(grad) # update the perturbation following the direction of the gradient 
            delta = torch.clamp(delta, -self.eps, self.eps)
            delta = torch.clamp(image_init + delta, self.min_bound, self.max_bound) - image_init

        adv_img = torch.clamp(image_init + delta, min=self.min_bound, max=self.max_bound)
        return adv_img.detach()

