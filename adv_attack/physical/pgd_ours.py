
import torch
from typing import Optional
import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from models.structures import Instances
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from attack_vectors import build_attack_vector

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

class PhyPGDAttackerOurs:
    def __init__(
        self,
        model,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        eps: float = 4 / 255,
        alpha: float = 1 / 255,
        steps: int = 40,
        attack_type: str = "phy_ours_1", # "ff", "hijack", "dedalus", "ours_1", "ours_2"
        attack_vector: Optional[str] = "acoustic",
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
        self.phy_attack_init, self.phy_attack_generator, self.phy_params_init = None, None, None
        
        # TODO: Update the loss weights based on the attack type
        if self.attack_type == "phy_ours_1" or self.attack_type == "phy_ours_2":
            self.loss_weights = loss_weights
        else:
            raise ValueError(f"Unknown mode: {attack_type}. Supported modes: 'phy_ours_1', 'phy_ours_2'.")

        try:
            self.phy_attack_init, self.phy_attack_generator, self.phy_params_init = build_attack_vector(attack_vector)
        
        except ValueError as e:
            raise ValueError(f"Unknown vector: {attack_vector}. Supported modes: 'acoustic', 'emi'.")

        # Freeze model weights (saves memory, gradients only for inputs)
        for p in self.model.parameters():
            p.requires_grad = False

        self.eps = (eps / torch.tensor(std)).to(device).view(3, 1, 1)
        self.alpha = (alpha / torch.tensor(std)).to(device).view(3, 1, 1)

        if attack_vector == "acoustic":
            self.perturb = self.perturb_aai 
        elif attack_vector == "emi":
            self.perturb = self.perturb_eai
        else:
            raise ValueError(f"Unknown attack vector: {attack_vector}. Supported vectors: 'acoustic', 'emi'.")

    def _forward(self, img, img_hw, track_instances):
        """Forward **with** gradients enabled so PGD can back‑prop."""
        return self.model.inference_single_image(img, img_hw, track_instances)


    # Acoustic Adversarial Injection -----------------------------------------------------------------------------

    def perturb_aai(self, fid: int, img: torch.Tensor, img_hw: tuple, track_instances: Optional[Instances] = None):
    
        image_init = img.to(self.device).detach()
        
        # Initialize transformation parameters 
        x_init, y_init, phi_init, div_init = self.phy_attack_init()

        x_param = torch.tensor([x_init['min']], requires_grad=True, device=self.device)
        y_param = torch.tensor([y_init['min']], requires_grad=True, device=self.device)
        phi_param = torch.tensor([phi_init['min']], requires_grad=True, device=self.device)

        # Define parameter bounds 
        min_x, max_x = x_init['min'], x_init['max']  # pixel translation range
        min_y, max_y = y_init['min'], y_init['max']  # pixel translation range
        min_phi, max_phi = phi_init['min'], phi_init['max']  # rotation range in degrees

        div = 20 # 6 is an estimated low bound, strong effect observed from 10 to 30
                
        for step in range(self.steps):

            # Ensure parameters require gradients
            x_param = x_param.detach().requires_grad_(True)
            y_param = y_param.detach().requires_grad_(True)
            phi_param = phi_param.detach().requires_grad_(True)
            
            # Apply spatial transformation instead of additive perturbation
            adv_img = self.phy_attack_generator(image_init, x_param, y_param, phi_param, div, self.device, skip_gamma=True)
            
            # Ensure output is in valid range
            adv_img = torch.clamp(adv_img, min=self.min_bound, max=self.max_bound)
            
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

            if self.attack_type == "phy_ours_1":
                loss =  loss_matching 
            elif self.attack_type == "phy_ours_2":
                loss = - loss_reborn 
            else:
                raise ValueError(f"Unknown mode: {self.mode}. Supported modes: 'phy_ours_1', 'phy_ours_2'.")

            print(f"Step {step}: bbox {loss_bbox:.3f}, ce {loss_ce:.3f}, giou {loss_giou:.3f}, "
                f"matching {loss_matching:.3f}, conf {loss_conf:.3f}, daedalus {loss_dedalus:.3f}, "
                f"coolshrink {loss_shrink:.3f}, reborn {loss_reborn:.3f}")
            
            # Clear gradients
            self.model.zero_grad(set_to_none=True)
            
            if x_param.grad is not None:
                print(f"x_param.grad: {x_param.grad.item():.6f}")
                x_param.grad.zero_()
            if y_param.grad is not None:
                print(f"y_param.grad: {y_param.grad.item():.6f}")
                y_param.grad.zero_()
            if phi_param.grad is not None:
                print(f"phi_param.grad: {phi_param.grad.item():.6f}")
                phi_param.grad.zero_()

            loss.backward()
            
            if x_param.grad is None or y_param.grad is None or phi_param.grad is None:
                print("Warning: One or more parameter gradients are None. Check transformation function.")
                break

            # print(f"Step {step}: x_param {x_param.grad.item():.3f}, y_param {y_param.grad.item():.3f}, " 
            #       f"phi_param {phi_param.grad.item():.3f}, ")
            
            alpha_scalar = float(self.alpha.mean())
            with torch.no_grad():
                x_param += alpha_scalar * x_param.grad.sign()
                y_param += alpha_scalar * y_param.grad.sign()
                phi_param += alpha_scalar * phi_param.grad.sign()
                
                # Project back to allowed range
                x_param.clamp_(min_x, max_x)
                y_param.clamp_(min_y, max_y)
                phi_param.clamp_(min_phi, max_phi)

        # Generate final adversarial image
        adv_img = self.phy_attack_generator(image_init, x_param, y_param, phi_param, div, self.device)
        adv_img = torch.clamp(adv_img, min=self.min_bound, max=self.max_bound)
        
        return adv_img.detach()

    # Electromagnetic Adversarial Injection -----------------------------------------------------------------------------

    def perturb_eai(self, fid: int, img: torch.Tensor, img_hw: tuple, track_instances: Optional[Instances] = None):

        image_init = img.to(self.device).detach()
        H, W = img.shape[2], img.shape[3]
                
        num_strips = 12  # Mild: [1, 6], Moderate: [7, 12], and Severe: [13, 20]
        emi_mask = self.phy_attack_init(
                    num_strips=num_strips, 
                    H=H, 
                    device=self.device, 
                    init_method='uniform'  # Try: 'uniform', 'random', 'sparse'
                    )

        for step in range(self.steps):

            # Ensure parameters require gradients
            emi_mask = emi_mask.detach().requires_grad_(True)
            
            # Apply EMI attack transformation
            adv_img = self.phy_attack_generator(image_init, emi_mask, device=self.device)
            
            # Ensure output is in valid range
            adv_img = torch.clamp(adv_img, min=self.min_bound, max=self.max_bound)
            
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

            if self.attack_type == "phy_ours_1":
                loss =  loss_matching 
            elif self.attack_type == "phy_ours_2":
                loss = - loss_reborn 
            else:
                raise ValueError(f"Unknown mode: {self.mode}. Supported modes: 'phy_ours_1', 'phy_ours_2'.")

            print(f"Step {step}: bbox {loss_bbox:.3f}, ce {loss_ce:.3f}, giou {loss_giou:.3f}, "
                f"matching {loss_matching:.3f}, conf {loss_conf:.3f}, daedalus {loss_dedalus:.3f}, "
                f"coolshrink {loss_shrink:.3f}, reborn {loss_reborn:.3f}")
                               
            # Clear gradients
            self.model.zero_grad(set_to_none=True)
            
            if  emi_mask.grad is not None:
                emi_mask.grad.zero_()

            # Backward pass
            loss.backward()
            
            # Check gradients
            if emi_mask.grad is None:
                print("Warning: EMI mask gradients are None.")
                break

            # print(f"EMI mask grad stats - min: {emi_mask.grad.min().item():.6f}," 
            #       f"max: {emi_mask.grad.max().item():.6f}, mean: {emi_mask.grad.mean().item():.6f}")
    
            alpha_scalar = float(self.alpha.mean())
            with torch.no_grad():
                emi_mask += alpha_scalar * emi_mask.grad.sign()                

        # Generate final adversarial image
        adv_img = self.phy_attack_generator(image_init, emi_mask, device=self.device)

        adv_img = torch.clamp(adv_img, min=self.min_bound, max=self.max_bound)

        return adv_img.detach()