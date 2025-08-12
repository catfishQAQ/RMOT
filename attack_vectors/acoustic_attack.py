import os
import torch
import math
import torch.nn.functional as F
from typing import Callable
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# --- All of your motion blur functions go here ---
def _gamma_correction(imgs: torch.Tensor, gamma: float) -> torch.Tensor:
    return (imgs + 1e-6)**gamma

def radial_blur(img, s, div, device):
    s = torch.linspace(1 - s, 1, div, device=device)
    zeros = torch.zeros_like(s)
    affine_tensor = torch.stack([
        torch.stack([s, zeros, zeros]),
        torch.stack([zeros, s, zeros]),
    ]).permute(2, 0, 1)
    grid = F.affine_grid(affine_tensor, [div, *img.shape[1:]], align_corners=False)
    imgs = img.unsqueeze(dim=1).expand(-1, div, -1, -1, -1)
    # This loop is inefficient but differentiable. Vectorizing it is possible but more complex.
    res = []
    for i in range(img.shape[0]):
        samples = F.grid_sample(imgs[i], grid, padding_mode="border", align_corners=False)
        blur_img = torch.mean(samples, dim=0, keepdim=True)
        res.append(blur_img)
    res = torch.cat(res, dim=0)
    return res

def _sine_grid_new(div: int, device: str, phi: float) -> torch.Tensor:
    grid = torch.linspace(-math.pi, math.pi, div, device=device)
    grid = torch.sin(grid + phi)
    return grid

def _stn_blur(img: torch.Tensor, div: int, x: torch.Tensor, y: torch.Tensor,
              mean_func: Callable, device: str) -> torch.Tensor:
    # --- SUGGESTED VECTORIZED VERSION (MUCH FASTER) ---
    N, C, H, W = img.shape
    ones = torch.ones_like(x, device=device)
    zeros = torch.zeros_like(x, device=device)
    affine_tensor = torch.stack([
        torch.stack([ones, zeros, x]),
        torch.stack([zeros, ones, y]),
    ]).permute(2, 0, 1)
    # The grid is the same for all images in the batch
    grid = F.affine_grid(affine_tensor, [div, C, H, W], align_corners=False)
    # Repeat the grid for each image in the batch
    grid = grid.unsqueeze(0).repeat(N, 1, 1, 1, 1) # Shape: (N, div, H, W, 2)
    # Expand the input image tensor to match the number of transformations
    imgs_expanded = img.unsqueeze(dim=1).expand(-1, div, -1, -1, -1) # Shape: (N, div, C, H, W)
    # Reshape for grid_sample
    # grid_sample expects input (N, C, H_in, W_in) and grid (N, H_out, W_out, 2)
    # We "fold" the div dimension into the batch dimension
    imgs_reshaped = imgs_expanded.reshape(N * div, C, H, W)
    grid_reshaped = grid.reshape(N * div, H, W, 2)
    # Sample all images and transformations at once
    samples = F.grid_sample(imgs_reshaped, grid_reshaped, padding_mode="border", align_corners=False)
    # Reshape back to (N, div, C, H, W)
    samples = samples.view(N, div, C, H, W)
    # Apply the mean function along the 'div' dimension
    blur_img = mean_func(samples)
    return blur_img

# -------------------------------------------------------------------- #
#  The Main Differentiable Acoustic Attack Simulator Function          #
# -------------------------------------------------------------------- #

def stn_blur_general(img: torch.Tensor,
                     x: float,
                     y: float,
                     phi: float,
                     div: int,
                     device: str,
                     gamma: float = 2.2,
                     skip_gamma: bool = True) -> torch.Tensor:  # Add flag
    
    x_offsets = _sine_grid_new(div, device, phi) * x
    y_offsets = _sine_grid_new(div, device, 0) * y
    
    if skip_gamma:
        # Simple mean without gamma correction for normalized images
        mean_func = lambda s: torch.mean(s, dim=1)
        res = _stn_blur(img, div, x_offsets, y_offsets, mean_func, device)
        return res
    else:
        # Original gamma correction for [0,1] images
        mean_func = lambda s: torch.mean(_gamma_correction(s, gamma), dim=1)
        res = _stn_blur(img, div, x_offsets, y_offsets, mean_func, device)
        return (res + 1e-6)**(1/gamma)

def init_blur_params() -> tuple:
    """
    :return: Tuple of initialized parameters
    :param x: Horizontal offset
    :param y: Vertical offset
    :param phi: Rotation angle in radians  
    """
    x =   {'min': 1e-2, 'max': 3e-2}  # Horizontal translation range
    y =   {'min': 1e-2, 'max': 3e-2}  # Vertical translation range
    phi = {'min': 1e-1, 'max': 1.}  # Rotation range in degrees
    div = {'min': 1.0, 'max': 2.0}  # Number of divisions for the radial blur
    
    return x, y, phi, div  # Convert degrees to radians for internal consistency

# -------------------------------------------------------------------- #
#                    Testing and Verification                          #
# -------------------------------------------------------------------- #

if __name__ == "__main__":

    image_path = "./images/clean.png"
    img_pil = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts PIL to (C, H, W) tensor in [0, 1]
    ])
    img = transform(img_pil).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = img.to(device)

    # --- PROOF OF DIFFERENTIABILITY (Attacking the Parameters) ---
    print("\n--- Performing Gradient Check on Attack Parameters ---")
    
    # Define the attack parameters as tensors that require gradients
    x_param = torch.tensor([0.01], requires_grad=True, device=device)
    y_param = torch.tensor([1e-3], requires_grad=True, device=device)
    phi_param_deg = torch.tensor([0.], requires_grad=True, device=device) # In degrees

    # IMPORTANT: Use torch.deg2rad for the differentiable conversion
    phi_param_rad = torch.deg2rad(phi_param_deg)

    # Run the blur function with the tensor parameters to build the computation graph
    test_output = stn_blur_general(img, x_param, y_param, phi_param_rad, 10, device, skip_gamma=True)

    print("**************", test_output.shape, test_output.min(), test_output.max())

    # Calculate a dummy loss. In a real attack, this is the tracker's failure score.
    loss = test_output.mean()
    loss.backward()

    # Check if gradients were computed for each of our input parameters
    for name, param in [('x', x_param), ('y', y_param), ('phi', phi_param_deg)]:
        if param.grad is not None and torch.any(param.grad != 0):
            print(f"SUCCESS: Gradient for parameter '{name}' computed successfully: {param.grad.item()}")
        else:
            print(f"FAILURE: Gradient for parameter '{name}' was not computed.")
    print("--- Gradient Check Complete ---")

    # --- Display Results ---
    original_display = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    blurred_display = test_output.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

    output_dir = "./images/results"
    os.makedirs(output_dir, exist_ok=True)

    # Convert numpy arrays to PIL Images and save
    Image.fromarray((blurred_display * 255).astype('uint8')).save(os.path.join(output_dir, "blurred.png"))

    # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # axes[0].imshow(original_display)
    # axes[0].set_title("Original Image")
    # axes[0].axis("off")

    # axes[1].imshow(blurred_display)
    # axes[1].set_title(f"Differentiable Motion Blur\n(x={x_viz}, y={y_viz}, phi={phi_viz_deg}°)")
    # axes[1].axis("off")

    # plt.tight_layout()
    # plt.show()