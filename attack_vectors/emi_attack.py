import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Tuple, Literal
import os
from PIL import Image
import numpy as np
import glob
import cv2

class LoadImages: 
    def __init__(self, path, img_size=(1536, 800)):
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Directory not found: {path}")
        
        img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        
        self.img_files = []
        for ext in img_extensions:
            self.img_files.extend(glob.glob(os.path.join(path, ext)))
            self.img_files.extend(glob.glob(os.path.join(path, ext.upper())))
        
        self.img_files.sort()
        
        if not self.img_files:
            raise FileNotFoundError(f"No image files found in directory: {path}")
        
        self.nf = len(self.img_files) 
        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0
        
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
            
    def __iter__(self):
        self.count = -1
        return self
    
    def __next__(self):
        self.count += 1
        if self.count == len(self):
            raise StopIteration
        
        img_path = self.img_files[self.count]
        img = cv2.imread(img_path)  # BGR
        
        if img is None:
            raise ValueError(f'Failed to load image: {img_path}')
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
        
        cur_img, ori_img = self.init_img(img)
        return self.count, cur_img, ori_img, img_path
    
    def init_img(self, img):
        ori_img = img.copy()
        seq_h, seq_w = img.shape[:2]
        scale = self.height / min(seq_h, seq_w)
        if max(seq_h, seq_w) * scale > self.width:
            scale = self.width / max(seq_h, seq_w)
        target_h = int(seq_h * scale)
        target_w = int(seq_w * scale)
        img = cv2.resize(img, (target_w, target_h))
        img = transforms.functional.normalize(transforms.functional.to_tensor(img), self.mean, self.std)
        img = img.unsqueeze(0)
        return img, ori_img
    
    def __len__(self):
        return self.nf  # number of files

    def tensor_to_img(self, img_tensor):
        if img_tensor.dim() == 4:
            img_tensor = img_tensor.squeeze(0)
        
        img = img_tensor.detach().cpu().numpy()
        
        if np.isnan(img).any() or np.isinf(img).any():
            return np.zeros((img.shape[1], img.shape[2], 3), dtype=np.uint8)
        
        img = img * np.array(self.std)[:, None, None] + np.array(self.mean)[:, None, None]
                
        if img.max() > 1.0 or img.min() < 0.0:
            img_min, img_max = img.min(), img.max()
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.clip(img, 0, 1)
        
        img = (img * 255).astype(np.uint8)
        img = np.transpose(img, (1, 2, 0))
        
        return img

# ----------------------------------------------------------------------------------------- #

def differentiable_rgb_to_bayer_rggb(rgb_tensor: torch.Tensor, device: str) -> torch.Tensor:
    """Simulates the RGGB Bayer pattern from an RGB tensor."""
    B, C, H, W = rgb_tensor.shape
    
    # Create masks with gradient tracking
    r_mask = torch.zeros((H, W), device=device, dtype=rgb_tensor.dtype, requires_grad=False)
    r_mask[0::2, 0::2] = 1
    
    g_mask = torch.zeros((H, W), device=device, dtype=rgb_tensor.dtype, requires_grad=False)
    g_mask[0::2, 1::2] = 1
    g_mask[1::2, 0::2] = 1
    
    b_mask = torch.zeros((H, W), device=device, dtype=rgb_tensor.dtype, requires_grad=False)
    b_mask[1::2, 1::2] = 1
    
    # Apply masks to channels
    r_channel = rgb_tensor[:, 0, :, :] * r_mask
    g_channel = rgb_tensor[:, 1, :, :] * g_mask
    b_channel = rgb_tensor[:, 2, :, :] * b_mask
    
    # Combine channels
    bayer = r_channel.unsqueeze(1) + g_channel.unsqueeze(1) + b_channel.unsqueeze(1)
    return bayer

def differentiable_demosaic_bilinear(bayer_tensor: torch.Tensor, device: str) -> torch.Tensor:
    """Differentiable bilinear demosaicing for an RGGB Bayer pattern using convolutions."""
    B, C, H, W = bayer_tensor.shape
    
    # Create kernels with matching dtype and device
    r_kernel = torch.tensor([[[[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]]]], 
                            dtype=bayer_tensor.dtype, device=device)
    g_kernel = torch.tensor([[[[0.0, 0.25, 0.0], [0.25, 1.0, 0.25], [0.0, 0.25, 0.0]]]], 
                            dtype=bayer_tensor.dtype, device=device)
    b_kernel = torch.tensor([[[[0.25, 0.25, 0.25], [0.25, 1.0, 0.25], [0.25, 0.25, 0.25]]]], 
                            dtype=bayer_tensor.dtype, device=device)
    
    # Create masks with matching dtype
    r_mask = torch.zeros((H, W), device=device, dtype=bayer_tensor.dtype)
    r_mask[0::2, 0::2] = 1
    
    g_mask = torch.zeros((H, W), device=device, dtype=bayer_tensor.dtype)
    g_mask[0::2, 1::2] = 1
    g_mask[1::2, 0::2] = 1
    
    b_mask = torch.zeros((H, W), device=device, dtype=bayer_tensor.dtype)
    b_mask[1::2, 1::2] = 1
    
    # Apply masks and convolve
    r_pre = bayer_tensor * r_mask
    g_pre = bayer_tensor * g_mask
    b_pre = bayer_tensor * b_mask
    
    r = F.conv2d(r_pre, r_kernel, padding='same')
    g = F.conv2d(g_pre, g_kernel, padding='same')
    b = F.conv2d(b_pre, b_kernel, padding='same')
    
    rgb_out = torch.cat([r, g, b], dim=1)
    return torch.clamp(rgb_out, 0, 1)

def create_stripe_mask_from_2d_params(stripe_params_2d: torch.Tensor, H: int, device: str) -> torch.Tensor:
    num_strips = stripe_params_2d.shape[0]
    stripe_mask = torch.zeros(H, device=device, dtype=stripe_params_2d.dtype)
    
    # Create row indices tensor for vectorized operations
    row_indices = torch.arange(H, device=device, dtype=stripe_params_2d.dtype)
    
    for i in range(num_strips):
        row_id = stripe_params_2d[i, 0]
        width = stripe_params_2d[i, 1]
        
        # Apply soft constraints (differentiable)
        row_id_clamped = torch.clamp(row_id, 0.0, float(H - 1))
        max_width = float(H) - row_id_clamped
        min_width = torch.tensor(1.0, device=device, dtype=stripe_params_2d.dtype)
        width_clamped = torch.clamp(width, min_width, max_width)
        
        # Soft assignment instead of discrete rounding
        # Create a smooth "box" function using sigmoids
        row_start = row_id_clamped
        row_end = row_id_clamped + width_clamped
        
        # Use sigmoid functions to create soft boundaries
        # sigmoid(k*(x - start)) creates a soft step function at 'start'
        # sigmoid(k*(end - x)) creates a soft step function at 'end'
        # Multiply them to get a soft box
        sharpness = 10.0  # Controls how sharp the boundaries are
        
        soft_start = torch.sigmoid(sharpness * (row_indices - row_start))
        soft_end = torch.sigmoid(sharpness * (row_end - row_indices))
        soft_box = soft_start * soft_end
        
        # Union behavior: take maximum activation for overlapping strips
        stripe_mask = torch.maximum(stripe_mask, soft_box)
    
    return stripe_mask

# -------------------------------------------------------------------- #
#  The Main Differentiable EMI Attack Simulator Function               #
# -------------------------------------------------------------------- #

def apply_emi_attack(
    rgb_tensor: torch.Tensor,
    stripe_params: torch.Tensor,
    mode: Literal['shift', 'glitch'] = 'glitch',
    sharpness: float = 10.0,
    device: str = 'cuda',
    intensity: float = 1.0,
    imagenet_mean: tuple = (0.485, 0.456, 0.406),
    imagenet_std: tuple = (0.229, 0.224, 0.225)
) -> torch.Tensor:
    """
    Enhanced EMI attack using 2D stripe parameters [row_id, width].
    Args:
        rgb_tensor: Input tensor (B, 3, H, W)
        stripe_params: 2D tensor of shape (num_strips, 2) where each row is [row_id, width]
        mode: Attack type ('shift' or 'glitch')
        sharpness: Controls mask edge softness
        device: Computation device
        intensity: Attack strength
        imagenet_mean, imagenet_std: Normalization parameters
    """
    B, C, H, W = rgb_tensor.shape
    
    # Validate 2D stripe_params with shape (num_strips, 2)
    if stripe_params.dim() != 2 or stripe_params.shape[1] != 2:
        raise ValueError(f"stripe_params must be 2D with shape (num_strips, 2), got {stripe_params.shape}")
        
    num_strips = stripe_params.shape[0]
        
    # Auto-detect tensor format
    min_val = rgb_tensor.min().item()
    max_val = rgb_tensor.max().item()
    is_imagenet_normalized = (min_val < -0.5 and max_val > 1.5) or (min_val < 0 and abs(rgb_tensor.mean().item()) < 1.0)
    
    # Create tensors with proper dtype and device
    mean_tensor = torch.tensor(imagenet_mean, device=device, dtype=rgb_tensor.dtype).view(1, 3, 1, 1)
    std_tensor = torch.tensor(imagenet_std, device=device, dtype=rgb_tensor.dtype).view(1, 3, 1, 1)
    
    if is_imagenet_normalized:
        # ImageNet normalized tensor processing
        
        # Step 1: Denormalize to [0,1] space for Bayer processing
        rgb_denorm = rgb_tensor * std_tensor + mean_tensor
        
        # Step 2: Apply EMI attack in [0,1] space
        raw = differentiable_rgb_to_bayer_rggb(rgb_denorm, device)
        raw_attacked = raw.clone()
        
        if mode == 'shift':
            raw_attacked = torch.roll(raw_attacked, shifts=-1, dims=2)
        elif mode == 'glitch':
            # Create mask with proper dtype
            mask_g_channel = torch.ones_like(raw, dtype=raw.dtype, device=device)
            mask_g_channel[:, :, 0::2, 1::2] = 0  # Zero out G pixels in even rows, odd columns
            mask_g_channel[:, :, 1::2, 0::2] = 0  # Zero out G pixels in odd rows, even columns
            raw_attacked = raw * mask_g_channel
        
        attacked_rgb = differentiable_demosaic_bilinear(raw_attacked, device)
        
        # Step 3: Create stripe mask from 2D parameters
        soft_mask_1d = create_stripe_mask_from_2d_params(stripe_params, H, device)
        
        # Apply sharpness if specified
        if sharpness > 0:
            soft_mask_1d = torch.sigmoid(sharpness * (soft_mask_1d - 0.5))
        
        # Expand 1D mask (H,) to 2D (H, W) by repeating across width
        stripe_params_2d_mask = soft_mask_1d.unsqueeze(1).expand(H, W)
        
        # Expand to match tensor dimensions (B, C, H, W)
        mask_4d = stripe_params_2d_mask.unsqueeze(0).unsqueeze(0).expand(B, C, H, W)
        
        # Blend with intensity scaling in [0,1] space
        rgb_result = rgb_denorm * (1 - mask_4d) + attacked_rgb * mask_4d * intensity
        
        # Step 4: Re-normalize back to ImageNet space
        final_image = (rgb_result - mean_tensor) / std_tensor
                
    else:
        # [0,1] tensor processing
        raw = differentiable_rgb_to_bayer_rggb(rgb_tensor, device)
        raw_attacked = raw.clone()
        
        if mode == 'shift':
            raw_attacked = torch.roll(raw_attacked, shifts=-1, dims=2)
        elif mode == 'glitch':
            mask_g_channel = torch.ones_like(raw, dtype=raw.dtype, device=device)
            mask_g_channel[:, :, 0::2, 1::2] = 0
            mask_g_channel[:, :, 1::2, 0::2] = 0
            raw_attacked = raw * mask_g_channel
        
        attacked_rgb_full = differentiable_demosaic_bilinear(raw_attacked, device)
        
        # Create stripe mask from 2D parameters
        soft_mask_1d = create_stripe_mask_from_2d_params(stripe_params, H, device)
        
        # Apply sharpness if specified
        if sharpness > 0:
            soft_mask_1d = torch.sigmoid(sharpness * (soft_mask_1d - 0.5))
        
        # Expand 1D mask (H,) to 2D (H, W) by repeating across width
        stripe_params_2d_mask = soft_mask_1d.unsqueeze(1).expand(H, W)
        
        # Expand to match tensor dimensions (B, C, H, W)
        mask_4d = stripe_params_2d_mask.unsqueeze(0).unsqueeze(0).expand(B, C, H, W)
        
        final_image = rgb_tensor * (1 - mask_4d) + attacked_rgb_full * mask_4d * intensity
            
    return final_image

def initialize_stripe_params(num_strips: int, H: int, device: str, 
                               init_method: str = 'uniform') -> torch.Tensor:
    # Returns:   2D tensor of shape (num_strips, 2) where each row is [row_id, width]
    if init_method == 'uniform':
        # Uniformly distribute strips across image height
        stripe_params = torch.zeros(num_strips, 2, device=device, dtype=torch.float32)
        
        # Initialize row IDs uniformly across height
        for i in range(num_strips):
            stripe_params[i, 0] = (i + 0.5) * H / num_strips  # row_id
            stripe_params[i, 1] = H // (num_strips * 2)       # width (small initial width)
    
    elif init_method == 'random':
        # Random initialization
        stripe_params = torch.zeros(num_strips, 2, device=device, dtype=torch.float32)
        stripe_params[:, 0] = torch.rand(num_strips, device=device) * (H - 1)  # row_id in [0, H-1]
        stripe_params[:, 1] = torch.rand(num_strips, device=device) * 20 + 5   # width in [5, 25]
    
    elif init_method == 'sparse':
        # Sparse initialization with small strips
        stripe_params = torch.zeros(num_strips, 2, device=device, dtype=torch.float32)
        
        # Space strips out evenly but with small widths
        for i in range(num_strips):
            stripe_params[i, 0] = torch.rand(1, device=device) * H * 0.8 + H * 0.1  # Avoid edges
            stripe_params[i, 1] = 5 + torch.rand(1, device=device) * 10  # width in [5, 15]
    
    else:
        raise ValueError(f"Unknown init_method: {init_method}")
    
    return stripe_params.requires_grad_(True).requires_grad_(True)

def init_emi_params() -> tuple:
    num_stripes =   {'min': 3, 'max': 20}  # Horizontal translation range
    intensity  = {'min': 0.1, 'max': 1.0}  # Intensity range
    sharpness = {'min': 1.0, 'max': 20.0}  # Sharpness range
    
    return num_stripes, intensity, sharpness

# -------------------------------------------------------------------- #
#                    Testing and Verification                          #
# -------------------------------------------------------------------- #

if __name__ == "__main__":
    
    loader = LoadImages('./images')
    for i, (count, processed_img, original_img, img_path) in enumerate(loader):
        clean_image_tensor = processed_img
        break

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean_image_tensor = clean_image_tensor.to(device)
    
    H = clean_image_tensor.shape[2] # Get height for relative conversion
    W = clean_image_tensor.shape[3] # Get width

    num_strips = 3  
    stripe_params = initialize_stripe_params(
        num_strips=num_strips, 
        H=H, 
        device=device, 
        init_method='uniform'  # 'uniform', 'random', 'sparse'
    )

    print(f"Initialized {num_strips} strips:")
    for i in range(num_strips):
        row_id = stripe_params[i, 0].item()
        width = stripe_params[i, 1].item()
        print(f"  Strip {i}: row_id={row_id:.1f}, width={width:.1f}")

    # Apply EMI attack with 2D parameters
    attacked_image = apply_emi_attack(
        clean_image_tensor,
        stripe_params,      # Now 2D tensor (num_strips, 2)
        mode='glitch',   
        sharpness=10,         
        device=device,
        intensity=1.0,           
        imagenet_mean=loader.mean, 
        imagenet_std=loader.std
    )

    # Calculate a dummy loss for gradient testing
    loss = attacked_image.mean() 
    
    # Clear any existing gradients
    if stripe_params.grad is not None:
        stripe_params.grad.zero_()
    
    loss.backward()

    # Check gradients for both row_id and width parameters
    if stripe_params.grad is not None:
        print(f"\n✅ Gradient flow SUCCESS for 2D parameters:")
        for i in range(num_strips):
            row_grad = stripe_params.grad[i, 0].item()
            width_grad = stripe_params.grad[i, 1].item()
            print(f"  Strip {i}: row_id_grad={row_grad:.6f}, width_grad={width_grad:.6f}")
    else:
        print("❌ No gradients computed!")

    attacked_image_np = loader.tensor_to_img(attacked_image)

    output_dir = "./images/results"
    os.makedirs(output_dir, exist_ok=True)

    img_show = cv2.cvtColor(attacked_image_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_dir, "glitch.jpg"), img_show)
    
    print(f"\n💾 Saved result to: {os.path.join(output_dir, 'glitch.jpg')}")
        
    # fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    
    # axes[0].imshow(original_display)
    # axes[0].set_title("Original Image")
    # axes[0].axis("off")
    
    # axes[1].imshow(glitch_display)
    # axes[1].set_title("Differentiable 'Glitch' Attack")
    # axes[1].axis("off")

    # axes[2].imshow(shift_display)
    # axes[2].set_title("Differentiable 'Shift' Attack")
    # axes[2].axis("off")
    
    # plt.tight_layout()
    # plt.suptitle("Testing the Differentiable Physical Attack Simulator", fontsize=16, y=1.02)
    # plt.show()
    