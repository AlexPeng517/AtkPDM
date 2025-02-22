import torch
import torchvision

import numpy as np
from PIL import Image

def color_to_gray(image):
    gray_image = 0.2999 * image[:, 0] + 0.587 * image[:, 1] + 0.1114 * image[:, 2]
    return gray_image.unsqueeze(1)

def recover_image(clean_img: torch.tensor, protect_img: torch.tensor, threshold: int = 0.5):
    residual_image = (color_to_gray(protect_img) - color_to_gray(clean_img)).abs()
    mask = (residual_image > threshold).int()

    return (1-mask) * protect_img + mask * clean_img

def to_pil(images):
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).numpy()

    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def format_ckpt_dir(ckpt):
    return ckpt.replace('/', '_')

def visualize_feature_maps(features, save_path, T, prefix="", figsize=(30, 50)):
    processed_features = []
    # Process features
    for feature in features:
        feature = feature.squeeze(0)
        gray_scale = torch.sum(feature, 0)
        gray_scale = gray_scale / feature.shape[0]
        processed_features.append(gray_scale.data.cpu().numpy())
    
    # Create visualization
    fig = plt.figure(figsize=figsize)
    for i in range(len(processed_features)):
        a = fig.add_subplot(5, 4, i+1)
        imgplot = plt.imshow(processed_features[i])
        a.axis("off")
        a.set_title(f"{prefix} Feature at {T.item()}", fontsize=20)
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory
