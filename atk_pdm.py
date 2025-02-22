import argparse
import os
import sys
import json
from PIL import Image

import numpy as np
from einops import rearrange, repeat
from piq import LPIPS
import lpips

import torch
from torch import nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms
import torchvision.datasets as datasets
from torchvision.utils import save_image, make_grid

from diffusers import AutoencoderKL, DiffusionPipeline


import matplotlib.pyplot as plt
from tqdm import tqdm, trange

from utils.attack_validator import AttackValidator
from utils.utils import to_pil, visualize_feature_maps
torch.cuda.empty_cache()
device = "cuda"


transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])


class FidelityLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to("cuda")
        vgg16.eval()
        self.vgg16 = vgg16.features
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.style_layers = [1, 6, 11, 18, 25]
        self.scale_factor = 1e-5

    def calc_features(self, im, target_layers=(18, 25)):
        x = self.normalize(im)
        feats = []
        for i, layer in enumerate(self.vgg16[: max(target_layers) + 1]):
            x = layer(x)
            if i in target_layers:
                feats.append(x.clone())
        return feats

    def calc_2_moments(self, x):
        _, c, w, h = x.shape
        x = x.reshape(1, c, w * h)  # b, c, n
        mu = x.mean(dim=-1, keepdim=True)  # b, c, 1
        cov = torch.matmul(x - mu, torch.transpose(x - mu, -1, -2))
        return mu, cov

    def matrix_diag(self, diagonal):
        N = diagonal.shape[-1]
        shape = diagonal.shape[:-1] + (N, N)
        device, dtype = diagonal.device, diagonal.dtype
        result = torch.zeros(shape, dtype=dtype, device=device)
        indices = torch.arange(result.numel(), device=device).reshape(shape)
        indices = indices.diagonal(dim1=-2, dim2=-1)
        result.view(-1)[indices] = diagonal
        return result

    def l2wass_dist(self, mean_stl, cov_stl, mean_synth, cov_synth):
        # Prevent ill-conditioned eigenvalue
        eps = torch.eye(cov_stl.shape[0], device=device) * 1e-5
        # Calculate tr_cov and root_cov from mean_stl and cov_stl
        eigvals, eigvects = torch.linalg.eigh(
            cov_stl + eps
        )  
        eigroot_mat = self.matrix_diag(torch.sqrt(eigvals.clip(0)))
        root_cov_stl = torch.matmul(
            torch.matmul(eigvects, eigroot_mat), torch.transpose(eigvects, -1, -2)
        )
        tr_cov_stl = torch.sum(eigvals.clip(0), dim=1, keepdim=True)

        #Find ill condition problem
        try:
            tr_cov_synth = torch.sum(
                torch.linalg.eigvalsh(cov_synth + eps).clip(0), dim=1, keepdim=True
            )
        except:
            print("cov_synth: ", cov_synth)
            print("cov_synth + eps: ", cov_synth + eps)
        
        
        
        mean_diff_squared = torch.mean((mean_synth - mean_stl) ** 2)
        cov_prod = torch.matmul(torch.matmul(root_cov_stl, cov_synth), root_cov_stl)
        var_overlap = torch.sum(
            torch.sqrt(torch.linalg.eigvalsh(cov_prod).clip(0.1)), dim=1, keepdim=True
        )  # .clip(0) meant errors getting eigvals
        dist = mean_diff_squared + tr_cov_stl + tr_cov_synth - 2 * var_overlap
        return dist

    def forward(self, input, target, mask=None):
        input_features = self.calc_features(input, self.style_layers)  # get features
        target_features = self.calc_features(target, self.style_layers)  # get features
        l = 0
        for x, y in zip(input_features, target_features):
            mean_synth, cov_synth = self.calc_2_moments(x)  # input mean and cov
            mean_stl, cov_stl = self.calc_2_moments(y)  # target mean and cov
            l += self.l2wass_dist(mean_stl, cov_stl, mean_synth, cov_synth)
        return l.mean() * self.scale_factor

class FeatureAttackingLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.scale_factor = 1e-5


    def calc_2_moments(self, x):
        c, w, h = x.shape
        x = x.reshape(1, c, w * h)  # b, c, n
        mu = x.mean(dim=-1, keepdim=True)  # b, c, 1
        cov = torch.matmul(x - mu, torch.transpose(x - mu, -1, -2))
        return mu, cov

    def matrix_diag(self, diagonal):
        N = diagonal.shape[-1]
        shape = diagonal.shape[:-1] + (N, N)
        device, dtype = diagonal.device, diagonal.dtype
        result = torch.zeros(shape, dtype=dtype, device=device)
        indices = torch.arange(result.numel(), device=device).reshape(shape)
        indices = indices.diagonal(dim1=-2, dim2=-1)
        result.view(-1)[indices] = diagonal
        return result

    def l2wass_dist(self, mean_stl, cov_stl, mean_synth, cov_synth):
        # Prevent ill-conditioned eigenvalue
        eps = torch.eye(cov_stl.shape[0], device=device) * 1e-5
        # Calculate tr_cov and root_cov from mean_stl and cov_stl
        eigvals, eigvects = torch.linalg.eigh(
            cov_stl + eps
        ) 
        eigroot_mat = self.matrix_diag(torch.sqrt(eigvals.clip(0)))
        root_cov_stl = torch.matmul(
            torch.matmul(eigvects, eigroot_mat), torch.transpose(eigvects, -1, -2)
        )
        tr_cov_stl = torch.sum(eigvals.clip(0), dim=1, keepdim=True)

        tr_cov_synth = torch.sum(
            torch.linalg.eigvalsh(cov_synth + eps).clip(0), dim=1, keepdim=True
        )
        mean_diff_squared = torch.mean((mean_synth - mean_stl) ** 2)
        cov_prod = torch.matmul(torch.matmul(root_cov_stl, cov_synth), root_cov_stl)
        var_overlap = torch.sum(
            torch.sqrt(torch.linalg.eigvalsh(cov_prod).clip(0.1)), dim=1, keepdim=True
        )  # .clip(0) meant errors getting eigvals
        dist = mean_diff_squared + tr_cov_stl + tr_cov_synth - 2 * var_overlap
        return dist

    def forward(self, clean_features, protected_features, mask=None):
        l = 0
        for x, y in zip(clean_features, protected_features):
            mean_synth, cov_synth = self.calc_2_moments(x)  # input mean and cov
            mean_stl, cov_stl = self.calc_2_moments(y)  # target mean and cov
            l += self.l2wass_dist(mean_stl, cov_stl, mean_synth, cov_synth)
        return l.mean() * self.scale_factor

class FeatureAttackingLossL2(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.l2_loss = torch.nn.MSELoss()

    def forward(self, clean_features, protected_features, mask=None):
        l = 0
        for x, y in zip(clean_features, protected_features):
            l += self.l2_loss(x, y)
        return l.mean()

# class SemanticLoss(nn.Module):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.l2_loss = torch.nn.MSELoss()
        
#     def forward(self, noise, output):
#         noise = noise.mean(0)
#         output = output.mean(0)
#         return self.l2_loss(noise, output)


def atk_pdm_protect_image(args, hparams=None):
    
    img_name = args.protected_image_path.split('/')[-1]
    img_name = img_name.split('.')[0]

    #Initiate experiment save folder
    print(f"\n Saving results to: {args.save_folder_path}\n" )
    if not os.path.exists(args.save_folder_path):
        os.makedirs(f"{args.save_folder_path}/protected_sample")
        os.makedirs(f"{args.save_folder_path}/clean_sample")
        os.makedirs(f"{args.save_folder_path}/protected_sample/feature_maps_across_t")
        os.makedirs(f"{args.save_folder_path}/clean_sample/feature_maps_across_t")
        os.makedirs(f"{args.save_folder_path}/protected_image")
        os.makedirs(f"{args.save_folder_path}/loss_plots")
        os.makedirs(f"{args.save_folder_path}/sample_attack_analysis")
        os.makedirs(f"{args.save_folder_path}/feature_maps/clean")
        os.makedirs(f"{args.save_folder_path}/feature_maps/protected")
    print("")
    
    #Initiate hparams config
    if hparams:
        feature_attacking_coef = hparams['feature_attacking_coef']
        fidelity_coef = hparams['fidelity_coef']
        fidelity_budget = hparams['fidelity_budget']
        fidelity_update_times_max = hparams['fidelity_update_times_max']
    #Default hparams
    else:
        feature_attacking_coef = -100 
        fidelity_coef = 70 
        fidelity_budget = 500
        fidelity_update_times_max = 10
            
    hparams_config = {
        "feature_attacking_coef": feature_attacking_coef,
        "fidelity_coef": fidelity_coef,
        "fidelity_budget": fidelity_budget,
        "fidelity_update_times_max": fidelity_update_times_max
    }
    
    
    args_dict = vars(args)
    args_dict = {k: v for k, v in args_dict.items() if v is not None}
    args_dict.update(hparams_config)
    
    with open(f"{args.save_folder_path}/params_config.json", 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)
            
            
    img_name = args.protected_image_path.split('/')[-1]
    img_name = img_name.split('.')[0]
    

    #Load victim-model-agnostic VAE for manifold preserving optimization
    if args.VAE_unet_size == 256:
        args.VAE_modei_id = "lambdalabs/miniSD-diffusers"
    elif args.VAE_unet_size == 512:
        args.VAE_modei_id = "runwayml/stable-diffusion-v1-5"
    vae = AutoencoderKL.from_pretrained(args.VAE_model_id, subfolder="vae")
    vae = vae.to("cuda")
    
    vae_input_shape = [args.batch_size, vae.config.in_channels, vae.config.sample_size, vae.config.sample_size]
    
    #Load victim model unet
    victim_model_pipeline = DiffusionPipeline.from_pretrained(args.victim_model_id)
    victim_model_pipeline.to(device)
    victim_unet = victim_model_pipeline.unet
    
    #Initiate Attack Validator
    attack_validator = AttackValidator(args.victim_model_id, victim_model_type=args.victim_model_type, save_folder_path=args.save_folder_path)
    
    
    record_features = []
    #Setting up hook function
    def obtain_output_feature(module, feature_in, feature_out):
        record_features.append(feature_out[0]) #feature_out
    
    for resnet in victim_unet.mid_block.resnets:
        resnet.register_forward_hook(obtain_output_feature)
    
    for up_block in victim_unet.up_blocks: 
        for resnet in up_block.resnets:
            resnet.register_forward_hook(obtain_output_feature)
    
    
    victim_scheduler = victim_model_pipeline.scheduler
    victim_unet.requires_grad_(False)
    
    victim_unet_input_shape = [args.batch_size, victim_unet.in_channels, victim_unet.sample_size, victim_unet.sample_size]
    

    #Define loss functions
    
    #For attacking
    feature_attacking_loss_fn = FeatureAttackingLoss()
    #For fidelity constraint
    fidelity_loss_fn = FidelityLoss()
   
    #Default step size
    step_size = args.step_size / 255
    
        
    #Load image to be protected 
    pil_image = Image.open(args.protected_image_path)
    image = transform(pil_image)
    image = repeat(image, 'c h w -> b c h w', b=args.batch_size).to(device)
    image = F.interpolate(image, size=vae_input_shape[-2:], mode='bilinear')
    
    # Protection mode selection
    if args.protection_mode == "vae":
        image = image + (torch.randn_like(image) * 2 * args.random_start_eps - args.random_start_eps)
        latent_raw = vae.encode(image).latent_dist.sample() * vae.config.scaling_factor
        protected_latent = latent_raw.clone().detach()
        protected_latent.requires_grad_(True)
        
        # Check if the VAE is reconstructing the image correctly
        recon_image = vae.decode(latent_raw /vae.config.scaling_factor).sample
        recon_image = to_pil(recon_image.detach())[0]
        recon_image.save(f"{args.save_folder_path}/clean_sample/vae_recon.png")
    elif args.protection_mode == "pixel":
        protected_image_optim = image.clone().detach()
        protected_image_optim.requires_grad_(True)
    else:
        raise NotImplementedError(f"{args.protection_mode} not implemented")
    
    
    #Prepare loss optimization recordings
    total_losses = []
    feature_attacking_losses = []
    fidelity_losses = []
    
    #Initiate best protected latent
    best_protected_latent = protected_latent

    
    # attack_validator should generate clean sample for the first time
    do_clean_sample = True
    
    # Optimization loop
    pbar = trange(args.optim_steps, position=0, leave=False, desc='Optimization Steps')
    for step in pbar:
        T = torch.randint(0, 500, (1,), dtype=int, device=device)
        
        victim_unet.requires_grad_(False)
        protected_latent.requires_grad_(True)
        
        # Sample noise for forward diffusion to timestep T
        noise = torch.randn(victim_unet_input_shape).to(device)
        
        # Clean image forward diffusion (add noise) to timestep T
        clean_image = F.interpolate(image, size=victim_unet_input_shape[-2:], mode='bilinear')
        clean_image = repeat(clean_image[0], 'c h w -> b c h w', b=args.batch_size).to(device)
        clean_x_T = victim_scheduler.add_noise(clean_image, noise, T)

        # Protected image forward diffusion (add noise) to timestep T
        if args.protection_mode == "vae":
            protected_image = vae.decode(protected_latent / vae.config.scaling_factor).sample
        elif args.protection_mode == "pixel":
            protected_image = protected_image_optim
        else:
            raise NotImplementedError(f"{args.protection_mode} not implemented")
            
        protected_image = F.interpolate(protected_image, size=victim_unet_input_shape[-2:], mode='bilinear')
        protected_image = repeat(protected_image[0], 'c h w -> b c h w', b=args.batch_size).to(device)
        protected_x_T = victim_scheduler.add_noise(protected_image, noise, T)

        # Prepare feature recordings of clean and protected images feeding to the victim_unet at timestep T
        record_features = []
        clean_model_output = victim_unet(clean_x_T, T).sample
        clean_record_features = [record_feature.clone() for record_feature in record_features]
        del record_features
        
        record_features = []
        model_output = victim_unet(protected_x_T, T).sample
        protected_record_features = [record_feature.clone() for record_feature in record_features]
        
        feature_attacking_loss = feature_attacking_coef * feature_attacking_loss_fn(clean_record_features[:3], protected_record_features[0:3])
        feature_attacking_loss.requires_grad_(True)
        
        
        fidelity_loss = fidelity_coef * fidelity_loss_fn(protected_image, clean_image)
        fidelity_loss.requires_grad_(True)
        
        # For optimization trend visualization only, not used for optimization
        total_loss = feature_attacking_loss + fidelity_loss
        
        fidelity_loss.retain_grad()
        fidelity_loss.backward(retain_graph=True)
        fidelity_grad = protected_latent.grad.clone()
        protected_latent.grad.zero_()
        
        feature_attacking_loss.backward()
        feature_attacking_grad = protected_latent.grad.clone()
        
        protected_latent.data = protected_latent.data - feature_attacking_grad.sign() * step_size 
        
        # Update protected latent to minimize the fidelity loss, here we limit the update times to avoid infinite loop
        update_times = 0
        while fidelity_loss > fidelity_budget and update_times < fidelity_update_times_max: 
            protected_latent.data = protected_latent.data - fidelity_grad * step_size
            
            # Decode protected latent to image for fidelity loss calculation
            protected_image = vae.decode(protected_latent / vae.config.scaling_factor).sample
            protected_image = F.interpolate(protected_image, size=victim_unet_input_shape[-2:], mode='bilinear')
            protected_image = repeat(protected_image[0], 'c h w -> b c h w', b=args.batch_size).to(device)
                
            fidelity_loss = fidelity_coef * fidelity_loss_fn(protected_image, clean_image)
            
            fidelity_loss.retain_grad()
            fidelity_loss.backward(retain_graph=True)
            fidelity_grad = protected_latent.grad.clone()
            protected_latent.grad.zero_()
            
            update_times+=1
        
        # Clamp the protected latent to the range of [-2, 2]
        protected_latent.data = protected_latent.data.clamp(-2, 2)
        protected_latent.grad = None
        
        total_losses.append(total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss)
        feature_attacking_losses.append(feature_attacking_loss.item())
        fidelity_losses.append(fidelity_loss.item())
        
        pbar.set_description(f"Total Loss : {total_loss:.10f}, Attacking Loss : {feature_attacking_loss:.10f}, Fidelity Loss : {fidelity_loss:.10f}")
        
        
        best_protected_latent = protected_latent
        
        if (step+1) % args.feature_maps_viz_period == 0:
            visualize_feature_maps(clean_record_features, f"{args.save_folder_path}/feature_maps/clean/clean_feature_maps_{step+1}.png", T, prefix="Clean")
            visualize_feature_maps(protected_record_features, f"{args.save_folder_path}/feature_maps/protected/protected_feature_maps_{step+1}.png", T, prefix="Protected")
        
        if (step+1) % args.validation_period == 0:
            best_protected_image = vae.decode(best_protected_latent / vae.config.scaling_factor).sample
            best_protected_image = F.interpolate(best_protected_image, size=victim_unet_input_shape[-2:], mode='bilinear')
            best_protected_pil_image = to_pil(best_protected_image.detach())[0]
        
            attack_validator.edit(args, best_protected_pil_image, step+1, sdedit_t=args.sdedit_t, do_clean_sample=do_clean_sample)
            # After the first validation, we do not need to generate clean sample, reduce validation time
            do_clean_sample = False
            
        
    best_protected_image = vae.decode(best_protected_latent / vae.config.scaling_factor).sample
    best_protected_pil_image = to_pil(best_protected_image.detach())[0]

    print("Final Losses:")
    print(f"Total Loss : {total_loss:.10f}, Attacking Loss : {feature_attacking_loss:.10f}, Fidelity Loss : {fidelity_loss:.10f}")
    #Plot loss curve
    print("Plotting Loss Graph...")
    x_axis = [i for i in range(len(feature_attacking_losses))]
    loss_plot_path = f"{args.save_folder_path}/loss_plots/loss.png"

    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(131)
    plt.plot(x_axis, total_losses, 'b')
    plt.title("Total Loss")
    ax2 = fig.add_subplot(132)
    plt.plot(x_axis, feature_attacking_losses, 'b')
    plt.title("Feature Attacking Loss")
    ax3 = fig.add_subplot(133)
    plt.plot(x_axis, fidelity_losses, 'b')
    plt.title("Fidelity Loss")
    plt.savefig(loss_plot_path)
    
    print("Getting final validation result...")
    attack_validator.edit(args, best_protected_pil_image, args.optim_steps, sdedit_t=args.sdedit_t, do_clean_sample=do_clean_sample)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protected_image_path', default="./to_protect_images", help="Path to image to be protected (optimized), recommended: ./to_protect_images/celeba or ./to_protect_images/cat")
    parser.add_argument('--save_folder_path', default="./test_output", help="Path to save protected results and analysis")
    parser.add_argument('--protection_mode', default="vae", help="what protection method is acheived", choices=["vae", "pixel"])
    parser.add_argument('--victim_model_type', default="unconditional_pdm", help="victim model type for attack validation", choices=["unconditional_pdm","ldm", "IF"])
    parser.add_argument('--random_start_eps', default=10e-5, help="Initial small perturbation added to image before optimization")
    parser.add_argument('--VAE_model_id', default="runwayml/stable-diffusion-v1-5", help="VAE id for manifold preserving optimizaiton")
    parser.add_argument('--VAE_unet_size', default=512, help="VAE's unet sample size", choices=[256, 512])
    parser.add_argument('--optim_steps', default=2, type=int, help="Number of optimization steps")
    parser.add_argument('--validation_period', default=100, type=int, help="validation period, for every n steps, perform an attack validation")
    parser.add_argument('--feature_maps_viz_period', default=10, type=int, help="feature maps visualization period, for every n steps, visualize the current recorded feature maps")
    parser.add_argument('--step_size', default=0.5, type=float, help="step size of each attack optimization")
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--attack_validator_batch_size', default=5, type=int, help="batch size for attack validator, we generate 5 varitions of the clean and protected images for attack validation in the paper")
    parser.add_argument('--victim_model_id', default="google/ddpm-cat-256", help='victim_model_id', choices=["google/ddpm-ema-celebahq-256", "google/ddpm-ema-church-256", "google/ddpm-cat-256"])
    parser.add_argument('--sdedit_t', default=100, help="sdedit_t forward diffusion timestep for attack validation")
    parser.add_argument('--enable_attack_analysis', default=True, help="Perform attack analysis over sampling steps")
    parser.add_argument('--hparams_config_path', help="specify hyper-parameters from json (optional)")
    
    args = parser.parse_args()
    
    if args.victim_model_id == "google/ddpm-ema-celebahq-256":
        args.dataset = "celeba"
    elif args.victim_model_id == "google/ddpm-ema-church-256":
        args.dataset = "church"
    elif args.victim_model_id == "google/ddpm-cat-256":
        args.dataset = "cat"
    i = 1
    args.protected_image_path = f"{args.protected_image_path}/{args.dataset}/{i}.png"
    atk_pdm_protect_image(args)
       
    
    