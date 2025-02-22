import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torchvision.utils import save_image, make_grid

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import lpips
from einops import rearrange, repeat
from piq import LPIPS

import cv2
import random

import argparse
from PIL import Image
from tqdm import tqdm, trange

from transformers import ViTImageProcessor, ViTForImageClassification
from utils.measure_utils import psnr_, ssim_
from utils.clip_similarity import clip_sim

from diffusers import DDPMScheduler
from diffusers import UNet2DModel, UNet2DConditionModel, VQModel, AutoencoderKL
from diffusers import AutoPipelineForImage2Image, DDPMPipeline, DiffusionPipeline, StableDiffusionPipeline, IFImg2ImgPipeline, IFImg2ImgSuperResolutionPipeline

device="cuda"
os.environ['HF_HOME'] = '../HF_cache'

def single2uint(img):

    return np.uint8((img.clip(0, 1)*255.).round())

def uint2single(img):

    return np.float32(img/255.)

def add_JPEG_noise(img):
    # quality_factor = random.randint(95, 96)
    # img = cv2.cvtColor(single2uint(img), cv2.COLOR_RGB2BGR)
    result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 25])
    img = cv2.imdecode(encimg, 1)
    # img = cv2.cvtColor(uint2single(img), cv2.COLOR_BGR2RGB)
    return img

def crop_and_resize(img):
    width, height = img.size
 
    # Setting the points for cropped image
    left = 0.05 * width
    top = 0.05 * height
    right = 0.95 * width
    bottom = 0.95 * height
 
    # Cropped image of above dimension
    # (It will not change original image)
    img = img.crop((left, top, right, bottom))

    img = img.resize((width, height), Image.ANTIALIAS)

    return img

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



class UnconditionalDDPMPipeline(DiffusionPipeline):
    def __init__(self, model_id, save_folder_path):
        super().__init__()
        self.model_id = model_id
        self.model_id_name = model_id.replace("/","_")
        self.save_folder_path = save_folder_path
        self.transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
                                ])
        
    @torch.no_grad()
    def __call__(self, args, pil_image, curr_steps, is_protected=False, feature_maps_viz_freq = 0, sdedit_t=500, num_inference_steps=1000, is_defended=None) -> torch.Tensor:
        self.unet = UNet2DModel.from_pretrained(self.model_id).to(device)
        self.scheduler = DDPMScheduler.from_pretrained(self.model_id)
        self.register_modules(unet=self.unet, scheduler=self.scheduler)
        
        is_defended_tag = f"_is_defended_{is_defended}" if is_defended else ""
        
        self.record_features = []
        
        if feature_maps_viz_freq:
            def obtain_output_feature(module, feature_in, feature_out):
                self.record_features.append(feature_out[0]) #feature_out
            
            for resnet in self.unet.mid_block.resnets: #block.resnets:
                resnet.register_forward_hook(obtain_output_feature)
            
            for up_block in self.unet.up_blocks: 
                for resnet in up_block.resnets:
                    resnet.register_forward_hook(obtain_output_feature)
                    
        
        
        self.scheduler.set_timesteps(num_inference_steps)
        random_generator = torch.manual_seed(0)
        unet_input_shape = [args.attack_validator_batch_size, self.unet.config.in_channels, self.unet.config.sample_size, self.unet.config.sample_size]
        noise = torch.randn(unet_input_shape).to(device)
        
        T = torch.tensor(sdedit_t-1, dtype=int).to(device)
        start_index = int(self.scheduler.timesteps[0] - T)
        #Convert pil image to tensor with b, c, h, w
        image = self.transform(pil_image)
        image = repeat(image, 'c h w -> b c h w', b=args.attack_validator_batch_size).to(device)
        image = F.interpolate(image, size=unet_input_shape[-2:], mode='bilinear')
        
        #Forward diffusion (Add noise to target noise level T) 
        init = self.scheduler.add_noise(image, noise, T)
        image = init
        #Record unet out put at timestep 0
        unet_output_at_0 = None
        #Record intermediate latents and scores (model_output) for error analysis
        intermediate_scores = []
        intermediate_latents = []
        intermediate_pred_x0s = []
        for t in self.progress_bar(self.scheduler.timesteps[start_index:]):
            
            self.record_features = []
            
            model_output = self.unet(image, t).sample
            image = self.scheduler.step(model_output, t, image, generator=random_generator).prev_sample
            pred_x0 = self.scheduler.step(model_output, t, image, generator=random_generator).pred_original_sample
            
            model_output_cpu = model_output.clone().cpu()
            image_cpu = image.clone().cpu()
            pred_x0_cpu = pred_x0.clone().cpu()
            intermediate_scores.append(model_output_cpu)
            intermediate_latents.append(image_cpu)
            intermediate_pred_x0s.append(pred_x0_cpu)
            
            if (feature_maps_viz_freq!=0 and (t+1) % feature_maps_viz_freq == 0):
                processed = []
                for feature in self.record_features:
                    feature = feature.squeeze(0)
                    gray_scale = torch.sum(feature,0)
                    gray_scale = gray_scale / feature.shape[0]
                    processed.append(gray_scale.data.cpu().numpy())
                    
                fig = plt.figure(figsize=(30, 50))
                print("len(processed)",len(processed))
                for i in range(len(processed)):
                    a = fig.add_subplot(5, 4, i+1)
                    imgplot = plt.imshow(processed[i])
                    a.axis("off")
                    a.set_title(f"ResNet Feature Maps", fontsize=30)
                    
                if (t+1) % feature_maps_viz_freq == 0:
                    feature = self.record_features[-2]
                    feature = feature.squeeze(0)
                    gray_scale = torch.sum(feature,0)
                    gray_scale = gray_scale / feature.shape[0]
                    gray_scale = gray_scale.data.cpu().numpy()
                    
                    if is_protected:
                        plt.savefig(f'{self.save_folder_path}/protected_sample/feature_maps_across_t/protected_sample_feature_maps_t={t+1}.jpg', bbox_inches='tight')
                        plt.close(fig)
                        plt.imshow(gray_scale)
                        plt.savefig(f"{self.save_folder_path}/protected_sample/feature_maps_across_t/protected_sample_feature_maps_t={t+1}_last_2_layer.png")
                        save_image((intermediate_latents[sdedit_t - t - 1][1] / 2 + 0.5).clamp(0, 1), f"{self.save_folder_path}{is_defended_tag}/protected_sample/{self.model_id_name}_uncond_ddpm_{curr_steps}_sample_t={t+1}.png")
                    else:
                        plt.savefig(f'{self.save_folder_path}/clean_sample/feature_maps_across_t/clean_sample_feature_maps_t={t+1}.jpg', bbox_inches='tight')
                        plt.close(fig)
                        plt.imshow(gray_scale)
                        plt.savefig(f"{self.save_folder_path}/clean_sample/feature_maps_across_t/clean_sample_feature_maps_t={t+1}_last_2_layer.png")
                        save_image((intermediate_latents[sdedit_t - t - 1][1] / 2 + 0.5).clamp(0, 1), f"{self.save_folder_path}{is_defended_tag}/clean_sample/{self.model_id_name}_uncond_ddpm_{curr_steps}_sample_t={t+1}.png")
                #reset record_features to empty list for next recordings
            
            del self.record_features
            
        image = (image / 2 + 0.5).clamp(0, 1)
        
        
        if is_protected:
            save_image(make_grid(image, nrow=args.attack_validator_batch_size), f"{self.save_folder_path}{is_defended_tag}/protected_sample/{self.model_id_name}_uncond_ddpm_{curr_steps}.png")
        else:
            save_image(make_grid(image, nrow=args.attack_validator_batch_size), f"{self.save_folder_path}{is_defended_tag}/clean_sample/{self.model_id_name}_uncond_ddpm_{curr_steps}.png")
        return image, intermediate_scores, intermediate_latents, intermediate_pred_x0s
  
        
        
        

class AttackValidator():
    def __init__(self, model_id, victim_model_type, save_folder_path):
     
        self.model_id = model_id
        self.model_id_name = model_id.replace("/", "_")
        self.victim_model_type = victim_model_type
        
        self.save_folder_path = save_folder_path
        if not os.path.exists(save_folder_path):
            os.makedirs(f"{save_folder_path}/protected_sample")
            os.makedirs(f"{save_folder_path}/protected_sample/feature_maps_across_t")
            os.makedirs(f"{save_folder_path}/clean_sample")
            os.makedirs(f"{save_folder_path}/clean_sample/feature_maps_across_t")
            os.makedirs(f"{save_folder_path}/protected_image")
            os.makedirs(f"{save_folder_path}/loss_plots")
            os.makedirs(f"{save_folder_path}/sample_analysis")
       
        
        if self.victim_model_type == "unconditional_pdm":
            self.pipe = UnconditionalDDPMPipeline(self.model_id, self.save_folder_path)
        elif self.victim_model_type == "IF":
            self.pipe = IFImg2ImgPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
            self.sr_pipe = IFImg2ImgSuperResolutionPipeline.from_pretrained("DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16)
            self.pipe.enable_model_cpu_offload()
            self.sr_pipe.enable_model_cpu_offload()
        elif self.victim_model_type == "ldm":
            try:
                self.pipe = AutoPipelineForImage2Image.from_pretrained(self.model_id, requires_safety_checker=False, safety_checker=None).to(device) 
            except ValueError as e:
                raise NotImplementedError(f"{self.victim_model_type} is not implemented")
        else:
            raise NotImplementedError(f"{self.victim_model_type} is not implemented")
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])

        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.classifier = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        self.classifier.eval()

        self.evalute_metrics = {}
        
    
    def analyze_attack(self, args, curr_steps, scores=[], latents=[], pred_x0s=[]):
        score_mse_list = [F.mse_loss(clean , protected) for clean, protected in zip(scores[0], scores[1])]
        latent_mse_list = [F.mse_loss(clean , protected) for clean, protected in zip(latents[0], latents[1])]
        latent_cossim_list = [F.cosine_similarity(clean.view(clean.shape[0], -1), protected.view(protected.shape[0], -1)).mean() for clean, protected in zip(latents[0], latents[1])]
        pred_x0_cossim_list = [F.cosine_similarity(clean.view(clean.shape[0], -1), protected.view(protected.shape[0], -1)).mean() for clean, protected in zip(pred_x0s[0], pred_x0s[1])]
        pred_x0_mse_list = [F.mse_loss(clean , protected) for clean, protected in zip(pred_x0s[0], pred_x0s[1])]
        t_list = list(reversed(range(len(score_mse_list))))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t_list, y=score_mse_list, mode='lines', name='Score MSE'))
        fig.add_trace(go.Scatter(x=t_list, y=latent_mse_list, mode='lines', name='Latent MSE'))
        fig.add_trace(go.Scatter(x=t_list, y=pred_x0_mse_list, mode='lines', name='Pred x0 MSE'))
        fig.add_trace(go.Scatter(x=t_list, y=latent_cossim_list, mode='lines', name='Latent CosSim'))
        fig.add_trace(go.Scatter(x=t_list, y=pred_x0_cossim_list, mode='lines', name='Pred x0 CosSim'))
        fig.update_layout(xaxis=dict(autorange="reversed"))        
        fig.update_layout(title=f'{self.model_id_name}_uncond_ddpm_{curr_steps} across time',
                   xaxis_title='t',
                   yaxis_title='values')
        fig.write_image(f"{self.save_folder_path}/sample_attack_analysis/{self.model_id_name}_uncond_ddpm_{curr_steps}_across_time.png")
        
        
        
        
        
            
    
    
    def edit(self, args, best_protected_pil_image, curr_steps, is_defended=None, **kwargs):
        
        is_defended_tag = f"_is_defended_{is_defended}" if is_defended else ""
        
        #Save best protected pil image
        best_protected_pil_image.save(f"{self.save_folder_path}{is_defended_tag}/protected_image/{self.model_id_name}_{curr_steps}.png")
        
        #Open clean image for edit protection comparison
        clean_image = Image.open(args.protected_image_path)
        clean_image_save_name = args.protected_image_path.replace("/", "_")
        clean_image.save(f"{self.save_folder_path}{is_defended_tag}/clean_sample/original_image_from_{clean_image_save_name}.png")
        
        if self.victim_model_type == "ldm":
            
            self.clean_samples = self.pipe(kwargs["prompt"], image=clean_image, num_inference_steps=1000, strength=0.5, generator=torch.manual_seed(0)).images
            self.clean_samples[0].save(f"{self.save_folder_path}/clean_sample/{self.model_id_name}_auto_img2img_{curr_steps}.png")
            
            best_protected_pil_image = best_protected_pil_image.resize(clean_image.size)
            
            self.protected_samples = self.pipe(kwargs["prompt"], image=best_protected_pil_image, num_inference_steps=1000, strength=0.5, generator=torch.manual_seed(0)).images
            self.protected_samples[0].save(f"{self.save_folder_path}/protected_sample/{self.model_id_name}_auto_img2img_{curr_steps}.png")
            
        elif self.victim_model_type == "IF":
            
            prompt_embeds, negative_embeds = self.pipe.encode_prompt(kwargs["prompt"])
            clean_image = clean_image.resize((512, 512))
            self.clean_samples = self.pipe(prompt_embeds=prompt_embeds, image=clean_image, num_inference_steps=1000, strength=0.5, generator=torch.manual_seed(0)).images
            self.clean_samples_sr = self.sr_pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, image=self.clean_samples, original_image=clean_image).images
            self.clean_samples_sr[0].save(f"{self.save_folder_path}/clean_sample/{self.model_id_name}_IF_img2img_{curr_steps}.png")
            
            best_protected_pil_image = best_protected_pil_image.resize(clean_image.size)
            
            self.protected_samples = self.pipe(prompt_embeds=prompt_embeds, image=best_protected_pil_image, num_inference_steps=1000, strength=0.5, generator=torch.manual_seed(0)).images
            self.protected_samples_sr = self.sr_pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, image=self.protected_samples, original_image=best_protected_pil_image).images
            self.protected_samples_sr[0].save(f"{self.save_folder_path}/protected_sample/{self.model_id_name}_IF_img2img_{curr_steps}.png")
            
        elif self.victim_model_type == "unconditional_pdm":
            
            #If clean sample exists, skip
            if kwargs['do_clean_sample']:
                print("Generating clean sample for comparison")
                self.clean_samples, clean_intermediate_scores, clean_intermediate_latents, clean_intermediate_pred_x0s = self.pipe(args, clean_image, 0, is_protected=False, sdedit_t=kwargs["sdedit_t"], num_inference_steps=1000, is_defended=is_defended)
            else:
                print("Clean sample for comaprison had been generated, skip here...")
        
            #Protected edit
            print("Generating protected sample for comparison")
            self.protected_samples, protected_intermediate_scores, protected_intermediate_latents, protected_intermediate_pred_x0s = self.pipe(args, best_protected_pil_image, curr_steps, is_protected=True, sdedit_t=kwargs["sdedit_t"], num_inference_steps=1000, is_defended=is_defended)
        
        else:
            raise NotImplementedError(f"{self.victim_model_type} is not implemented")
        
        
        self.clean_classes = []
        with torch.no_grad():
            for edit_output in self.clean_samples:
                if torch.is_tensor(edit_output):
                    image = to_pil(edit_output.unsqueeze(0))[0]
                else:
                    image = edit_output
                inputs = self.processor(images=image, return_tensors="pt")
                outputs = self.classifier(**inputs)
                logits = outputs.logits
                predicted_class_idx = logits.argmax(-1).item()

                self.clean_classes.append(self.classifier.config.id2label[predicted_class_idx])
                
        self.pred_classes = []
        with torch.no_grad():
            for edit_output in self.protected_samples:
                if torch.is_tensor(edit_output):
                    image = to_pil(edit_output.unsqueeze(0))[0]
                else:
                    image = edit_output
                inputs = self.processor(images=image, return_tensors="pt")
                outputs = self.classifier(**inputs)
                logits = outputs.logits
                predicted_class_idx = logits.argmax(-1).item()

                self.pred_classes.append(self.classifier.config.id2label[predicted_class_idx])
                
        if args.victim_model_type != "ldm":
            clean_image = clean_image.resize([self.pipe.unet.config.sample_size, self.pipe.unet.config.sample_size])
            best_protected_pil_image = best_protected_pil_image.resize([self.pipe.unet.config.sample_size, self.pipe.unet.config.sample_size])
        else:
            best_protected_pil_image = best_protected_pil_image.resize(clean_image.size)
        
        x = self.transform(clean_image).to("cuda")
        x_adv = self.transform(best_protected_pil_image).to("cuda")
        
        print("x.shape: ", x.shape)
        print("x_adv.shape: ", x_adv.shape)
        
        
        edit_psnr_list = []
        edit_ssim_list = []
        edit_clip_list = []
        
        alex_percept = lpips.LPIPS(net='alex').to("cuda")
        clean_edits_tensor = torch.zeros([args.attack_validator_batch_size, ] + list(x_adv.shape[-3:]))
        adv_edits_tensor = torch.zeros([args.attack_validator_batch_size, ] + list(x_adv.shape[-3:]))                
        
        for i, edit in enumerate(zip(self.clean_samples, self.protected_samples)):
            
            if torch.is_tensor(edit_output):
                clean_edit = to_pil(edit[0].cpu().unsqueeze(0))[0]
                adv_edit = to_pil(edit[1].cpu().unsqueeze(0))[0]
            else:
                clean_edit = edit[0].resize(clean_image.size)
                adv_edit = edit[1].resize(clean_image.size)

            edit_ssim_list.append(ssim_(adv_edit, clean_edit))
            edit_psnr_list.append(psnr_(adv_edit, clean_edit))
            edit_clip_list.append(clip_sim(adv_edit, clean_edit, "cuda"))

            print("clean_edit ", clean_edit)
            print("clean_edit ", clean_edit.size)

            clean_edits_tensor[i] = self.transform(clean_edit)
            adv_edits_tensor[i] = self.transform(adv_edit)
        
        protected_lpips = alex_percept(x_adv[0].unsqueeze(0), x[0].unsqueeze(0), normalize=True).item()
        edited_lpips = alex_percept(adv_edits_tensor.to("cuda"), clean_edits_tensor.to("cuda"), normalize=True).cpu().detach().flatten().numpy()
        
        print(best_protected_pil_image.size, clean_image.size)
        print(f"Protected SSIM : {ssim_(best_protected_pil_image, clean_image)}")
        print(f"Protected PSNR : {psnr_(best_protected_pil_image, clean_image)}")
        print(f"Protected LPIPS : {protected_lpips}")
        print(f"Edit SSIM : {np.mean(edit_ssim_list)}, {edit_ssim_list}")
        print(f"Edit PSNR :  {np.mean(edit_psnr_list)}, {edit_psnr_list})")
        print(f"Edit LPIPS :  {np.mean(edited_lpips), edited_lpips}")
        print(f"Edit CLIP :  {np.mean(edit_clip_list)}, {edit_clip_list}")
        print(self.clean_classes, self.pred_classes) 


        self.evalute_metrics[curr_steps] = {
            "protected_ssim": ssim_(best_protected_pil_image, clean_image),
            "protected_psnr": psnr_(best_protected_pil_image, clean_image),
            "protected_lpips": protected_lpips,
            "edit_ssim": [np.mean(edit_ssim_list), str(edit_ssim_list)],
            "edit_psnr": [np.mean(edit_psnr_list), str(edit_psnr_list)],
            "edit_lpips": [np.mean(edited_lpips, dtype=float), str(edited_lpips)],
            "edit_clip_sim": [np.mean(edit_clip_list), str(edit_clip_list)],
            "clean_class": self.clean_classes,
            "pred_class": self.pred_classes,
        }

        sdedit_t = kwargs["sdedit_t"]
        with open(f"{self.save_folder_path}/evalute_metrics_{sdedit_t}{is_defended_tag}.json", "w") as json_file:
            json.dump(self.evalute_metrics, json_file, indent=4)
        
        #Perform sampling error analysis
        if args.enable_attack_analysis:
            self.analyze_attack(args, curr_steps, scores=[clean_intermediate_scores, protected_intermediate_scores], latents=[clean_intermediate_latents, protected_intermediate_latents], pred_x0s=[clean_intermediate_pred_x0s, protected_intermediate_pred_x0s])
        
    
        






        
        
        
        
    


    

