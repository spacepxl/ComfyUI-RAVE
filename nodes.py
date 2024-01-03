import torch
import os
import sys
import math
import copy
import numpy as np
from torchvision.utils import make_grid
from tqdm.auto import trange, tqdm

import comfy.sample
import comfy.utils
import latent_preview


def grid_compose(images, x_dim, random, rs, pad=0):
    
    grid_size = x_dim * x_dim
    batch_size = math.ceil(images.size(dim=0) / grid_size)
    
    shuffled_images = torch.zeros(batch_size * grid_size, images.size(1), images.size(2), images.size(3))
    if random:
        torch.manual_seed(rs)
        order = torch.randperm(batch_size * grid_size)
        order = torch.clamp(order, max=images.size(0) - 1)
        shuffled_images = images[order]
    else:
        shuffled_images[0:images.size(0)] = images
    
    batch_tensor = []
    
    for i in range(batch_size):
        offset = i * grid_size
        img_batch = shuffled_images[offset:offset+grid_size]
        
        grid = make_grid(img_batch.movedim(-1,1), nrow=x_dim, padding=pad).movedim(0,2)[None,]
        
        if pad > 0:
            grid = grid[:, pad:-pad, pad:-pad, :]
            
        batch_tensor.append(grid)
    
    batch_tensor = torch.cat(batch_tensor, 0)
    
    return batch_tensor


def grid_decompose(images, x_dim, random, rs, pad=0):
    
    grid_size = x_dim * x_dim
    batch_size = images.size(0) * grid_size
    
    padding = pad * (x_dim - 1)
    
    orig_w = int((images.size(1) - padding) / x_dim)
    orig_h = int((images.size(2) - padding) / x_dim)
    
    batch_tensor = []
    
    for i in range(images.size(0)):
        grid = images[i]
        
        for j in range (grid_size):
            w0 = int(math.floor(j / x_dim) * (orig_w + pad))
            h0 = int((j % x_dim) * orig_h) + ((j % x_dim) * pad)
            w1 = w0 + orig_w
            h1 = h0 + orig_h
            img = grid[w0:w1, h0:h1]
            
            batch_tensor.append(img[None,])
    
    t = torch.cat(batch_tensor, 0)
    
    if random:
        torch.manual_seed(rs)
        order = torch.randperm(batch_size)
        t[order] = t.clone()
    
    return t


def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    latent_image = latent["samples"]
    
    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    callback = latent_preview.prepare_callback(model, steps)
    #disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=True, seed=seed)
    out = latent.copy()
    out["samples"] = samples
    return (out, )


def calc_sigma(model, sampler_name, scheduler, steps, start_at_step, end_at_step):
    device = comfy.model_management.get_torch_device()
    end = min(steps, end_at_step)
    start = min(start_at_step, end)
    real_model = None
    comfy.model_management.load_model_gpu(model)
    real_model = model.model
    sampler = comfy.samplers.KSampler(real_model, steps=steps, device=device, sampler=sampler_name, scheduler=scheduler, denoise=1.0, model_options=model.model_options)
    sigmas = sampler.sigmas
    sigma = sigmas[start] - sigmas[end]
    sigma /= model.model.latent_format.scale_factor
    return sigma.cpu().numpy()


class KSamplerRAVE:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "grid_size": ("INT", {"default": 3, "min": 2, "max": 8}),
                    "pad_grid": ("BOOLEAN", {"default": False}),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "add_noise": ("BOOLEAN", {"default": False}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.1}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                     }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "RAVE"

    def sample(self, model, grid_size, pad_grid, noise_seed, add_noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step):
        latent = latent_image["samples"].clone()
        batch_length = latent.size(0)
        pad = 0
        if pad_grid:
            pad = 1
        
        print("RAVE sampling with %d frames (%d grids)" % (batch_length, math.ceil(batch_length / (grid_size ** 2))))
        
        # check pos and neg for controlnets
        controlnet_exist = False
        for conditioning in [positive, negative]:
            for t in conditioning:
                if 'control' in t[1]:
                    controlnet_exist = True
        
        # get list of controlnet objs and images
        control_objs = []
        control_images = []
        if controlnet_exist:
            for t in positive:
                control = t[1]['control']
                control_objs.append(control)
                control_images.append(control.cond_hint_original)
                
                prev = control.previous_controlnet
                while prev != None:
                    control_objs.append(prev)
                    control_images.append(prev.cond_hint_original)
                    prev = prev.previous_controlnet
        
        # add random noise if enabled
        if add_noise:
            noise = comfy.sample.prepare_noise(latent, noise_seed)
            sigma = calc_sigma(model, sampler_name, scheduler, steps, start_at_step, end_at_step)
            latent = latent + noise * sigma
        
        # iterate steps
        seed = noise_seed
        total_steps = min(steps, end_at_step) - start_at_step
        pbar = comfy.utils.ProgressBar(total_steps)
        for step in trange(total_steps, delay=1):
            # grid latents in random arrangement
            latent = grid_compose(latent.movedim(1,3), grid_size, True, seed, pad).movedim(-1,1)
            
            # grid controlnet images and apply
            if controlnet_exist:
                for i in range(len(control_objs)):
                    ctrl_img = grid_compose(control_images[i].movedim(1,3), grid_size, True, seed, pad*8).movedim(-1,1)
                    control_objs[i].set_cond_hint(ctrl_img, control_objs[i].strength, control_objs[i].timestep_percent_range)
            
            # sample 1 step
            start = start_at_step + step
            end = start + 1
            result = common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, {"samples":latent}, denoise=1.0, disable_noise=True, start_step=start, last_step=end, force_full_denoise=False)
            
            # ungrid latents and increment seed to shuffle grids with a different arrangement on the next step
            latent = grid_decompose(result[0]["samples"].movedim(1,3), grid_size, True, seed, pad).movedim(-1,1)
            seed += 1
            pbar.update(1)
        
        # restore original controlnet images (may cause issues if job is interrupted)
        if controlnet_exist:
            for i in range(len(control_objs)):
                control_objs[i].set_cond_hint(control_images[i], control_objs[i].strength, control_objs[i].timestep_percent_range)
        
        return ({"samples":latent[:batch_length]}, ) # slice latents to original batch length


class ImageGridCompose:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "images": ("IMAGE", ),
                    "x_dim": ("INT", {"default": 3, "min": 2, "max": 8}),
                    "pad_grid": ("BOOLEAN", {"default": False}),
                    "random": ("BOOLEAN", {"default": False}),
                    "rs": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    }
                }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "compose"
    
    CATEGORY = "RAVE/Image"
    
    def compose(self, images, x_dim, pad_grid, random, rs):
        pad = 0
        if pad_grid:
            pad = 1
        
        return (grid_compose(images, x_dim, random, rs, pad*8),)


class ImageGridDecompose:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "images": ("IMAGE", ),
                    "x_dim": ("INT", {"default": 3, "min": 2, "max": 8}),
                    "pad_grid": ("BOOLEAN", {"default": False}),
                    "random": ("BOOLEAN", {"default": False}),
                    "rs": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    }
                }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decompose"
    
    CATEGORY = "RAVE/Image"
    
    def decompose(self, images, x_dim, pad_grid, random, rs):
        pad = 0
        if pad_grid:
            pad = 1
        
        return (grid_decompose(images, x_dim, random, rs, pad*8),)


class LatentGridCompose:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "latents": ("LATENT", ),
                    "x_dim": ("INT", {"default": 3, "min": 2, "max": 8}),
                    "pad_grid": ("BOOLEAN", {"default": False}),
                    "random": ("BOOLEAN", {"default": False}),
                    "rs": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    }
                }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "compose"
    
    CATEGORY = "RAVE/Latent"
    
    def compose(self, latents, x_dim, pad_grid, random, rs):
        pad = 0
        if pad_grid:
            pad = 1
        
        t = grid_compose(latents["samples"].movedim(1,3), x_dim, random, rs, pad).movedim(-1,1)
        
        return ({"samples":t}, )


class LatentGridDecompose:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "latents": ("LATENT", ),
                    "x_dim": ("INT", {"default": 3, "min": 2, "max": 8}),
                    "pad_grid": ("BOOLEAN", {"default": False}),
                    "random": ("BOOLEAN", {"default": False}),
                    "rs": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    }
                }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "decompose"
    
    CATEGORY = "RAVE/Latent"
    
    def decompose(self, latents, x_dim, pad_grid, random, rs):
        pad = 0
        if pad_grid:
            pad = 1
        
        t = grid_decompose(latents["samples"].movedim(1,3), x_dim, random, rs, pad).movedim(-1,1)
        
        return ({"samples":t}, )


class ConditioningDebug:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", )}}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "debug"

    CATEGORY = "RAVE/debug"

    def debug(self, conditioning):
        control_objs = []
        control_images = []
        for t in conditioning:
            control = t[1]['control']
            control_objs.append(control)
            control_images.append(control.cond_hint_original)
            
            prev = control.previous_controlnet
            while prev != None:
                control_objs.append(prev)
                control_images.append(prev.cond_hint_original)
                prev = prev.previous_controlnet
        
        print("control_objs")
        for element in control_objs:
            print(element)
        print("control_images")
        for element in control_images:
            print(element.shape)
        
        return (conditioning, )


NODE_CLASS_MAPPINGS = {
    "KSamplerRAVE": KSamplerRAVE,
    "ImageGridCompose": ImageGridCompose,
    "ImageGridDecompose": ImageGridDecompose,
    "LatentGridCompose": LatentGridCompose,
    "LatentGridDecompose": LatentGridDecompose,
    # "ConditioningDebug": ConditioningDebug,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KSamplerRAVE": "KSampler (RAVE)",
    "ImageGridCompose": "ImageGridCompose",
    "ImageGridDecompose": "ImageGridDecompose",
    "LatentGridCompose": "LatentGridCompose",
    "LatentGridDecompose": "LatentGridDecompose",
    # "ConditioningDebug": "ConditioningDebug",
}
