# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import sys
import os
import torch
from PIL import Image
import numpy as np
import random
from transformers import  CLIPTokenizer,  CLIPVisionModelWithProjection, CLIPImageProcessor
from diffusers import  PriorTransformer, UNet2DConditionModel, KandinskyV22Pipeline
from .model import pops_utils
from .model.pipeline_pops import pOpsPipeline
import folder_paths
from comfy.utils import common_upscale

MAX_SEED = np.iinfo(np.int32).max
dir_path = os.path.dirname(os.path.abspath(__file__))
path_dir = os.path.dirname(dir_path)
file_path = os.path.dirname(path_dir)

device="cuda"
weight_dtype = torch.float16
output_dir = folder_paths.output_directory

def instance_path(path, repo):
    if repo == "":
        if path == "none":
            raise "you need fill repo_id or download model in diffusers directory "
        elif path != "none":
            model_path = get_local_path(file_path, path)
            repo = get_instance_path(model_path)
    return repo


paths = []
for search_path in folder_paths.get_folder_paths("diffusers"):
    if os.path.exists(search_path):
        for root, subdir, files in os.walk(search_path, followlinks=True):
            if "model_index.json" in files:
                paths.append(os.path.relpath(root, start=search_path))
if paths:
    paths = ["none"] + [x for x in paths if x]
else:
    paths = ["none", ]
    
    
def phi2narry(img):
    narry = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
    return narry

def narry_list(list_in):
    for i in range(len(list_in)):
        value = list_in[i]
        modified_value = phi2narry(value)
        list_in[i] = modified_value
    return list_in

def get_instance_path(path):
    instance_path = os.path.normpath(path)
    if sys.platform == 'win32':
        instance_path = instance_path.replace('\\', "/")
    return instance_path


def tensor_to_pil(tensor):
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image

def nomarl_upscale_topil(img_tensor, width, height):
    samples = img_tensor.movedim(-1, 1)
    img = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = img.movedim(1, -1)
    img_pil = tensor_to_pil(samples)
    return img_pil

def get_local_path(file_path, model_path):
    path = os.path.join(file_path, "models", "diffusers", model_path)
    model_path = os.path.normpath(path)
    if sys.platform == 'win32':
        model_path = model_path.replace('\\', "/")
    return model_path


def process_image(inputs_obj, clip,width, height):
    image_caption_suffix = ''
    if inputs_obj is not None and isinstance(inputs_obj, str):
        if inputs_obj.suffix == '.pth':
            image = torch.load(inputs_obj).image_embeds.to(device).to(weight_dtype)
            image_caption_suffix = '(embedding)'
        else:
            raise "input embeds_a must be a pth file"
    else:
        if isinstance(inputs_obj, Image.Image):
            image_pil =inputs_obj
        else:
            image_pil = Image.new('RGB', (height, width), (255, 255, 255))
        image = torch.Tensor(clip(image_pil)['pixel_values'][0]).to(device).unsqueeze(0).to(weight_dtype)
    return image, image_caption_suffix


def get_embedding(model,clip,inputs_obj, drop_condition_a, drop_condition_b,prior_seeds,prior_guidance_scale,prior_steps,height,width):
    filename_prefix = ''.join(random.choice("0123456789") for _ in range(4))
    should_drop_cond = [(drop_condition_a, drop_condition_b)]
    image_list, image_pil_list, caption_suffix_list=[],[],[]
    for input_path in inputs_obj:
        # Process both inputs
        image,caption_suffix = process_image(input_path, clip, width, height)
        image_list.append(image)
        caption_suffix_list.append(caption_suffix)
    
    image_list=[i for i in image_list if i is not None]
    input_image_embeds, input_hidden_state = pops_utils.preprocess(image_list[0], image_list[1],
                                                                   model.image_encoder,
                                                                   model.prior.clip_mean.detach(),
                                                                   model.prior.clip_std.detach(),
                                                                   should_drop_cond=should_drop_cond)

    captions = [f"objects{caption_suffix_list[0]}", f"textures{caption_suffix_list[1]}"]
    negative_input_embeds = torch.zeros_like(input_image_embeds)
    negative_hidden_states = torch.zeros_like(input_hidden_state)
    
    img_emb =model(input_embeds=input_image_embeds, input_hidden_states=input_hidden_state,
                             negative_input_embeds=negative_input_embeds,
                             negative_input_hidden_states=negative_hidden_states,
                             num_inference_steps=prior_steps,
                             num_images_per_prompt=1,
                             guidance_scale=prior_guidance_scale,
                             generator=torch.Generator(device=device).manual_seed(prior_seeds))
    
    
    img_emb_file = os.path.join(output_dir, f"infer_binary{filename_prefix}_s_{prior_seeds}_cfg_{prior_guidance_scale}_img_emb.pth")
    if not os.path.exists(img_emb_file):
        torch.save(img_emb, img_emb_file)
    positive_emb=img_emb.image_embeds
    negative_emb = img_emb.negative_image_embeds

    return positive_emb, negative_emb,input_hidden_state,img_emb_file


def get_embedding_instruct(model,clip,tokenizer, inputs_obj, texts,prior_guidance_scale, prior_seeds, prior_steps,height,width):
    #print(texts, type(texts))
    image_a, caption_suffix_a = process_image(inputs_obj,clip,width, height)
    text_inputs = tokenizer(text=texts,padding="max_length",max_length=tokenizer.model_max_length,truncation=True,return_tensors="pt",)
    mask = text_inputs.attention_mask.bool()  # [0]
    text_encoder_output = model.text_encoder(text_inputs.input_ids.to(device))
    text_encoder_hidden_states = text_encoder_output.last_hidden_state
    text_encoder_concat = text_encoder_hidden_states[:, :mask.sum().item()]
    #
    input_image_embeds, input_hidden_state = pops_utils.preprocess(image_a, None,
                                                                   model.image_encoder,
                                                                   model.prior.clip_mean.detach(),
                                                                   model.prior.clip_std.detach(),
                                                                   concat_hidden_states=text_encoder_concat)

    negative_input_embeds = torch.zeros_like(input_image_embeds)
    negative_hidden_states = torch.zeros_like(input_hidden_state)
    # for scale in prior_guidance_scale:
    img_emb = model(input_embeds=input_image_embeds, input_hidden_states=input_hidden_state,
                             negative_input_embeds=negative_input_embeds,
                             negative_input_hidden_states=negative_hidden_states,
                             num_inference_steps=prior_steps,
                             num_images_per_prompt=1,
                             guidance_scale=prior_guidance_scale,
                             generator=torch.Generator(device=device).manual_seed(prior_seeds))

    img_emb_file = os.path.join(output_dir, f"{texts}_s_{prior_seeds}_cfg_{prior_guidance_scale}_img_emb.pth")
    if not os.path.exists(img_emb_file):
        torch.save(img_emb, img_emb_file)
    positive_emb = img_emb.image_embeds
    negative_emb = img_emb.negative_image_embeds
    return positive_emb, negative_emb,input_hidden_state,img_emb_file


class Pops_Repo_Loader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "local_prior": (paths,),
                "prior_repo": ("STRING", {"default": "kandinsky-community/kandinsky-2-2-prior"}),
                "local_decoder": (paths,),
                "decoder_repo": ("STRING", {"default": "kandinsky-community/kandinsky-2-2-decoder"}),
                "pops_ckpt": (["none"] + folder_paths.get_filename_list("checkpoints"),),
                "function_type": (["Binary","instruct",],),
            }
        }

    RETURN_TYPES = ("MODEL","CLIP","VAE","MODEL",)
    RETURN_NAMES = ("model","clip","vae","tokenizer",)
    FUNCTION = "pops_repo_loader"
    CATEGORY = "Pops"

    def pops_repo_loader(self, local_prior, prior_repo, local_decoder, decoder_repo, pops_ckpt,function_type):
        
        kandinsky_prior_repo = instance_path(local_prior, prior_repo)
        kandinsky_decoder_repo = instance_path(local_decoder, decoder_repo)
        Pops_ckpt = folder_paths.get_full_path("checkpoints", pops_ckpt)
        #clip = CLIPImageProcessor.from_pretrained(kandinsky_prior_repo,subfolder='image_processor')
        clip = CLIPImageProcessor()
        prior = PriorTransformer.from_pretrained(kandinsky_prior_repo, subfolder="prior", use_safetensors=True)
        prior_state_dict = torch.load(Pops_ckpt, map_location=torch.device('cuda'))
        prior.load_state_dict(prior_state_dict, strict=False)
        prior.eval()
        prior = prior.to(weight_dtype)
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(kandinsky_prior_repo, subfolder='image_encoder',
                                                             torch_dtype=torch.float16).eval()
        # Freeze text_encoder and image_encoder
        image_encoder.requires_grad_(False)

        model = pOpsPipeline.from_pretrained(kandinsky_prior_repo,
                                                      prior=prior,
                                                      image_encoder=image_encoder,
                                                      torch_dtype=torch.float16).to(device)
        
        # Load decoder model f
        unet = UNet2DConditionModel.from_pretrained(kandinsky_decoder_repo,
                                                    subfolder='unet').to(torch.float16).to(device)
        vae = KandinskyV22Pipeline.from_pretrained(kandinsky_decoder_repo, unet=unet,
                                                   torch_dtype=torch.float16).to(device)
        
        if function_type=="instruct":
            tokenizer = CLIPTokenizer.from_pretrained(kandinsky_prior_repo, subfolder='tokenizer')
        else:
            tokenizer =None
        return (model,clip,vae,tokenizer,)


class Pops_Sampler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":("MODEL",),
                "clip": ("CLIP",),
                "texts": ("STRING",{"default": "smooth"}),
                "drop_condition_a": ("BOOLEAN", {"default": False},),
                "drop_condition_b": ("BOOLEAN", {"default": False},),
                "prior_guidance_scale": (
                    "FLOAT", {"default": 1.0, "min": 0.1, "max": 24.0, "step": 0.1, "round": False}),
                "seed": ("INT", {"default": 2, "min": 1, "max": MAX_SEED}),
                "prior_steps": ("INT", {"default": 25, "min": 1, "max": 4096}),
                "height": ("INT", {"default": 768, "min": 256, "max": 4096,"step": 64}),
                "width": ("INT", {"default": 768, "min": 256, "max": 4096,"step": 64}),
                "use_mean": ("BOOLEAN", {"default": False},),},
            "optional": {
                "tokenizer": ("MODEL",),
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "embeds_a": ("STRING", {"forceInput": True},),
                "embeds_b": ("STRING", {"forceInput": True},),
                }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING","STRING",)
    RETURN_NAMES = ("positive_emb", "ng_image_embeds","img_emb_file",)
    FUNCTION = "pops_sampler"
    CATEGORY = "Pops"

    def pops_sampler(self, model,clip,texts
                         , drop_condition_a,drop_condition_b,prior_guidance_scale,seed,prior_steps,height,width,use_mean,**kwargs):
        tokenizer=kwargs.get("tokenizer")
        image_a = kwargs.get("image_a")
        image_b = kwargs.get("image_b")
        if isinstance(image_a, torch.Tensor):
            image_a = nomarl_upscale_topil(image_a, width, height)
        if isinstance(image_b, torch.Tensor):
            image_b = nomarl_upscale_topil(image_b, width, height)
        embeds_a = kwargs.get("embeds_a")
        embeds_b = kwargs.get("embeds_b")
        inputs_obj=[image_a,image_b, embeds_a,embeds_b]
        inputs_obj = [i for i in inputs_obj if i is not None]
        if tokenizer:
            inputs_obj=inputs_obj[0]
            positive_emb, negative_emb,input_hidden_state,img_emb_file= get_embedding_instruct(model,tokenizer,clip, inputs_obj,texts,
                   prior_guidance_scale, seed, prior_steps,height,width)
        else:
            if len(inputs_obj)>2:
                inputs_obj = random.choices(inputs_obj,k=2)
            elif len(inputs_obj)<2:
                inputs_obj=inputs_obj.append([None])
            else:
                pass
            positive_emb, negative_emb,input_hidden_state,img_emb_file= get_embedding(model,clip,inputs_obj,drop_condition_a, drop_condition_b,seed,prior_guidance_scale,prior_steps,height,width)
        if use_mean:
            mean_emb = 0.5 * input_hidden_state[:, 0] + 0.5 * input_hidden_state[:, 1]
            mean_emb = (mean_emb * model.prior.clip_std) + model.prior.clip_mean
            negative_emb = model.get_zero_embed(mean_emb.shape[0], device=mean_emb.device)
            positive_emb=mean_emb
        return (positive_emb, negative_emb,img_emb_file,)


class Pops_Decoder:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "positive_emb": ("CONDITIONING",),
                "negative_emb": ("CONDITIONING",),
                "seed": ("INT", {"default": 2, "min": 1, "max": MAX_SEED}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 4096}),
                "guidance_scale": (
                    "FLOAT", {"default": 1.0, "min": 0.1, "max": 24.0, "step": 0.1, "round": False}),
                "height": ("INT", {"default": 768, "min": 256, "max": 4096, "step": 64}),
                "width": ("INT", {"default": 768, "min": 256, "max": 4096, "step": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "pops_decoder"
    CATEGORY = "Pops"

    def pops_decoder(self,vae,positive_emb,negative_emb,seed, steps,guidance_scale,height,width,):
        images = vae(image_embeds=positive_emb, negative_image_embeds=negative_emb,
                       num_inference_steps=steps, height=height,
                       width=width, guidance_scale=guidance_scale,
                       generator=torch.Generator(device=device).manual_seed(seed)).images
        images=phi2narry(images[0])
        return (images,)


NODE_CLASS_MAPPINGS = {
    "Pops_Repo_Loader": Pops_Repo_Loader,
    "Pops_Sampler":Pops_Sampler,
    "Pops_Decoder":Pops_Decoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Pops_Repo_Loader": "Pops_Repo_Loader",
    "Pops_Sampler":"Pops_Sampler",
    "Pops_Decoder":"Pops_Decode",
}
