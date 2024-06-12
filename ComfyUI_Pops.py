# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import sys
import os
import torch
from PIL import Image
import numpy as np
import random
from pathlib import Path
from huggingface_hub import hf_hub_download
from transformers import  CLIPTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPImageProcessor
from diffusers import AutoencoderKL, PriorTransformer, UNet2DConditionModel, KandinskyV22Pipeline,StableDiffusionXLPipeline
from diffusers.loaders.ip_adapter import IPAdapterMixin
from .model import pops_utils
from .model.pipeline_pops import pOpsPipeline
from nodes import ImageScale
import folder_paths

MAX_SEED = np.iinfo(np.int32).max
dir_path = os.path.dirname(os.path.abspath(__file__))
path_dir = os.path.dirname(dir_path)
file_path = os.path.dirname(path_dir)


def instance_path(path, repo):
    if repo == "":
        if path == "none":
            raise "you need fill repo_id or download model in diffusers directory "
        elif path != "none":
            model_path = get_local_path(file_path, path)
            repo = get_instance_path(model_path)
    return repo

def function_type_choice(type_choice):
    if type_choice == "texturing":
        pop_pth_path = 'models/texturing/learned_prior.pth'
    elif type_choice == "instruct":
        pop_pth_path = 'models/instruct/learned_prior.pth'
    elif type_choice == "scene":
        pop_pth_path = 'models/scene/learned_prior.pth'
    else:
        pop_pth_path = 'models/union/learned_prior.pth'
    return pop_pth_path


paths = []
paths_a = []
for search_path in folder_paths.get_folder_paths("diffusers"):
    if os.path.exists(search_path):
        for root, subdir, files in os.walk(search_path, followlinks=True):
            if "model_index.json" in files:
                paths.append(os.path.relpath(root, start=search_path))
            if "config.json" in files:
                paths_a.append(os.path.relpath(root, start=search_path))
                paths_a = [z for z in paths_a if "controlnet-depth-sdxl-1.0" in z]

if paths != []:
    paths = ["none"] + [x for x in paths if x]
else:
    paths = ["none", ]


def phi2narry(img):
    img = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
    return img

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
    # tensor = tensor.cpu()
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image


def get_local_path(file_path, model_path):
    path = os.path.join(file_path, "models", "diffusers", model_path)
    model_path = os.path.normpath(path)
    if sys.platform == 'win32':
        model_path = model_path.replace('\\', "/")
    return model_path


def get_embedding(kandinsky_prior_repo, prior_repo,
                 type_choice, kandinsky_decoder_repo,inputs_a, inputs_b, drop_condition_a, drop_condition_b,prior_seeds,prior_guidance_scale,prior_steps,height,width):

    prior_path = function_type_choice(type_choice)
    weight_dtype = torch.float16
    device = 'cuda:0'
    #prior_seeds = [prior_seeds]
    prior_guidance_scale = [prior_guidance_scale]
    output_dir = folder_paths.output_directory
    x = "/"
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(kandinsky_prior_repo,
                                                                  subfolder='image_encoder',
                                                                  torch_dtype=weight_dtype).eval()
    if kandinsky_prior_repo.count(x) == 1: #online
        try:
            image_processor = CLIPImageProcessor.from_pretrained(kandinsky_prior_repo,
                                                                 subfolder='image_processor')
        except:
            path_pro = hf_hub_download(kandinsky_prior_repo, filename='image_processor/preprocessor_config.json')
            image_processor = CLIPImageProcessor.from_pretrained(path_pro)
        finally:
            pass
    else:
        try:
            image_processor = CLIPImageProcessor.from_pretrained(kandinsky_prior_repo,subfolder='image_processor')
        except:
            path_pro = hf_hub_download("kandinsky-community/kandinsky-2-2-prior",
                                       filename='image_processor/preprocessor_config.json')
            image_processor = CLIPImageProcessor.from_pretrained(path_pro)
        finally:
            pass

    prior = PriorTransformer.from_pretrained(
        kandinsky_prior_repo, subfolder="prior", use_safetensors=True
    )

    if prior_repo.count(x) == 1:
        # Load from huggingface
        prior_path = hf_hub_download(repo_id=prior_repo, filename=prior_path)
    else:
        prior_path = get_instance_path(os.path.join(prior_repo, prior_path))

    prior_state_dict = torch.load(prior_path, map_location=device)
    msg = prior.load_state_dict(prior_state_dict, strict=False)
    print(msg)
    prior.eval()
    # Freeze text_encoder and image_encoder
    image_encoder.requires_grad_(False)
    # Load full model for vis
    unet = UNet2DConditionModel.from_pretrained(kandinsky_decoder_repo,
                                                subfolder='unet').to(torch.float16).to(device)

    prior_pipeline = pOpsPipeline.from_pretrained(kandinsky_prior_repo,
                                                  prior=prior,
                                                  image_encoder=image_encoder,
                                                  torch_dtype=torch.float16)
    prior_pipeline = prior_pipeline.to(device)
    prior = prior.to(weight_dtype)
    decoder = KandinskyV22Pipeline.from_pretrained(kandinsky_decoder_repo, unet=unet,
                                                   torch_dtype=torch.float16)
    decoder = decoder.to(device)

    inputs_a = [Path(inputs_a)]  # 链接转为列表,加入函数
    inputs_b = [Path(inputs_b)]
    paths = [(input_a, input_b) for input_a in inputs_a for input_b in inputs_b]  # 将输入图片的路径变成list，路径组成元组
    # just so we have more variety to look at during the inference
    random.seed(42)
    random.shuffle(paths)
    for input_a_path, input_b_path in paths:
        def process_image(input_path):
            image_caption_suffix = ''
            if input_path is not None and input_path.suffix == '.pth':  # 判断结尾是否是pth文件
                image = torch.load(input_path).image_embeds.to(device).to(weight_dtype)
                embs_unnormed = image
                zero_embeds = prior_pipeline.get_zero_embed(embs_unnormed.shape[0], device=embs_unnormed.device)
                direct_from_emb = decoder(image_embeds=embs_unnormed, negative_image_embeds=zero_embeds,
                                          num_inference_steps=50, height=height,
                                          width=width, guidance_scale=4,
                                          generator=torch.Generator(device=device).manual_seed(0)).images
                image_pil = direct_from_emb[0]
                image_caption_suffix = '(embedding)'
            else:  # 加入的是图片文件链接，加载图片方法
                if input_path is not None:
                    image_pil = Image.open(input_path).convert("RGB").resize((height, width))
                else:
                    image_pil = Image.new('RGB', (height, width), (255, 255, 255))

                image = torch.Tensor(image_processor(image_pil)['pixel_values'][0]).to(device).unsqueeze(0).to(
                    weight_dtype)
            return image, image_pil, image_caption_suffix
        # Process both inputs
        image_a, image_pil_a, caption_suffix_a = process_image(input_a_path)
        image_b, image_pil_b, caption_suffix_b = process_image(input_b_path)
        should_drop_cond = [(drop_condition_a, drop_condition_b)]
        input_image_embeds, input_hidden_state = pops_utils.preprocess(image_a, image_b,
                                                                       image_encoder,
                                                                       prior.clip_mean.detach(),
                                                                       prior.clip_std.detach(),
                                                                       should_drop_cond=should_drop_cond)

        captions = [f"objects{caption_suffix_a}", f"textures{caption_suffix_b}"]
        out_name = f"{input_a_path.stem if input_a_path is not None else ''}_{input_b_path.stem if input_b_path is not None else ''}"

        negative_input_embeds = torch.zeros_like(input_image_embeds)
        negative_hidden_states = torch.zeros_like(input_hidden_state)
        for scale in prior_guidance_scale:
            img_emb = prior_pipeline(input_embeds=input_image_embeds, input_hidden_states=input_hidden_state,
                                     negative_input_embeds=negative_input_embeds,
                                     negative_input_hidden_states=negative_hidden_states,
                                     num_inference_steps=prior_steps,
                                     num_images_per_prompt=1,
                                     guidance_scale=scale,
                                     generator=torch.Generator(device=device).manual_seed(prior_seeds))

            img_emb_file = os.path.join(output_dir, f"{out_name}_s_{prior_seeds}_cfg_{scale}_img_emb.pth")
            torch.save(img_emb, img_emb_file)
            positive_emb=img_emb.image_embeds
            negative_emb = img_emb.negative_image_embeds

    return decoder,prior,prior_pipeline,positive_emb, negative_emb,input_hidden_state,img_emb_file


def get_embedding_instruct(type_choice, kandinsky_prior_repo, prior_repo, kandinsky_decoder_repo, inputs_a, texts,
                   prior_guidance_scale, prior_seeds, prior_steps,height,width):
    prior_path = function_type_choice(type_choice)
    output_dir = folder_paths.output_directory
    weight_dtype = torch.float16
    device = 'cuda:0'
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(kandinsky_prior_repo,
                                                                  subfolder='image_encoder',
                                                                  torch_dtype=weight_dtype).eval()
    x = "/"
    if kandinsky_prior_repo.count(x) == 1:  # online
        try:
            image_processor = CLIPImageProcessor.from_pretrained(kandinsky_prior_repo,
                                                                 subfolder='image_processor')
        except:
            path_pro = hf_hub_download(kandinsky_prior_repo, filename='image_processor/preprocessor_config.json')
            image_processor = CLIPImageProcessor.from_pretrained(path_pro)
        finally:
            pass
    else:
        try:
            image_processor = CLIPImageProcessor.from_pretrained(kandinsky_prior_repo, subfolder='image_processor')
        except:
            path_pro = hf_hub_download("kandinsky-community/kandinsky-2-2-prior",
                                       filename='image_processor/preprocessor_config.json')
            image_processor = CLIPImageProcessor.from_pretrained(path_pro)
        finally:
            pass
    tokenizer = CLIPTokenizer.from_pretrained(kandinsky_prior_repo, subfolder='tokenizer')
    text_encoder = CLIPTextModelWithProjection.from_pretrained(kandinsky_prior_repo,
                                                               subfolder='text_encoder',
                                                               torch_dtype=weight_dtype).eval().to(device)
    prior = PriorTransformer.from_pretrained(
        kandinsky_prior_repo, subfolder="prior"
    )
    if prior_repo.count(x) == 1:
        # Load from huggingface
        prior_path = hf_hub_download(repo_id=prior_repo, filename=prior_path)
    else:
        prior_path = get_instance_path(os.path.join(prior_repo, prior_path))

    prior_state_dict = torch.load(prior_path, map_location=device)
    msg = prior.load_state_dict(prior_state_dict, strict=False)
    print(msg)
    prior.eval()
    # Freeze text_encoder and image_encoder
    image_encoder.requires_grad_(False)
    # Load full model for vis
    unet = UNet2DConditionModel.from_pretrained(kandinsky_decoder_repo,
                                                subfolder='unet').to(torch.float16).to(device)
    prior_pipeline = pOpsPipeline.from_pretrained(kandinsky_prior_repo,
                                                  prior=prior,
                                                  image_encoder=image_encoder,
                                                  torch_dtype=torch.float16)
    prior_pipeline = prior_pipeline.to(device)
    prior = prior.to(weight_dtype)
    decoder = KandinskyV22Pipeline.from_pretrained(kandinsky_decoder_repo, unet=unet,
                                                   torch_dtype=torch.float16)
    decoder = decoder.to(device)
    inputs_a = [Path(inputs_a)]
    paths = [(input_a, text) for input_a in inputs_a for text in texts]

    # just so we have more variety to look at during the inference
    random.shuffle(paths)

    for input_a_path, text in paths:
        # for input_a_path, text in tqdm(input_path):
        def process_image(input_path):
            image_caption_suffix = ''
            if input_path is not None and input_path.suffix == '.pth':
                image = torch.load(input_path).image_embeds.to(device).to(weight_dtype)
                embs_unnormed = (image * prior.clip_std) + prior.clip_mean
                zero_embeds = prior_pipeline.get_zero_embed(embs_unnormed.shape[0], device=embs_unnormed.device)
                direct_from_emb = decoder(image_embeds=embs_unnormed, negative_image_embeds=zero_embeds,
                                          num_inference_steps=50, height=height,
                                          width=width, guidance_scale=4).images
                image_pil = direct_from_emb[0]
                image_caption_suffix = '(embedding)'
            else:
                if input_path is not None:
                    image_pil = Image.open(input_path).convert("RGB").resize((height, width))
                else:
                    image_pil = Image.new('RGB', (height, width), (255, 255, 255))

                image = torch.Tensor(image_processor(image_pil)['pixel_values'][0]).to(device).unsqueeze(0).to(
                    weight_dtype)
            return image, image_pil, image_caption_suffix

        # Process both inputs
        image_a, image_pil_a, caption_suffix_a = process_image(input_a_path)

        text_inputs = tokenizer(
            text,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        mask = text_inputs.attention_mask.bool()  # [0]
        text_encoder_output = text_encoder(text_inputs.input_ids.to(device))
        text_encoder_hidden_states = text_encoder_output.last_hidden_state
        text_encoder_concat = text_encoder_hidden_states[:, :mask.sum().item()]
        #
        input_image_embeds, input_hidden_state = pops_utils.preprocess(image_a, None,
                                                                       image_encoder,
                                                                       prior.clip_mean.detach(),
                                                                       prior.clip_std.detach(),
                                                                       concat_hidden_states=text_encoder_concat)

        out_name = f"{input_a_path.stem if input_a_path is not None else ''}_{text}"

        negative_input_embeds = torch.zeros_like(input_image_embeds)
        negative_hidden_states = torch.zeros_like(input_hidden_state)
        for scale in prior_guidance_scale:
            img_emb = prior_pipeline(input_embeds=input_image_embeds, input_hidden_states=input_hidden_state,
                                     negative_input_embeds=negative_input_embeds,
                                     negative_input_hidden_states=negative_hidden_states,
                                     num_inference_steps=prior_steps,
                                     num_images_per_prompt=1,
                                     guidance_scale=scale,
                                     generator=torch.Generator(device=device).manual_seed(prior_seeds))

            img_emb_file = os.path.join(output_dir, f"{out_name}_s_{prior_seeds}_cfg_{scale}_img_emb.pth")
            torch.save(img_emb, img_emb_file)
            positive_emb = img_emb.image_embeds
            negative_emb = img_emb.negative_image_embeds

    return decoder,prior,prior_pipeline,positive_emb, negative_emb,input_hidden_state,img_emb_file


class Pops_Repo_Choice:
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
                "Pops_repo": ("STRING", {"default": "pOpsPaper/operators"})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("repo_id",)
    FUNCTION = "pops_repo_choice"
    CATEGORY = "Pops"

    def pops_repo_choice(self, local_prior, prior_repo, local_decoder, decoder_repo, Pops_repo
                         ):
        kandinsky_prior_repo = instance_path(local_prior, prior_repo)
        kandinsky_decoder_repo = instance_path(local_decoder, decoder_repo)
        get_Pops_repo = get_instance_path(Pops_repo)
        repo_id = str(";".join([kandinsky_prior_repo, kandinsky_decoder_repo, get_Pops_repo]))
        return (repo_id,)


class Pops_Prior_Embedding:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo_id": ("STRING", {"forceInput": True}),
                "texts": (['shiny', 'enormous', 'aged'],),
                "inputs_a": ("STRING", {"forceInput": True}),
                "inputs_b": ("STRING", {"forceInput": True}),
                "drop_condition_a": ("BOOLEAN", {"default": False},),
                "drop_condition_b": ("BOOLEAN", {"default": False},),
                "prior_guidance_scale": (
                    "FLOAT", {"default": 1.0, "min": 0.1, "max": 24.0, "step": 0.1, "round": False}),
                "seed": ("INT", {"default": 2, "min": 1, "max": MAX_SEED}),
                "prior_steps": ("INT", {"default": 25, "min": 1, "max": 4096}),
                "function_type": (["texturing", "scene", "union", "instruct", ],),
                "height": ("INT", {"default": 1024, "min": 256, "max": 4096,"step": 64}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096,"step": 64})
            }
        }

    RETURN_TYPES = ("MODEL","MODEL","MODEL","CONDITIONING", "CONDITIONING","CONDITIONING","STRING",)
    RETURN_NAMES = ("model","prior","prior_pipeline","positive", "negative","input_hidden_state","img_emb_file",)
    FUNCTION = "apply_condition"
    CATEGORY = "Pops"

    def apply_condition(self, repo_id, texts, inputs_a,inputs_b
                         , drop_condition_a,drop_condition_b,prior_guidance_scale,seed,prior_steps,function_type,height,width):
        kandinsky_prior_repo, kandinsky_decoder_repo, prior_repo = repo_id.split(";")
        if function_type == "instruct":
            decoder,prior,prior_pipeline,positive, negative,input_hidden_state,img_emb_file= get_embedding_instruct(function_type, kandinsky_prior_repo, prior_repo, kandinsky_decoder_repo, inputs_a, texts,
                   prior_guidance_scale, seed, prior_steps,height,width)
        else:
            decoder,prior,prior_pipeline,positive, negative,input_hidden_state,img_emb_file= get_embedding(kandinsky_prior_repo, prior_repo,
                 function_type, kandinsky_decoder_repo,  inputs_a, inputs_b, drop_condition_a, drop_condition_b,seed,prior_guidance_scale,prior_steps,height,width)

        return (decoder,prior,prior_pipeline,positive, negative,input_hidden_state,img_emb_file)


class Pops_Unet_Sampler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "seed": ("INT", {"default": 2, "min": 1, "max": MAX_SEED}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 4096}),
                "guidance_scale": (
                    "FLOAT", {"default": 1.0, "min": 0.1, "max": 24.0, "step": 0.1, "round": False}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "pops_unet_sampler"
    CATEGORY = "Pops"

    def pops_unet_sampler(self,model, positive,negative, seed, steps,guidance_scale,height,width):
        device="cuda"
        images = model(image_embeds=positive, negative_image_embeds=negative,
                         num_inference_steps=steps, height=height,
                         width=width, guidance_scale=guidance_scale,
                         generator=torch.Generator(device=device).manual_seed(seed)).images
        images=phi2narry(images[0])
        return (images,)


class Pops_Ipadapter_Sampler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "checkpoints": (folder_paths.get_filename_list("checkpoints"),),
                "seed": ("INT", {"default": 2, "min": 1, "max": MAX_SEED}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 4096}),
                "guidance_scale": (
                    "FLOAT", {"default": 1.0, "min": 0.1, "max": 24.0, "step": 0.1, "round": False}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "pops_adapter_sampler"
    CATEGORY = "Pops"

    def pops_adapter_sampler(self,positive,checkpoints,seed,steps,guidance_scale,height,width):
        base_diffuser = get_instance_path(folder_paths.get_full_path("checkpoints", checkpoints))
        device="cuda"
        ipadapter_model = get_instance_path(os.path.join(dir_path, "weights", "ip-adapter_sdxl.bin"))
        original_config_file = get_instance_path(os.path.join(dir_path, "weights", "config", "sd_xl_base.yaml"))
        ip_pipeline = StableDiffusionXLPipeline.from_single_file(base_diffuser,
                                                                 original_config_file=original_config_file,
                                                                 torch_dtype=torch.float16)
        if os.path.exists(ipadapter_model):
            ip_pipeline.load_ip_adapter(ipadapter_model,
                                        subfolder="",
                                        weight_name="ip-adapter_sdxl.bin", image_encoder_folder=None)
        else:
            ip_pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin",
                                        image_encoder_folder=None)
        ip_pipeline.to(device)
        ip_pipeline.set_ip_adapter_scale(guidance_scale)
        images = ip_pipeline(prompt="", ip_adapter_image_embeds=[
            torch.stack([torch.zeros_like(positive), positive])],
                             negative_prompt="deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality",
                             num_inference_steps=steps,
                             height=height,
                             width=width,
                             generator=torch.Generator(device="cuda").manual_seed(seed),
                             ).images
        images=phi2narry(images[0])
        return (images,)


class Pops_Controlnet_Sampler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", ),
                "vae": (folder_paths.get_filename_list("vae"),),
                "positive": ("CONDITIONING",),
                "control_net": ("CONTROL_NET",),
                "checkpoints": (folder_paths.get_filename_list("checkpoints"),),
                "seed": ("INT", {"default": 2, "min": 1, "max": MAX_SEED}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 4096}),
                "guidance_scale": (
                    "FLOAT", {"default": 1.0, "min": 0.1, "max": 24.0, "step": 0.1, "round": False}),
                "controlnet_scale": (
                    "FLOAT", {"default": 0.5, "min": 0.1, "max": 24.0, "step": 0.1, "round": False}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "pops_controlnet_sampler"
    CATEGORY = "Pops"

    def pops_controlnet_sampler(self,image,vae,positive,control_net,checkpoints,seed,steps,guidance_scale,controlnet_scale,height,width):
        ss = ImageScale()
        image = ss.upscale(image, "lanczos", height, width, "center")[0]
        base_diffuser = get_instance_path(folder_paths.get_full_path("checkpoints", checkpoints))
        vae=get_instance_path(folder_paths.get_full_path("vae", vae))
        device = "cuda"
        ipadapter_model = get_instance_path(os.path.join(dir_path, "weights", "ip-adapter_sdxl.bin"))
        original_config_file = get_instance_path(os.path.join(dir_path, "weights", "config", "sd_xl_base.yaml"))
        config_file=get_instance_path(os.path.join(file_path,"models","configs","v1-inference.yaml"))
        vae = AutoencoderKL.from_single_file(vae, config_file=config_file,torch_dtype=torch.float16).to("cuda")
        pipeline = StableDiffusionXLPipeline.from_single_file(base_diffuser,
                                                                 original_config_file=original_config_file,
                                                                 controlnet=control_net,
                                                                 vae=vae,
                                                                 variant="fp16",
                                                                 use_safetensors=True,
                                                                 torch_dtype=torch.float16)
        pipeline.enable_xformers_memory_efficient_attention()
        pipeline.enable_vae_tiling()

        # 加载ip
        if os.path.exists(ipadapter_model):
            pipeline.load_ip_adapter(ipadapter_model,
                                        subfolder="",
                                        weight_name="ip-adapter_sdxl.bin", image_encoder_folder=None)
        else:
            pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin",
                                        image_encoder_folder=None)
        pipeline.to(device)
        pipeline.set_ip_adapter_scale(guidance_scale)
        pipeline.enable_model_cpu_offload()

        images = pipeline(
            prompt="",
            image=image,
            ip_adapter_image_embeds=[
                torch.stack([torch.zeros_like(positive), positive])],
            negative_prompt="",
            num_inference_steps=steps,
            controlnet_conditioning_scale=controlnet_scale,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).images
        images=phi2narry(images[0])
        return (images,)


class Pops_Mean_Sampler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "prior": ("MODEL",),
                "prior_pipeline": ("MODEL",),
                "input_hidden_state": ("CONDITIONING",),
                "seed": ("INT", {"default": 2, "min": 1, "max": MAX_SEED}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 4096}),
                "guidance_scale": (
                    "FLOAT", {"default": 4.0, "min": 0.1, "max": 50.0, "step": 0.1, "round": False}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "pops_mean_sampler"
    CATEGORY = "Pops"

    def pops_mean_sampler(self,model,prior,prior_pipeline,input_hidden_state,seed,steps,guidance_scale,height,width):
        device = "cuda"
        mean_emb = 0.5 * input_hidden_state[:, 0] + 0.5 * input_hidden_state[:, 1]
        mean_emb = (mean_emb * prior.clip_std) + prior.clip_mean
        del prior
        zero_embeds = prior_pipeline.get_zero_embed(mean_emb.shape[0], device=mean_emb.device)
        del prior_pipeline
        images = model(image_embeds=mean_emb, negative_image_embeds=zero_embeds,
                                  num_inference_steps=steps, height=height,
                                  width=width, guidance_scale=guidance_scale,
                                  generator=torch.Generator(device=device).manual_seed(seed)).images

        images=phi2narry(images[0])
        return (images,)

class Imgae_To_Path:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "image_operator": ("IMAGE",),
                "height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64})
            }
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("inputs_a", "inputs_b",)
    FUNCTION = "img2path"
    CATEGORY = "Pops"

    def main_img2path(self, img, index, filename_prefix,height,width):
        s = ImageScale()
        img = s.upscale(img, "lanczos", height, width, "center")[0]
        image_name = f"{index}_{filename_prefix}.png"
        temp_dir = get_instance_path(os.path.join(folder_paths.output_directory, image_name))
        # print(temp_dir)
        img = tensor_to_pil(img)
        img.save(temp_dir)
        return temp_dir

    def img2path(self, image, image_operator,height,width):
        filename_prefix = ''.join(random.choice("0123456789") for _ in range(5))
        index_a = "a"
        inputs_a = self.main_img2path(image, index_a, filename_prefix,height,width)
        index_b = "b"
        inputs_b = self.main_img2path(image_operator, index_b, filename_prefix,height,width)
        return inputs_a, inputs_b



NODE_CLASS_MAPPINGS = {
    "Pops_Repo_Choice": Pops_Repo_Choice,
    "Pops_Prior_Embedding":Pops_Prior_Embedding,
    "Pops_Unet_Sampler":Pops_Unet_Sampler,
    "Pops_Ipadapter_Sampler":Pops_Ipadapter_Sampler,
    "Pops_Controlnet_Sampler":Pops_Controlnet_Sampler,
    "Pops_Mean_Sampler":Pops_Mean_Sampler,
    "Imgae_To_Path": Imgae_To_Path,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Pops_Repo_Choice": "Pops_Repo_Choice",
    "Pops_Prior_Embedding":"Pops_Prior_Embedding",
    "Pops_Unet_Sampler":"Pops_Unet_Sampler",
    "Pops_Ipadapter_Sampler":"Pops_Ipadapter_Sampler",
    "Pops_Controlnet_Sampler":"Pops_Controlnet_Sampler",
    "Pops_Mean_Sampler":"Pops_Mean_Sampler",
    "Imgae_To_Path": "Imgae_To_Path",
}
