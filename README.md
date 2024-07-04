# ComfyUI_Pops
using pOpsPaper method  in ComfyUI

pOpsPaper method From: [pOpsPaper](https://github.com/pOpsPaper/pOps)
----

My ComfyUI node list：
-----

1、ParlerTTS node:[ComfyUI_ParlerTTS](https://github.com/smthemex/ComfyUI_ParlerTTS)     
2、Llama3_8B node:[ComfyUI_Llama3_8B](https://github.com/smthemex/ComfyUI_Llama3_8B)      
3、HiDiffusion node：[ComfyUI_HiDiffusion_Pro](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro)   
4、ID_Animator node： [ComfyUI_ID_Animator](https://github.com/smthemex/ComfyUI_ID_Animator)       
5、StoryDiffusion node：[ComfyUI_StoryDiffusion](https://github.com/smthemex/ComfyUI_StoryDiffusion)  
6、Pops node：[ComfyUI_Pops](https://github.com/smthemex/ComfyUI_Pops)   
7、stable-audio-open-1.0 node ：[ComfyUI_StableAudio_Open](https://github.com/smthemex/ComfyUI_StableAudio_Open)        
8、GLM4 node：[ComfyUI_ChatGLM_API](https://github.com/smthemex/ComfyUI_ChatGLM_API)   
9、CustomNet node：[ComfyUI_CustomNet](https://github.com/smthemex/ComfyUI_CustomNet)           
10、Pipeline_Tool node :[ComfyUI_Pipeline_Tool](https://github.com/smthemex/ComfyUI_Pipeline_Tool)    
11、Pic2Story node :[ComfyUI_Pic2Story](https://github.com/smthemex/ComfyUI_Pic2Story)   
12、PBR_Maker node:[ComfyUI_PBR_Maker](https://github.com/smthemex/ComfyUI_PBR_Maker)      
13、ComfyUI_Streamv2v_Plus node:[ComfyUI_Streamv2v_Plus](https://github.com/smthemex/ComfyUI_Streamv2v_Plus)   
14、ComfyUI_MS_Diffusion node:[ComfyUI_MS_Diffusion](https://github.com/smthemex/ComfyUI_MS_Diffusion)   

Tips
---
--本节点主要引用pOpsPaper的方法，可以达成材质迁移，背景迁移，融合，语义增强功能，以及这四种功能的组合，每次调用embedding节点，都会在output目录生成该次工作流创建的embedding文件，你可以在下次使用中，加载该embedding文件，从而达到树状组合的目的。   
--采样方式分为unet，ipadapter，controlnet+ipadapter，mean 四种，其中ipadapter，controlnet+ipadapter调用SDXL社区模型（比较耗时），unet和mean 调用kandinsky模型。   

---This node mainly references the methods of pOpsPaper, which can achieve material migration, background migration, fusion, semantic enhancement functions, and a combination of these four functions. Each time the embedding node is called, the embedding file created in the output directory will be generated. You can load the embedding file in the next use to achieve the goal of tree structure combination.   
---The sampling methods are divided into four types: unet, ipadapter, controllnet+ipadapter, and mean. Among them, ipadapter, controllnet+ipadapter call the SDXL community model (which is relatively time-consuming), while unet and mean call the Kandinsky model.   

1.Installation
-----
  In the ./ComfyUI /custom_node directory, run the following:   
  
  ``` python 
  git clone https://github.com/smthemex/ComfyUI_Pops.git
  ```
  
2.requirements  
----
  ``` python 
pip install -r requirements.txt
 ```
   
3 Need  models 
----
3.1  

默认的repo节点会自动联网下载所有必需的模型，如果你要使用离线模式，可以把模型先下载到comfyUI的diffuers目录下，这些就可以使用repo节点的菜单。  

当然，你的C盘足够大，或者你的huggingface的缓存已经切换到其他盘，建议用默认的下载就好。

可能会要求联外网，包括会下载一些clip文件，每个人的系统不同，不能一一例举。。

The default repo node will automatically connect to the network to download all necessary models  ！！
--
kandinsky-community/kandinsky-2-2-prior: [link](https://huggingface.co/kandinsky-community/kandinsky-2-2-prior)   

kandinsky-community/kandinsky-2-2-decoder: [link](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder)   

h94/IP-Adapter ip-adapter_sdxl.bin  [link](https://huggingface.co/h94/IP-Adapter)   

pOpsPaper/operators  （four models）     [link](https://huggingface.co/pOpsPaper/operators)  

----and---    
Menu 'checkpoint' choice any comfyUI or Web UI SDXL model   
Example：Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors    

Menu 'vae' choice any comfyUI or Web UI SDXL vae   
Example：madebyollin/sdxl-vae-fp16-fix or sdxl.vae.safetensors       

3.2 离线模式 offline   
在插件的weights目录下，如图放置ip-adapter_sdxl.bin，然后diffusers目录，放置其余3组模型。  

```   
├── ComfyUI/models/diffusers/
|      ├──kandinsky-community/
|             ├── kandinsky-2-2-decoder/
|                    ├── model_index.json 
|                    ├──unet/
|                        ├── config.json
|                        ├── diffusion_pytorch_model.safetensors
|                    ├──scheduler/
|                        ├── scheduler_config.json
|                    ├──movq/
|                        ├── config.json
|                        ├── diffusion_pytorch_model.safetensors
|             ├── kandinsky-2-2-prior/       
|                    ├── model_index.json 
|                    ├──tokenizer/
|                        ├── merges.txt
|                        ├── special_tokens_map.json
|                        ├── tokenizer_config.json
|                        ├── vocab.json
|                    ├──text_encoder/
|                        ├── config.json
|                        ├── model.safetensors
|                    ├──scheduler/
|                        ├── scheduler_config.json
|                    ├──prior/
|                        ├── config.json
|                        ├── diffusion_pytorch_model.safetensors  
|                    ├──image_processor/
|                        ├── preprocessor_config.json  
|                    ├──image_encoder/
|                        ├── config.json
|                        ├── model.safetensors
│
│      ├──pOpsPaper/operators/models/
|             ├── instruct/
|                    ├── learned_prior.pth
|             ├── scene/
|                    ├── learned_prior.pth
|             ├── texturing/
|                    ├── learned_prior.pth
|             ├── union/
|                    ├── learned_prior.pth
|                                                   
├── ComfyUI/custom_nodes/ComfyUI_Pops/
|      ├──weights/
|             ├── ip-adapter_sdxl.bin

```

如果使用离线模式，pOpsPaper/operators 这一栏，填写方式如下(后面的类似models/instructlearned_prior.pth路径切勿填写)：  
If using offline mode, fill in the pOpsPaper/operators column as follows (do not fill in the path similar to models/instructlearned_primary.pth):   

--Example--   
Your_local_model_path/pOpsPaper/operators     
---

4 Example
----
object2texture using controlnet  物体赋予材质，controlnet加ip采样。   
![](https://github.com/smthemex/ComfyUI_Pops/blob/main/example/controlnet.png)


object2texture  using IPadapter   物体赋予材质，  
![](https://github.com/smthemex/ComfyUI_Pops/blob/main/example/ipsampler.png)

use pt  使用插件生成的pt    
![](https://github.com/smthemex/ComfyUI_Pops/blob/main/example/use_pt.png)


mix pt   多PT混合   
![](https://github.com/smthemex/ComfyUI_Pops/blob/main/example/mix.png)


Citation
------

pOps
``` python  
https://popspaper.github.io/pOps

```
IP-Adapter
```
python  
@article{ye2023ip-adapter,
  title={IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models},
  author={Ye, Hu and Zhang, Jun and Liu, Sibo and Han, Xiao and Yang, Wei},
  booktitle={arXiv preprint arxiv:2308.06721},
  year={2023}



