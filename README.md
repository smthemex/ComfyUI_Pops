# ComfyUI_Pops
using pOpsPaper method  in ComfyUI

Notice
---
本节点主要引用pOpsPaper的方法，可以达成材质迁移，背景迁移，融合，语义增强功能，以及这四种功能的组合，每次调用embedding节点，都会在output目录生成该次工作流创建的embedding文件，你可以在下次使用中，加载该embedding文件，从而达到树状组合的目的。  

This node mainly references the methods of pOpsPaper, which can achieve material migration, background migration, fusion, semantic enhancement functions, and a combination of these four functions. Each time the embedding node is called, the embedding file created in the output directory will be generated. You can load the embedding file in the next use to achieve the goal of tree structure combination.   

pOpsPaper method From: [link](https://github.com/pOpsPaper/pOps)
----


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
Menu 'checkpoint' choice any comfyUI or Web UI SDXL model (example：Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors )   
Menu 'vae' choice any comfyUI or Web UI SDXL vae (example：sdxl.vae.safetensors )    

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


4 Example
----
object2texture  using Unet    物体赋予材质，Unet采样     
![](https://github.com/smthemex/ComfyUI_Pops/blob/main/example/example_unet.png)

embedding2scene  使用插件生成的embedding 加入到背景融合工作流中    
![](https://github.com/smthemex/ComfyUI_Pops/blob/main/example/emb.png)

object2texture using controlnet  物体赋予材质，controlnet加ip采样，示例底模和VAE没选好。   
![](https://github.com/smthemex/ComfyUI_Pops/blob/main/example/example_controlnet.png)

mix   多节点组合   
![](https://github.com/smthemex/ComfyUI_Pops/blob/main/example/textruring_scene_example.png)

using embedding     使用之前生成的embedding      
![](https://github.com/smthemex/ComfyUI_Pops/blob/main/example/usingemb.png)

using embedding and mean method   使用之前生成的embedding 和mean 方法   
![](https://github.com/smthemex/ComfyUI_Pops/blob/main/example/usingmean.png)


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



