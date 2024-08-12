# ComfyUI_Pops
using pOpsPaper method  in ComfyUI

pOps: Photo-Inspired Diffusion Operators
pOpsPaper method From: [pOpsPaper](https://github.com/pOpsPaper/pOps)
----

Update
---
-- 暂时去掉SDXL板块/Temporarily remove the SDXL sector  
-- 改了代码结构，运行很快，但是抽卡的几率太大了。 
--后期直接连入其他工作流，待完成。


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
3.1  online   

The default repo node will automatically connect to the network to download all necessary models  ！！
--
kandinsky-community/kandinsky-2-2-prior: [link](https://huggingface.co/kandinsky-community/kandinsky-2-2-prior)   

kandinsky-community/kandinsky-2-2-decoder: [link](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder)   

pOpsPaper/operators  （four models）     [link](https://huggingface.co/pOpsPaper/operators)  
  

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
├── ComfyUI/models/diffusers/
|      ├──kandinsky-community/
|            ├── kandinsky-2-2-prior/       
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
├──ComfyUI/models/checkpoints
|             ├── instruct_learned_prior.pth  (rename from instruct/learned_prior.pth )
|             ├── scene_learned_prior.pth  (rename from scene/learned_prior.pth )
|             ├── texturing_learned_prior.pth  (rename from texturing/learned_prior.pth )
|             ├── union_learned_prior.pth  (rename from union/learned_prior.pth )

```


4 Example
----
object2texture   物体材质模型     
![](https://github.com/smthemex/ComfyUI_Pops/blob/main/example/texture.png))

object scene    物体场景模型，   
![](https://github.com/smthemex/ComfyUI_Pops/blob/main/example/scene.png)

drop cond 删掉某个条件    物体场景模型，   
![](https://github.com/smthemex/ComfyUI_Pops/blob/main/example/drop_cond.png)


5 My ComfyUI node list：
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
15、ComfyUI_AnyDoor node: [ComfyUI_AnyDoor](https://github.com/smthemex/ComfyUI_AnyDoor)  
16、ComfyUI_Stable_Makeup node: [ComfyUI_Stable_Makeup](https://github.com/smthemex/ComfyUI_Stable_Makeup)  
17、ComfyUI_EchoMimic node:  [ComfyUI_EchoMimic](https://github.com/smthemex/ComfyUI_EchoMimic)   
18、ComfyUI_FollowYourEmoji node: [ComfyUI_FollowYourEmoji](https://github.com/smthemex/ComfyUI_FollowYourEmoji)   
19、ComfyUI_Diffree node: [ComfyUI_Diffree](https://github.com/smthemex/ComfyUI_Diffree)    
20、ComfyUI_FoleyCrafter node: [ComfyUI_FoleyCrafter](https://github.com/smthemex/ComfyUI_FoleyCrafter)


6 Citation
------

pOps
``` python  
@article{richardson2024pops,
  title={pOps: Photo-Inspired Diffusion Operators},
  author={Richardson, Elad and Alaluf, Yuval and Mahdavi-Amiri, Ali and Cohen-Or, Daniel},
  journal={arXiv preprint arXiv:2406.01300},
  year={2024}
}

```
IP-Adapter
```
python  
@article{ye2023ip-adapter,
  title={IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models},
  author={Ye, Hu and Zhang, Jun and Liu, Sibo and Han, Xiao and Yang, Wei},
  booktitle={arXiv preprint arxiv:2308.06721},
  year={2023}



