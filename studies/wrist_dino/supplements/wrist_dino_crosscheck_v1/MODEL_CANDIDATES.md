# 其他DINO策略候选（本轮只登记）

| 模型 | 在线DINO路径 | 公开资产 | 后续目的 |
|---|---|---|---|
| CogVLA | DINOv2+SigLIP及路由/压缩 | https://github.com/iLearn-Lab/NeurIPS25-CogVLA ；本地models/CogVLA已有四套件目录 | 检查路由与压缩后是否仍有DINO主导的传播 |
| TurboVLA | DINOv3 ViT-B、双相机、不同动作结构 | https://github.com/H-EmbodVis/TurboVLA ；https://huggingface.co/H-EmbodVis/TurboVLA | 检查DINO版本和非LLM中心策略 |
| Fast-VLA | DINOv2在线actor，冻结VLM任务记忆 | https://github.com/Garfield-kh/FastVLA ；https://huggingface.co/kehong/FastVLA-weights | 区分当前视觉输入与任务记忆的传播 |

当前没有下载、适配、运行这些模型。先完成Adapter-Pro、pi05及视频，后续另行决定。
