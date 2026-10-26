# 联合OFT：22张混合池旧补丁原生复测

固定官方联合Spatial/Object/Goal/Long checkpoint（revision在config.json），加载已合并权重、300000步动作头和本体投影器。复测旧PNG，不重训、不更新历史实验。各套件使用联合模型dataset_statistics.json中对应的`*_no_noops`动作和本体统计；每个GPU进程只加载一次模型。

## 手动入口

在项目根目录运行：

```bash
bash research_runs/wrist_joint_checkpoint_patch_retest_v1/run.sh --download
bash research_runs/wrist_joint_checkpoint_patch_retest_v1/run.sh --check
bash research_runs/wrist_joint_checkpoint_patch_retest_v1/run.sh --prepare
bash research_runs/wrist_joint_checkpoint_patch_retest_v1/run.sh --submit --workers 4
bash research_runs/wrist_joint_checkpoint_patch_retest_v1/run.sh --status --slurm
bash research_runs/wrist_joint_checkpoint_patch_retest_v1/run.sh --watch --slurm
bash research_runs/wrist_joint_checkpoint_patch_retest_v1/run.sh --resume --workers 4
```

下载只占CPU/网络，不升级依赖；中断后重复`--download`续传。`--submit`及`--resume`是唯一会执行sbatch的入口。重复调用发现本实验已有排队/运行任务则不重复提交。已核验完成的回合直接复用；失败attempt保留，续跑只补没有完成收据的回合。代码和资产来源变更会触发校验失败。

`--assets`仅核对旧PNG和初态，不要求模型下载完成。`--dry-run --workers 4`打印依赖链，不调用Slurm。`--report`及`--export-videos`可手动生成当前完整配对的阶段汇总/视频。确认下一批的配置保存到data/CONFIRMATION_PREPARED.json，入口不会提交它。

## 样本与口径

26条件=22张已训练共享PNG+两个初始化+历史Spatial UADA500+clean。四套件各10任务、初态2，共40起点、1040回合。每个条件都评测全部起点，包括clean失败起点。补丁50×50、腕部位置[x,y]=[26,137]、无EOT，原生双视角DINO/SigLIP、8步执行、warmup10。上限Spatial220/Object280/Goal300/Long520。种子20261020+10000×套件序号+100×任务编号+初态，套件顺序见config.json。

完整26条件及源模型、数据量、预算、初始化、PNG哈希见data/ASSETS.json。每条训练系列分别对照，不把不同训练池的分数差写作公平方法优势。正确/置乱关系、XYZ/全部动作、物理命令远边界是核心10张补丁。旧几何动作教师使用已预定的配对20000/19000候选检查点；不是按新验证结果选优。

先1张4090完成40个clean和接口检查；全部成功退出后，补丁数组最多4张4090，每worker4CPU/12G，按套件步数预算平衡任务分片。空闲依赖不占GPU；CPU汇总、WebM导出2CPU/4G。单GPU任务12小时限时是容错上限，不是运行时长。估计4张2.5–5小时、2张5–10小时，不含下载、排队和视频CPU导出；reports/PROGRESS每30秒更新实测吞吐/ETA，reports/COSTS保留每次失败和资源峰值。

启动检查核对300000组件唯一性、四套件归一化、单CUDA驻留无offload、冻结eval、官方动作接口误差≤1e-5、双分支双视角调用、零修改恒等。每条件新建环境，同种子同初态；物理数组、RGB、本体状态、相机时钟及首个clean动作严格配对。异常立即退出，不放宽阈值或自动重试。GPU模拟检查在用户提交后执行；CPU检查不会在登录节点导入MuJoCo或加载模型。

官方联合快照的modeling代码与本地版本只在未启用的扩散动作函数中相差一行scheduler初始化。加载器通过完整AST比较确认L1路径相同，仅允许这一处差异；强制L1、禁用diffusion/FiLM，不覆盖下载模型或历史文件。其他模型代码变化均拒绝加载，比较哈希记录到MODEL_PROVENANCE.json。

## 交付

results/CASR.csv：各套件任务成功率、CASR、缺失数和区间；等权套件均值。CASR分母是clean成功且该方法已完成的配对起点。按任务→起点重采样，不把帧/动作槽位当样本。当前每任务仅一个起点，跨初态信息在后续初态3批次增加。

results/PAIRED_CONTRASTS.csv：正确/置乱关系与各自对照的同起点差异。MODEL_COMPARISON.csv：与历史Goal模型同起点比较，攻击差值限定两模型共同clean成功且状态/RGB一致的起点。固定PNG跨模型部署不等同于在联合模型重训。

视频从原生控制步直接编码、前视和腕部分开，无文字栏/拼接。集中路径：`research_videos/wrist_joint_checkpoint_patch_retest_v1_singleview/MP4/方法/{front,wrist}/`和`IDE_WebM/方法/{front,wrist}/`。文件名含任务、初态、task_success。MP4为H264/yuv420p/faststart；WebM为VP8，CPU逐文件全解码核验维度、帧率和时长。GPU任务结束即释放，视频另由CPU继续。
