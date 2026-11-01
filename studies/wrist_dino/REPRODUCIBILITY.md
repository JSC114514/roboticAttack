# 阅读、复算与实际运行

## CPU结果复算

仓库提供每条件/起点一行的CSV、原始汇总、冻结配置、最终PNG、作图源码和来源哈希。`summarize_results.py`只使用Python标准库，可在不安装模型环境时重算各PNG的任务成功数和任务/套件等权CASR。

```bash
python studies/wrist_dino/summarize_results.py --study main --scope full
python studies/wrist_dino/summarize_results.py --study factorial --scope full
python studies/wrist_dino/summarize_results.py --study sensitivity --scope holdout
python studies/wrist_dino/summarize_results.py --study crosscheck --scope transfer --model pi05
```

`main`、`factorial`、`sensitivity`中的历史对照可能复用相同父阶段结果；CSV保留阶段成员身份，不把总行数称作新增实验数。开发起点可能复用于full，种子重复相同物理起点。机制恢复与历史动作回放不进入原生攻击排名。PNG文件哈希和RGB像素哈希不可混用。

## 科学源码与路径

`joint_repro/`及补充目录是已完成运行的注释快照；Python可执行AST与原始源码一致。`configs/`保存原始冻结配置，里面的机器路径是历史来源字段，不是要求其他机器创建同名用户目录。

通用入口以`--run-root`指定一个已准备的、可写的独立工作区。该目录按相应实验布局提供`configs/`、`data/`和需要的真实父阶段产物；入口将科学模块的ROOT定位到此目录，使用当前Python解释器，不调用集群调度器。不要把发布目录当作运行目录。

实际环境版本记录在[ENVIRONMENT.json](configs/ENVIRONMENT.json)。

工作区配置中需要绑定：`project`为本仓库根目录，`model_dir`为联合OFT，`expert_root`为原始专家数据，`libero_root`为LIBERO源码根目录，`python`为当前环境解释器。数值超参、mask、EOT与预算沿用本次冻结文件。可参考[paths.example.json](configs/paths.example.json)。外部资产不随仓库发布。

```bash
python studies/wrist_dino/run_experiment.py --help
# 以下操作使用你准备的工作区；不会申请GPU或提交Slurm。
python studies/wrist_dino/run_experiment.py main audit --run-root /path/to/prepared/main
python studies/wrist_dino/run_experiment.py main build-bank --run-root /path/to/prepared/main
python studies/wrist_dino/run_experiment.py main preflight --run-root /path/to/prepared/main
python studies/wrist_dino/run_experiment.py main train --run-root /path/to/prepared/main --id B01 --seed 42
python studies/wrist_dino/run_experiment.py main eval --run-root /path/to/prepared/main --id B01 --seed 42 --scope full
```

OFT运行环境沿用原PyTorch/Transformers/timm/TF/MuJoCo组合；版本记录见环境配置。需要单卡可用、正确的EGL设置及模型官方模块。联合模型必须有匹配revision的下载验证信息、300k头和四套件统计。首次复现先准备数据并通过原生/EOT/双分支梯度预检，再训练。

## 父实验依赖

- 四格消融只新增D07，复用父实验B01/O03/B04及clean。它需要父配置、数据、预检、校准与真实完成记录，不能用发布汇总JSON代替这些资产。
- 参数消融保留原单变量矩阵、默认对照、样本次序、EOT和完整梯度；需要相同候选池、初始化、父阶段预检及lambda校准。
- 原始`PARENT_SOURCES`/收据检查是运行身份检查。原机器上可使用已配置的工作区；新机器重新生成运行产物后应建立对应本次运行的来源绑定，保留原实验来源清单。不能把发布注释版hash伪装成旧运行hash，或删除检查让缺失资产通过。

```bash
python studies/wrist_dino/run_experiment.py factorial train --run-root /path/to/prepared/factorial --id D07 --seed 42
python studies/wrist_dino/run_experiment.py sensitivity train --run-root /path/to/prepared/sensitivity --id LR004-O03 --seed 42 --stop 500
```

## 跨模型

Adapter使用其原生作者仓库、四套件Pro checkpoint及v1动作头；π0.5使用OpenPI原生JAX环境与pi05_libero checkpoint。冻结文件记录模型文件SHA、初态SHA、PNG和80个起点。

准备工作区时需要将外部路径绑定到本地实际资产，并核对原始字节哈希。mtime是原机器的快速检查缓存，新机器必须在字节核验后生成自己的运行身份，不能只复制旧mtime。评测仍保留严格首动作/重置及双分支恢复检查。

```bash
# 在各目标原生环境和已分配的兼容单GPU内执行；本入口不负责提交资源。
python studies/wrist_dino/run_experiment.py crosscheck eval --run-root /path/to/prepared/crosscheck --model vla_adapter_pro --suite libero_goal --mode transfer
python studies/wrist_dino/run_experiment.py crosscheck eval --run-root /path/to/prepared/crosscheck --model pi05 --suite libero_goal --mode transfer
```

本次发布核验范围为源码/AST、配置、CSV统计、PNG字节和导航；未在新机器重新执行GPU训练。集群提交、资源轮询、后台监控、视频、权重、候选缓存与优化器均未打包。
