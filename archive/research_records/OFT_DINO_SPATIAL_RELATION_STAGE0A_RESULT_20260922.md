# Wrist-DINO 空间关系阶段 0a 结果核查

> **2026-09-22 来源审计更正：本次结果撤回，不可用于 Object 主线的机制判断。** Mask/RGB 来自 LIBERO-10 的历史 Phase-4B 数据，其采集配置为 `openvla-7b-oft-finetuned-libero-10`、`libero_10_no_noops`；脚本却加载 `openvla-7b-oft-finetuned-libero-object`、`libero_object_no_noops`。同编号任务语言也不同，例如 task 0 来源为“put both the alphabet soup and the tomato sauce in the basket”，Object bank 为“pick up the alphabet soup and place it in the basket”。下方数值仅保留作错误运行审计，不构成任何机制证据；原始 STATUS/CSV/日志保留，未篡改。代码现增加 checkpoint 与 unnorm_key 一致性保护，原配置在 CPU 预检即报 `SOURCE_MODEL_MISMATCH`。

日期：2026-09-22。作业：`135810`，NVIDIA RTX 3090，状态 `COMPLETE_FEASIBILITY_ONLY`。输入是历史 Phase-4B **LIBERO-10** 任务 0/1/2/4 共 240 帧，轨迹隔离的 discovery/validation/test 分别为每任务 30/10/20 帧。输出源文件为 `roboticAttack-main/outputs/exploring_oft_dino/spatial_relation_stage0a/{STATUS.json,TASK_READOUTS.csv,STATE_READOUTS.csv}`；本报告只核查已有输出，没有重跑 GPU，也没有提交后续作业。

## 数据质量与读数

10 帧缺少完整真值 mask，均来自任务 2（discovery 4、validation 2、test 4）；排除固定补丁区域及一圈 token 后，validation+test 有效 114/120 帧（95%），任务 2 为 24/30（80%），其他任务 100%。`STATE_READOUTS.csv` 共 470 行：有效源帧 230×完整/排除两种读数，加 10 行缺 mask 记录；`state_id×exclusion` 无重复。以下数值均为 validation+test、排除补丁 token 的中位数，单位为 16×16 DINO 网格的 token。

| 来源 LIBERO-10 任务编号 | 物体定位误差 | 夹爪定位误差 | RANGE 径向有符号变化 | LATERAL 相对变化范数 | 物体 oracle mask 被排除的中位比例 |
|---|---:|---:|---:|---:|---:|
| 0 | 1.054 | 0.891 | −0.0001 | 0.486 | 13.6% |
| 1 | 0.460 | 0.600 | +0.291 | 0.701 | 2.7% |
| 2 | 1.188 | 0.654 | +0.532 | 0.817 | 6.0% |
| 4 | **3.644** | 1.492 | +0.027 | 0.790 | **48.0%** |

总体物体/夹爪定位误差中位数为 1.207/0.824 token，达到配置的总体 ≤2 token 预筛门；程序按有符号值 `>0` 计，RANGE 有 3/4 个任务为正，也标记通过。但任务 4 的物体误差 >2 token，30 个留出帧中 25 个 >2；任务 0 的 RANGE 中位变化几乎为零，任务 4 的 +0.027 也很弱且逐帧正号比例仅 50%。因此形式上的 `pass_four_task_* = true` **只表示粗预筛代码门槛通过，不等于四任务一致的空间关系机制成立**。作为事后稳健性描述，若要求径向变化中位数 >0.1 token，仅任务 1、2 满足；此值不是预注册替代门槛。

任务 4 固定补丁排除区域覆盖了物体 oracle mask 的中位 48%，96.7% 的留出帧覆盖超过 20%；即使使用全部 token，其物体定位误差中位数也为 2.146 token，排除后升至 3.644。这个任务的读数失败可能同时受补丁位置遮挡与原型定位能力限制，不能归为单一原因。RANGE 的理想整图放大粗预测径向位移约为 `0.09×|物体−夹爪真值相对向量|`，任务 0/1/2/4 的中位数约 0.203/0.311/0.662/0.375 token；观察到的有符号中位位移分别为约 0/0.291/0.532/0.027。该粗预测不包含裁切、插值与 DINO 非等变性，**仅用于判断量级**。LATERAL 整图平移时，60.5% 的有效留出帧出现的相对读数变化范数还大于 RANGE；不能把 LATERAL 直接当作相对位置攻击目标。

## 结论与边界

此前关于任务 1、2 有机制线索、任务 0、4 不支持机制的表述全部撤回：来源模型和任务不一致，无法这样解释。阶段 0a 没有测完整动作、DINO 恢复后的动作、补丁扰动或闭环 CASR。现在**不进入 UADA+空间目标补丁训练**。下一步只能使用与 Object checkpoint、任务语言和当前 bank 匹配的图像及物体/夹爪标注，按轨迹留出并首先复核来源；之后再测 RANGE 空间变化与完整动作、DINO 恢复可消除动作分量的关系。后续 Slurm 提交仍由用户亲自执行。
