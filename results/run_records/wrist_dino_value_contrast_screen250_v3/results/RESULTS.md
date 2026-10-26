# DINO Value与生成器对比学习：最后筛选：250轮闭环筛选

主表逐suite CASR；表中为百分比。不跨suite平均。
clean、UADA、Key、B1生成器、关系为匹配v1/v2原始记录只读复用400回合；三张新PNG新增240回合。

## 无补丁清洁对照（v1复用）

| Suite | 成功/回合 | SR% | clean-success任务数 |
|---|---:|---:|---:|
| libero_object | 20/20 | 100.00 | 10 |
| libero_spatial | 19/20 | 95.00 | 10 |
| libero_goal | 19/20 | 95.00 | 10 |
| libero_10 | 18/20 | 90.00 | 10 |

## 250轮

| 方法 | Object | Spatial | Goal | Long |
|---|---:|---:|---:|---:|
| uada_original | 5.00 | 25.00 | 5.00 | 85.00 |
| dino_key | 5.00 | 20.00 | 40.00 | 85.00 |
| advencoder_cos | 5.00 | 10.00 | 45.00 | 90.00 |
| dino_affinity | 5.00 | 35.00 | 35.00 | 80.00 |
| dino_value | 0.00 | 10.00 | 5.00 | 55.00 |
| advencoder_cos_b8 | 10.00 | 30.00 | 20.00 | 100.00 |
| advencoder_nce_b8 | 0.00 | 25.00 | 20.00 | 75.00 |

候选相对同outer次数UADA的配对差（B8非等图像计算，百分点）：

| 方法 | Suite | 差值pp | 正向task | 负向task | 持平task |
|---|---|---:|---:|---:|---:|
| dino_key | libero_object | +0.00 | 1 | 1 | 8 |
| advencoder_cos | libero_object | +0.00 | 1 | 1 | 8 |
| dino_affinity | libero_object | +0.00 | 1 | 1 | 8 |
| dino_value | libero_object | -5.00 | 0 | 1 | 9 |
| advencoder_cos_b8 | libero_object | +5.00 | 2 | 1 | 7 |
| advencoder_nce_b8 | libero_object | -5.00 | 0 | 1 | 9 |
| dino_key | libero_spatial | -5.00 | 0 | 1 | 9 |
| advencoder_cos | libero_spatial | -15.00 | 1 | 3 | 6 |
| dino_affinity | libero_spatial | +10.00 | 3 | 1 | 6 |
| dino_value | libero_spatial | -15.00 | 0 | 3 | 7 |
| advencoder_cos_b8 | libero_spatial | +5.00 | 2 | 1 | 7 |
| advencoder_nce_b8 | libero_spatial | +0.00 | 1 | 1 | 8 |
| dino_key | libero_goal | +35.00 | 4 | 0 | 6 |
| advencoder_cos | libero_goal | +40.00 | 6 | 0 | 4 |
| dino_affinity | libero_goal | +30.00 | 5 | 0 | 5 |
| dino_value | libero_goal | +0.00 | 0 | 0 | 10 |
| advencoder_cos_b8 | libero_goal | +15.00 | 3 | 0 | 7 |
| advencoder_nce_b8 | libero_goal | +15.00 | 3 | 0 | 7 |
| dino_key | libero_10 | +0.00 | 1 | 1 | 8 |
| advencoder_cos | libero_10 | +5.00 | 1 | 0 | 9 |
| dino_affinity | libero_10 | -5.00 | 1 | 1 | 8 |
| dino_value | libero_10 | -30.00 | 0 | 5 | 5 |
| advencoder_cos_b8 | libero_10 | +15.00 | 3 | 0 | 7 |
| advencoder_nce_b8 | libero_10 | -10.00 | 1 | 2 | 7 |

InfoNCE−cos-B8：相同图像batch/生成器/计算预算的目标比较

| Suite | 差值pp | 正向task | 负向task | 持平task |
|---|---:|---:|---:|---:|
| libero_object | -10.00 | 0 | 2 | 8 |
| libero_spatial | -5.00 | 0 | 1 | 9 |
| libero_goal | +0.00 | 1 | 1 | 8 |
| libero_10 | -25.00 | 0 | 3 | 7 |

数据来自官方Long train专家演示；Long来源内，其他三套件补丁迁移。
测试使用正常OFT和RGB PNG，无训练hook/生成器推理。单训练seed；40/41为已查看开发初态。
新生成器图像B8有更多计算；NCE和cos-B8互相等预算，不与旧B1宣称等计算。
每task仅两个初态，是低成本探索筛选，不等于最终验证；Long来源内。
任务成功率、配对基线差值/区间、每任务结果分别见SUMMARY、CONTRASTS、PER_TASK；不作跨suite平均。
