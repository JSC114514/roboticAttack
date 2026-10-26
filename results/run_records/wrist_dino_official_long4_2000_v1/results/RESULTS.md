# 官方Long四路从零2000轮结果

只报告逐suite CASR；表中为百分比。

## 1000轮

| 方法 | Object | Spatial | Goal | Long |
|---|---:|---:|---:|---:|
| uada_original | 19.22 | 25.78 | 56.00 | 98.89 |
| uada_dino_freq | 22.67 | 29.00 | 36.19 | 98.89 |
| tma_xyz | 8.11 | 29.22 | 41.31 | 100.00 |
| tma_xyz_dino | 12.00 | 23.78 | 33.56 | 95.00 |

## 2000轮

| 方法 | Object | Spatial | Goal | Long |
|---|---:|---:|---:|---:|
| uada_original | 16.00 | 26.89 | 48.42 | 100.00 |
| uada_dino_freq | 21.56 | 31.00 | 34.31 | 98.89 |
| tma_xyz | 7.22 | 33.33 | 40.44 | 100.00 |
| tma_xyz_dino | 10.00 | 25.78 | 37.78 | 97.00 |

数据来自官方Long train专家演示；Long来源内，其他三套件补丁迁移。
测试使用正常OFT和RGB PNG，无DINO训练hook。单训练seed；40–49为已查看开发网格。
任务成功率、配对基线差值/区间、每任务结果分别见SUMMARY、CONTRASTS、PER_TASK；不作跨suite平均。
