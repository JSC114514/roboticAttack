# 官方Long八路1000轮结果

只报告逐suite CASR；表中为百分比。

| 方法 | Object | Spatial | Goal | Long |
|---|---:|---:|---:|---:|
| uada_original | 8.00 | 19.56 | 19.75 | 100.00 |
| uada_dino | 6.00 | 14.33 | 32.86 | 100.00 |
| uada_dino_freq | 24.11 | 29.67 | 46.89 | 100.00 |
| tma_xyz | 8.00 | 15.11 | 44.33 | 100.00 |
| tma_xyz_dino | 8.00 | 20.56 | 50.56 | 100.00 |
| tma_xyz_dino_freq | 13.00 | 14.11 | 20.97 | 100.00 |
| tma_xyz_geometry | 9.00 | 18.78 | 44.08 | 100.00 |
| tma_xyz_attention | 12.00 | 24.78 | 29.75 | 100.00 |

数据来自官方Long train专家演示；Long来源内，其他三套件补丁迁移。
测试使用正常OFT和RGB PNG，无DINO训练hook。单训练seed；40–49为已查看开发网格。
任务成功率、配对基线差值/区间、每任务结果分别见SUMMARY、CONTRASTS、PER_TASK；不作跨suite平均。
