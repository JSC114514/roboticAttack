# 发布阅读说明：保留本次完成实验的科学计算与来源检查。函数内已有输入、输出及约束说明；本模块不负责集群提交。
# 本文件为原运行源码的注释副本；训练目标、更新规则和原生前向的可执行AST未改变。
"""Reporting contracts: startup failures still consume allocated GPU time."""
import unittest
from joint_repro.analyze import gpu_cost
from joint_repro.figures import hierarchical_mean,selected_rows


class GPUAccounting(unittest.TestCase):
    def test_failed_startup_without_cuda_is_charged(self):
        value=gpu_cost({'elapsed_s':100,'gpu':None},{'elapsed_s':120,'allocated_tres':'cpu=4,gres/gpu=1,mem=12G'})
        self.assertEqual(value['allocated_gpu_count'],1)
        self.assertAlmostEqual(value['gpu_hours'],120/3600)
        self.assertEqual(value['gpu_cost_time_source'],'slurm_allocated_elapsed')

    def test_generic_and_typed_gpu_tres_are_not_double_counted(self):
        value=gpu_cost({'elapsed_s':100},{'elapsed_s':120,'allocated_tres':'gres/gpu=2,gres/gpu:4090=2'})
        self.assertEqual(value['allocated_gpu_count'],2)

    def test_cpu_allocation_is_not_charged_as_gpu(self):
        value=gpu_cost({'elapsed_s':100,'gpu':'stale device label'},{'elapsed_s':120,'allocated_tres':'cpu=2,mem=4G'})
        self.assertEqual(value['gpu_hours'],0)

    def test_missing_accounting_is_explicitly_labelled(self):
        value=gpu_cost({'elapsed_s':90},requested_gpu=True)
        self.assertAlmostEqual(value['gpu_hours'],90/3600)
        self.assertEqual(value['gpu_cost_time_source'],'process_elapsed_fallback')


class FigureInputs(unittest.TestCase):
    def test_episode_task_suite_weighting_is_not_flat_frame_weighting(self):
        rows=[dict(suite='a',task_id=0,value=0),dict(suite='a',task_id=0,value=2),dict(suite='a',task_id=1,value=9),dict(suite='b',task_id=0,value=3)]
        self.assertAlmostEqual(float(hierarchical_mean(rows,'value')),4)

    def test_partial_or_duplicate_seed_cannot_enter_final_figure(self):
        selection=dict(methods=['B01'],seeds=[42]);row=dict(condition='B01-seed42',completed=399,expected=400)
        with self.assertRaisesRegex(RuntimeError,'Incomplete'):selected_rows([row],selection)
        row['completed']=400
        with self.assertRaisesRegex(RuntimeError,'duplicated'):selected_rows([row,row],selection)

    def test_all_fixed_seeds_are_kept_in_requested_order(self):
        selection=dict(methods=['B01'],seeds=[42,43,44]);rows=[dict(condition=f'B01-seed{s}',completed=400,expected=400) for s in [44,42,43]]
        self.assertEqual([r['condition'] for r in selected_rows(rows,selection)],['B01-seed42','B01-seed43','B01-seed44'])


if __name__=='__main__':unittest.main()
