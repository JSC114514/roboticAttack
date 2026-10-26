from pipeline import command_for, ensure_runtime_directories, gpu_dependencies


def test_runtime_directories_exist_before_sbatch(tmp_path):
    paths = ensure_runtime_directories(tmp_path)
    assert all(path.is_dir() for path in paths)
    assert tmp_path.joinpath("logs").is_dir()
    assert tmp_path.joinpath("reports", "FAILURES").is_dir()


def test_gpu_dependencies_deduplicate_arrays_and_ignore_cpu():
    output = "\n".join([
        "123_1|RUNNING|gres/gpu:4090:1|None|old_gpu",
        "123_2|RUNNING|gres/gpu:4090:1|None|old_gpu",
        "124|PENDING|gpu:3090:1|Resources|other_gpu",
        "125|RUNNING|cpu=2|None|cpu_job",
    ])
    assert gpu_dependencies(output) == ["123", "124"]


def test_screen_waits_existing_and_analysis_waits_screen():
    screen = command_for("screen", "a" * 64, existing=("11", "12"))
    assert "--dependency=afterany:11:12" in screen
    analysis = command_for("analyze", "a" * 64, dependency="13", existing=("11",))
    assert "--dependency=afterok:13" in analysis
    assert all("afterany" not in item for item in analysis)
