"""Config validation and the ablation grid -- the checks that otherwise fail
deep inside a training step."""

from pathlib import Path

from coft.baseline import ABLATIONS, ablation_config, baseline_config
from coft.config import COFTConfig

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _expect_value_error(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_shipped_configs_are_valid():
    configs = list(CONFIG_DIR.glob("*.yaml"))
    assert configs, "no YAML configs found"
    for path in configs:
        COFTConfig.from_yaml(path)  # raises on an invalid or unknown key


def test_block_budget_enforced():
    _expect_value_error(lambda: COFTConfig(t_infer=768, delta_max=512).validate())
    COFTConfig(t_infer=512, delta_max=512).validate()  # exactly 1024 is allowed


def test_overextension_off_frees_the_budget():
    cfg = COFTConfig(t_infer=1024, delta_max=512, use_overextension=False).validate()
    assert cfg.max_block_size == 1024


def test_unknown_keys_rejected():
    _expect_value_error(lambda: COFTConfig.from_dict({"not_a_key": 1}))


def test_bad_prgg_mode_and_dtype_rejected():
    _expect_value_error(lambda: COFTConfig(prgg_mode="both").validate())
    _expect_value_error(lambda: COFTConfig(dtype="int8").validate())


def test_merge_ignores_none_and_applies_values():
    cfg = COFTConfig().merge(max_iters=None, t_infer=256)
    assert cfg.max_iters == COFTConfig().max_iters
    assert cfg.t_infer == 256


def test_effective_batch_and_run_dir():
    cfg = COFTConfig(batch_size=4, grad_accum=8, out_dir="runs", run_name="x")
    assert cfg.effective_batch == 32
    assert cfg.run_dir == Path("runs/x")


def test_baseline_turns_everything_off():
    cfg = baseline_config(COFTConfig())
    assert not (cfg.use_overextension or cfg.use_prgg or cfg.use_crs)
    assert cfg.run_name.endswith("-baseline")


def test_ablation_grid_covers_each_component():
    assert set(ABLATIONS) == {"baseline", "overext", "overext+prgg", "overext+crs", "coft-full"}
    full = ablation_config(COFTConfig(), "coft-full")
    assert full.use_overextension and full.use_prgg and full.use_crs
    assert ablation_config(COFTConfig(), "overext+prgg").use_crs is False


def test_round_trip_through_dict():
    cfg = COFTConfig(run_name="rt", t_infer=256, delta_max=128)
    assert COFTConfig.from_dict(cfg.to_dict()) == cfg
