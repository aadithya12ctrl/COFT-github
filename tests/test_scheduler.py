"""Delta scheduler: direction, clamping, tolerance and state round-trip."""

from coft.scheduler import DeltaScheduler


def test_first_update_does_not_move_delta():
    sched = DeltaScheduler(delta_init=100, delta_step=10)
    assert sched.update(0.5) == 100  # nothing to compare against yet


def test_rising_forgetting_increases_delta():
    sched = DeltaScheduler(delta_init=100, delta_step=10, delta_max=200)
    sched.update(0.1)
    assert sched.update(0.2) == 110
    assert sched.update(0.3) == 120


def test_falling_forgetting_decreases_delta():
    sched = DeltaScheduler(delta_init=100, delta_step=10, delta_min=0)
    sched.update(0.3)
    assert sched.update(0.2) == 90
    assert sched.update(0.1) == 80


def test_clamped_to_bounds():
    sched = DeltaScheduler(delta_init=20, delta_step=50, delta_min=0, delta_max=32)
    sched.update(0.0)
    assert sched.update(1.0) == 32  # clamped at max
    assert sched.update(0.0) == 0  # clamped at min


def test_delta_init_is_clamped_into_range():
    assert DeltaScheduler(delta_init=999, delta_max=256).delta == 256
    assert DeltaScheduler(delta_init=-5, delta_min=0).delta == 0


def test_tolerance_creates_a_dead_zone():
    sched = DeltaScheduler(delta_init=100, delta_step=10, tolerance=0.05)
    sched.update(0.50)
    assert sched.update(0.52) == 100  # inside the dead zone: hold
    assert sched.update(0.70) == 110  # outside: move


def test_history_tracks_every_update():
    sched = DeltaScheduler(delta_init=64, delta_step=16, delta_max=128)
    for ft in (0.1, 0.2, 0.3, 0.1):
        sched.update(ft)
    assert sched.history == [64, 80, 96, 80]
    assert sched.ft_history == [0.1, 0.2, 0.3, 0.1]


def test_state_dict_round_trip():
    sched = DeltaScheduler(delta_init=64, delta_step=16, delta_max=128)
    sched.update(0.1)
    sched.update(0.4)
    restored = DeltaScheduler()
    restored.load_state_dict(sched.state_dict())
    assert restored.delta == sched.delta
    assert restored.prev_ft == sched.prev_ft
    assert restored.history == sched.history


def test_zero_step_scheduler_is_frozen():
    """How the ablation arms disable overextension: step 0, bounds 0."""
    sched = DeltaScheduler(delta_init=0, delta_step=0, delta_min=0, delta_max=0)
    for ft in (0.5, 0.1, 0.9):
        assert sched.update(ft) == 0
