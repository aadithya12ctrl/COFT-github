"""Result plots: loss / PFCM / delta over training, the positional NLL curve,
and a COFT-vs-baseline comparison.

matplotlib is an optional dependency -- importing this module without it raises
a clear message instead of failing at call time.
"""

from __future__ import annotations

from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("plotting requires matplotlib: pip install matplotlib") from exc

__all__ = ["plot_history", "plot_positional_curve", "plot_comparison"]


def _save(fig, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"plot -> {out_path}")
    return out_path


def plot_history(history: dict[str, list], out_path: str | Path, title: str = "COFT") -> Path:
    """Three panels: loss, PFCM (F_t), and the adaptive delta."""
    iters = history["iter"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(title, fontsize=14)

    axes[0].plot(iters, history["train_loss"], label="train", alpha=0.85)
    axes[0].plot(iters, history["val_loss"], label="val", alpha=0.85)
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("loss")
    axes[0].set_title("train / val loss")
    axes[0].legend()

    axes[1].plot(iters, history["pfcm"], color="tomato")
    axes[1].axhline(0, linestyle="--", color="gray", linewidth=0.8)
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("PFCM  $F_t$")
    axes[1].set_title("positional forgetting contrast\n($F_t \\to 0$ = uniform retention)")

    axes[2].plot(iters, history["delta"], color="steelblue", drawstyle="steps-post")
    axes[2].set_xlabel("iteration")
    axes[2].set_ylabel(r"$\delta$ (extra tokens)")
    axes[2].set_title("adaptive $\\delta$ scheduler")

    return _save(fig, out_path)


def plot_positional_curve(
    history: dict[str, list], out_path: str | Path, title: str = "Positional NLL"
) -> Path:
    """The U-shaped curve PFCM summarises: NLL binned by position, first vs.
    last evaluation."""
    curves = history.get("positional_curve")
    if not curves:
        raise ValueError("history has no 'positional_curve' entries")

    fig, ax = plt.subplots(figsize=(6, 4))
    for label, curve, style in (
        (f"iter {history['iter'][0]}", curves[0], "--"),
        (f"iter {history['iter'][-1]}", curves[-1], "-"),
    ):
        xs = [i / (len(curve) - 1) for i in range(len(curve))]
        ax.plot(xs, curve, style, label=label)
    ax.axvspan(1 / 3, 2 / 3, color="tomato", alpha=0.08, label="middle third (M)")
    ax.set_xlabel("relative position in context")
    ax.set_ylabel("NLL (nats)")
    ax.set_title(title)
    ax.legend()
    return _save(fig, out_path)


def plot_comparison(
    coft: dict[str, list],
    baseline: dict[str, list],
    out_path: str | Path,
    title: str = "COFT vs. plain fine-tuning",
) -> Path:
    """Side-by-side val loss and PFCM for two arms."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(title, fontsize=14)

    axes[0].plot(coft["iter"], coft["val_loss"], label="COFT")
    axes[0].plot(baseline["iter"], baseline["val_loss"], label="baseline")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("val loss")
    axes[0].set_title("validation loss @ $T_{infer}$")
    axes[0].legend()

    axes[1].plot(coft["iter"], coft["pfcm"], label="COFT", color="tomato")
    axes[1].plot(baseline["iter"], baseline["pfcm"], label="baseline", color="gray")
    axes[1].axhline(0, linestyle="--", color="gray", linewidth=0.8)
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("PFCM  $F_t$")
    axes[1].set_title("middle-context forgetting (lower = better)")
    axes[1].legend()

    return _save(fig, out_path)
