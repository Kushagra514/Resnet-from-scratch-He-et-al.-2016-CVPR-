"""Generate ResNet depth comparison and architecture figures."""

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


RESULTS_DIR = Path(__file__).parent / "results"
MODEL_PATTERN = re.compile(r"(?P<family>plain|resnet)_cifar(?P<depth>\d+)\.csv$")
REQUIRED_COLUMNS = {
    "epoch",
    "train_loss",
    "train_accuracy",
    "test_loss",
    "test_accuracy",
}


def load_results():
    results = {}
    for path in RESULTS_DIR.glob("*_cifar*.csv"):
        match = MODEL_PATTERN.match(path.name)
        if match is None:
            continue

        family = match.group("family")
        depth = int(match.group("depth"))
        with path.open(newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
                missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
                raise ValueError(f"{path} is missing columns: {sorted(missing)}")

            rows = [
                {key: float(value) for key, value in row.items()}
                for row in reader
            ]

        if not rows:
            raise ValueError(f"{path} contains no experiment rows")
        results[(family, depth)] = rows

    if not results:
        raise FileNotFoundError("No plain_cifar*.csv or resnet_cifar*.csv files found")
    return dict(sorted(results.items(), key=lambda item: (item[0][1], item[0][0])))


def plot_depth_comparison(results):
    colors = {20: "#0072B2", 32: "#D55E00", 50: "#009E73"}
    metrics = [
        ("train_loss", "Training Loss", "Loss"),
        ("train_accuracy", "Training Accuracy", "Accuracy (%)"),
        ("test_loss", "Test Loss", "Loss"),
        ("test_accuracy", "Test Accuracy", "Accuracy (%)"),
    ]

    figure, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    figure.suptitle("CIFAR Plain vs Residual Networks by Depth", fontsize=17, fontweight="bold")

    curve_axes = axes[:2].flat
    for axis, (column, title, ylabel) in zip(curve_axes, metrics):
        for (family, depth), rows in results.items():
            epochs = [row["epoch"] for row in rows]
            values = [row[column] for row in rows]
            axis.plot(
                epochs,
                values,
                color=colors.get(depth),
                linewidth=2.2,
                linestyle="-" if family == "resnet" else "--",
                marker="o",
                markersize=3.5,
                markevery=2,
                label=f"{'ResNet' if family == 'resnet' else 'Plain'}-{depth}",
            )
            axis.scatter(
                epochs[-1],
                values[-1],
                color=colors.get(depth),
                edgecolor="white",
                linewidth=0.8,
                s=48,
                zorder=4,
            )
            axis.annotate(
                f"{values[-1]:.2f}",
                (epochs[-1], values[-1]),
                xytext=(5, 0),
                textcoords="offset points",
                color=colors.get(depth),
                fontsize=8,
                va="center",
                weight="bold",
            )
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.margins(x=0.04)

    final_accuracy_axis, best_accuracy_axis = axes[2]
    depths = [20, 32, 50]
    labels = [f"{family.title()}-{depth}" for family, depth in results]
    positions = list(range(len(results)))
    bar_width = 0.24
    final_train_accuracy = [rows[-1]["train_accuracy"] for rows in results.values()]
    final_test_accuracy = [rows[-1]["test_accuracy"] for rows in results.values()]
    best_test_accuracy = [max(row["test_accuracy"] for row in rows) for rows in results.values()]

    final_accuracy_axis.bar(
        [position - bar_width / 2 for position in positions],
        final_train_accuracy,
        width=bar_width,
        color="#666666",
        label="Final training accuracy",
    )
    final_accuracy_axis.bar(
        [position + bar_width / 2 for position in positions],
        final_test_accuracy,
        width=bar_width,
        color=[colors[depth] for family, depth in results],
        label="Final test accuracy",
    )
    final_accuracy_axis.set_title("Final Accuracy by Depth")
    final_accuracy_axis.set_ylabel("Accuracy (%)")
    final_accuracy_axis.set_xticks(positions, labels)
    final_accuracy_axis.set_ylim(0, 100)
    final_accuracy_axis.grid(True, axis="y", alpha=0.25)
    final_accuracy_axis.spines["top"].set_visible(False)
    final_accuracy_axis.spines["right"].set_visible(False)

    best_accuracy_axis.bar(
        positions,
        best_test_accuracy,
        color=[colors[depth] for family, depth in results],
        width=0.52,
    )
    for position, value in zip(positions, best_test_accuracy):
        best_accuracy_axis.text(position, value + 0.8, f"{value:.2f}%",
                                ha="center", va="bottom", fontsize=9, weight="bold")
    best_accuracy_axis.set_title("Best Recorded Test Accuracy")
    best_accuracy_axis.set_ylabel("Accuracy (%)")
    best_accuracy_axis.set_xticks(positions, labels)
    best_accuracy_axis.set_ylim(0, 100)
    best_accuracy_axis.grid(True, axis="y", alpha=0.25)
    best_accuracy_axis.spines["top"].set_visible(False)
    best_accuracy_axis.spines["right"].set_visible(False)

    handles, curve_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, curve_labels, loc="upper center", ncol=3,
                  frameon=False, bbox_to_anchor=(0.5, 0.96), fontsize=9)
    final_accuracy_axis.legend(loc="lower left", fontsize=8, frameon=False)
    figure.savefig(RESULTS_DIR / "resnet_depth_comparison.png", dpi=180, facecolor="white")
    plt.close(figure)


def plot_architecture(results):
    depths = list(results)
    figure, axis = plt.subplots(figsize=(14, 6.5))
    axis.set_xlim(0, 17)
    axis.set_ylim(-0.5, len(depths) - 0.5)
    axis.axis("off")

    stage_x = [3.0, 7.3, 11.6]
    stage_labels = [
        ("Stage 1", "16 channels", "32 x 32"),
        ("Stage 2", "32 channels", "16 x 16"),
        ("Stage 3", "64 channels", "8 x 8"),
    ]
    block_width = 0.34
    block_gap = 0.08
    stage_width = 3.1

    axis.text(0.2, len(depths) - 0.02, "CIFAR input", fontsize=10, weight="bold")
    axis.text(0.2, len(depths) - 0.30, "3 x 32 x 32", fontsize=9)
    axis.annotate("", xy=(2.7, len(depths) - 0.32), xytext=(1.45, len(depths) - 0.32),
                  arrowprops={"arrowstyle": "->", "color": "#555555"})

    for index, (stage_name, channels, resolution) in enumerate(stage_labels):
        center = stage_x[index] + stage_width / 2
        axis.text(center, len(depths) - 0.02, stage_name, ha="center", fontsize=10, weight="bold")
        axis.text(center, len(depths) - 0.30, channels, ha="center", fontsize=9)
        axis.text(center, len(depths) - 0.53, resolution, ha="center", fontsize=9, color="#555555")

    for row, depth in enumerate(depths):
        y = len(depths) - row - 1
        blocks_per_stage = (depth - 2) // 6
        axis.text(0.2, y, f"ResNet-{depth}", va="center", fontsize=11, weight="bold")
        axis.text(0.2, y - 0.22, f"n = {blocks_per_stage}; {blocks_per_stage} blocks/stage",
                  va="center", fontsize=8.5, color="#555555")

        for stage_index, x in enumerate(stage_x):
            total_width = blocks_per_stage * block_width + (blocks_per_stage - 1) * block_gap
            start_x = x + (stage_width - total_width) / 2
            for block_index in range(blocks_per_stage):
                block_x = start_x + block_index * (block_width + block_gap)
                axis.add_patch(Rectangle(
                    (block_x, y - 0.16), block_width, 0.32,
                    facecolor="#2f6f9f" if stage_index == 0 else "#4c956c" if stage_index == 1 else "#d97706",
                    edgecolor="white", linewidth=0.8,
                ))

        axis.annotate("", xy=(15.6, y), xytext=(14.9, y),
                      arrowprops={"arrowstyle": "->", "color": "#555555"})
        axis.text(15.72, y, "64 -> 10 logits", va="center", fontsize=9)

    for x in [6.35, 10.65]:
        axis.plot([x, x], [0.1, len(depths) - 0.9], color="#bbbbbb", linewidth=0.8,
                  linestyle="--", zorder=0)

    figure.suptitle(
        "CIFAR-Style ResNet Depths: Three Fixed Stages, More Blocks Per Stage",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.02,
        "All models use Stage 1 -> Stage 2 -> Stage 3; only the number of residual blocks inside each stage changes.",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    figure.savefig(RESULTS_DIR / "resnet_depth_architecture.png", dpi=180, facecolor="white",
                   bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    loaded_results = load_results()
    plot_depth_comparison(loaded_results)
    residual_results = {
        depth: rows
        for (family, depth), rows in loaded_results.items()
        if family == "resnet"
    }
    plot_architecture(residual_results)
    print("Generated comparison and architecture figures")