"""Simple, consistent architecture diagrams (a left-to-right chain of layer boxes)."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from src.plotting import SERIES, SERIES_MUTED, SURFACE, TEXT, TEXT_SECONDARY, save

KIND_STYLE = {
    "io": dict(face=SURFACE, edge=TEXT_SECONDARY, text=TEXT),
    "prep": dict(face="#e4e3df", edge=TEXT_SECONDARY, text=TEXT),
    "layer": dict(face=SERIES_MUTED, edge=SERIES, text=TEXT),
    "output": dict(face=SERIES, edge=SERIES, text="white"),
}


def draw_chain(layers, title: str, fig_name: str, subtitle: str = ""):
    """layers: list of (kind, name, detail) where kind is io | prep | layer | output.
    Boxes are drawn left to right with arrows; `detail` is a short second line (shape, units, ...)."""
    n = len(layers)
    box_w, box_h, gap = 2.1, 1.25, 0.5
    fig_w = n * (box_w + gap) + 0.4
    fig, ax = plt.subplots(figsize=(max(fig_w, 8), 3.1))
    ax.set_xlim(-0.2, n * (box_w + gap))
    ax.set_ylim(-0.9, 1.9)
    ax.axis("off")
    for i, (kind, name, detail) in enumerate(layers):
        st = KIND_STYLE[kind]
        x = i * (box_w + gap)
        ax.add_patch(FancyBboxPatch((x, 0), box_w, box_h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor=st["face"], edgecolor=st["edge"], linewidth=1.5))
        ax.text(x + box_w / 2, box_h * 0.64, name, ha="center", va="center", fontsize=10,
                fontweight="bold", color=st["text"], wrap=True)
        ax.text(x + box_w / 2, box_h * 0.28, detail, ha="center", va="center", fontsize=8.5,
                color=st["text"] if kind == "output" else TEXT_SECONDARY)
        if i < n - 1:
            ax.annotate("", xy=(x + box_w + gap - 0.05, box_h / 2), xytext=(x + box_w + 0.05, box_h / 2),
                        arrowprops=dict(arrowstyle="-|>", color=TEXT_SECONDARY, lw=1.4))
    ax.text(-0.15, 1.75, title, fontsize=13, fontweight="bold", color=TEXT, va="top")
    if subtitle:
        ax.text(-0.15, -0.35, subtitle, fontsize=9, color=TEXT_SECONDARY, va="top")
    save(fig, fig_name)
    return fig
