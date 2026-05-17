"""Generate diagrams for MC-ESO slides.

Design rules:
  • TEXT-FREE — no titles, no captions, no axis labels. Labels go on the slide
    via pptx textboxes so they are editable and language-portable.
  • DUAL OUTPUT — every figure is saved as both .svg (vector master) and
    .png (high-DPI raster for pptx embed). python-pptx can't embed SVG natively.
  • SQUARE PANELS — every figure documents its panel layout so the slide
    builder can position textboxes on top.
"""

from pathlib import Path
import numpy as np
import matplotlib
# Classical LaTeX-style italic math (Computer Modern) so formulas read as
# math typography, not the default DejaVu sans-serif.
matplotlib.rcParams["mathtext.fontset"] = "cm"
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

C_INK = "#1F242E"
C_MUTED = "#5A6576"
C_RULE = "#D0D5DB"
C_CONTACT = "#2E86AB"
C_DROPLET = "#E07A5F"
C_AIR = "#6B9A4C"
C_ACCENT = "#C0392B"


def save(fig, name):
    """Save fig as vector SVG only. The slide builder converts SVG→PNG via
    qlmanage on demand because python-pptx can't embed SVG directly; the
    converted PNGs live in figures/svg_cache/ (transient build artifacts)."""
    fig.savefig(OUT / f"{name}.svg", facecolor="white", bbox_inches="tight",
                pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {name}.svg")


def square_panel(ax, lim=2.5):
    ax.set_box_aspect(1)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(C_RULE); s.set_linewidth(0.8)


# ---------------------------------------------------------------------------
# Single-channel figures (one per transmission route)
# ---------------------------------------------------------------------------

def fig_channel_contact():
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    square_panel(ax)
    parent = np.array([0.0, 0.0])
    ax.scatter(*parent, s=500, color=C_INK, marker="*", zorder=5,
               edgecolor="white", linewidth=2)
    rng = np.random.default_rng(3)
    children = parent + rng.normal(0, 0.5, size=(8, 2))
    ax.scatter(children[:, 0], children[:, 1], s=200, color=C_CONTACT,
               edgecolor="white", linewidth=1.5, zorder=4)
    # σ_i radius circle
    ax.add_patch(Circle(parent, 0.95, fill=False, edgecolor=C_CONTACT,
                        linewidth=2.5, linestyle="--", alpha=0.7))
    save(fig, "channel_contact")


def fig_channel_droplet():
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    square_panel(ax)
    parent = np.array([-1.6, -1.3])
    strain = np.array([1.5, 1.5])
    ax.scatter(*parent, s=500, color=C_INK, marker="*", zorder=5,
               edgecolor="white", linewidth=2)
    ax.scatter(*strain, s=520, color=C_DROPLET, marker="*", zorder=5,
               edgecolor="white", linewidth=2)
    child = parent + 0.55 * (strain - parent)
    ax.add_patch(FancyArrowPatch(parent, child, arrowstyle="->",
                                  color=C_DROPLET, lw=4,
                                  mutation_scale=24, zorder=4))
    ax.scatter(*child, s=200, color=C_DROPLET,
               edgecolor="white", linewidth=1.5, zorder=5)
    save(fig, "channel_droplet")


def fig_channel_air():
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    square_panel(ax)
    parent = np.array([0.0, 0.0])
    ax.scatter(*parent, s=500, color=C_INK, marker="*", zorder=5,
               edgecolor="white", linewidth=2)
    rng = np.random.default_rng(11)
    children = parent + rng.normal(0, 1.6, size=(10, 2))
    children = np.clip(children, -2.2, 2.2)
    ax.scatter(children[:, 0], children[:, 1], s=200, color=C_AIR,
               edgecolor="white", linewidth=1.5, zorder=4)
    ax.add_patch(Circle(parent, 2.0, fill=False, edgecolor=C_AIR,
                        linewidth=2.5, linestyle="--", alpha=0.7))
    save(fig, "channel_air")


# ---------------------------------------------------------------------------
# Single-mechanism figures
# ---------------------------------------------------------------------------

def fig_mech_strain():
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    square_panel(ax)
    elites = np.array([[1.2, 0.8], [-1.2, 1.2], [0.0, -1.4]])
    for e in elites:
        ax.add_patch(Circle(e, 0.9, fill=False, edgecolor=C_ACCENT,
                            linewidth=1.8, linestyle="--", alpha=0.7))
        ax.scatter(*e, s=520, color=C_ACCENT, marker="*", zorder=5,
                   edgecolor="white", linewidth=2)
    save(fig, "mech_strain")


def fig_mech_host():
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    square_panel(ax)
    n = 5
    xs = np.linspace(-1.8, 1.8, n)
    y_top, y_bot = 1.4, -1.4
    for i, x in enumerate(xs):
        # top row: 4 survivors + 1 dead
        is_dead = i == n - 1
        ax.scatter(x, y_top, s=440,
                   color="#E8B4B0" if is_dead else C_MUTED,
                   edgecolor=C_ACCENT if is_dead else "white",
                   linewidth=2.5 if is_dead else 1.5, zorder=3)
        if is_dead:
            ax.text(x, y_top, "×", fontsize=24, color=C_ACCENT,
                    ha="center", va="center", fontweight="bold", zorder=4)
        # bottom row: 4 survivors + 1 rolled-back
        is_rb = i == n - 1
        ax.scatter(x, y_bot, s=440,
                   color="#E8B4B0" if is_rb else C_MUTED,
                   edgecolor=C_ACCENT if is_rb else "white",
                   linewidth=2.5 if is_rb else 1.5, zorder=3)
    ax.add_patch(FancyArrowPatch((0, y_top - 0.55), (0, y_bot + 0.55),
                                  arrowstyle="->", color=C_INK, lw=2.2,
                                  mutation_scale=20))
    save(fig, "mech_host")


def fig_mech_spillover():
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    square_panel(ax)
    rng = np.random.default_rng(5)
    cluster = rng.normal([-1.5, 0.0], 0.25, size=(8, 2))
    ax.scatter(cluster[:, 0], cluster[:, 1], s=200, color=C_MUTED,
               alpha=0.7, edgecolor="white", linewidth=1.2, zorder=3)
    ax.add_patch(Circle((-1.5, 0.0), 0.6, fill=False, edgecolor=C_MUTED,
                        linewidth=1.4, linestyle="--", alpha=0.6))
    ax.add_patch(FancyArrowPatch((-0.55, 0), (0.45, 0), arrowstyle="->",
                                  color=C_INK, lw=2.8, mutation_scale=22))
    spread = rng.uniform([0.7, -1.8], [2.2, 1.8], size=(8, 2))
    ax.scatter(spread[:, 0], spread[:, 1], s=200, color=C_ACCENT,
               alpha=0.85, edgecolor="white", linewidth=1.2, zorder=3)
    save(fig, "mech_spillover")


# ---------------------------------------------------------------------------
# Epidemic ↔ optimization analogy figures (3 small icons)
# ---------------------------------------------------------------------------

def fig_analogy_epidemic():
    """Schematic of an epidemic: many candidates, some infected, source highlighted."""
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    square_panel(ax)
    rng = np.random.default_rng(17)
    n_total = 30
    pts = rng.uniform(-2.2, 2.2, size=(n_total, 2))
    # 4 "infected" near a source
    src = np.array([0.6, 0.2])
    ax.scatter(pts[:, 0], pts[:, 1], s=160, color=C_MUTED, alpha=0.5,
               edgecolor="white", linewidth=1.0, zorder=2)
    near = np.linalg.norm(pts - src, axis=1) < 1.2
    ax.scatter(pts[near, 0], pts[near, 1], s=180, color=C_ACCENT, alpha=0.85,
               edgecolor="white", linewidth=1.2, zorder=3)
    ax.scatter(*src, s=520, color=C_ACCENT, marker="*", edgecolor="white",
               linewidth=2, zorder=5)
    save(fig, "analogy_epidemic")


def fig_analogy_optimization():
    """Schematic of optimization: contour landscape with low-f point highlighted."""
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    square_panel(ax)
    x = np.linspace(-2.5, 2.5, 120)
    y = np.linspace(-2.5, 2.5, 120)
    X, Y = np.meshgrid(x, y)
    # Two-basin landscape
    Z = -1.4 * np.exp(-((X - 0.6) ** 2 + (Y - 0.2) ** 2) / 0.9) \
        - 0.7 * np.exp(-((X + 1.5) ** 2 + (Y + 1.2) ** 2) / 0.6) \
        + 0.04 * (X ** 2 + Y ** 2)
    ax.contourf(X, Y, Z, levels=14, cmap="Blues_r", alpha=0.85)
    src = np.array([0.6, 0.2])
    ax.scatter(*src, s=520, color=C_ACCENT, marker="*", edgecolor="white",
               linewidth=2, zorder=5)
    save(fig, "analogy_optimization")


# ---------------------------------------------------------------------------
# Sigma trajectory (schematic — fallback when no real outbreak_dyn is available)
# ---------------------------------------------------------------------------

def fig_sigma_schematic():
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 1.0)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(C_RULE); s.set_linewidth(0.8)
    t = np.linspace(0, 100, 500)
    sigma = np.empty_like(t)
    for i, ti in enumerate(t):
        seg = ti % 35
        sigma[i] = 0.85 * np.exp(-0.10 * seg)
    ax.plot(t, sigma, color=C_ACCENT, linewidth=3.5)
    for x_r in [35, 70]:
        ax.axvline(x_r, color=C_MUTED, linestyle="--", linewidth=1.2,
                   alpha=0.6)
    save(fig, "sigma_schematic")


# ---------------------------------------------------------------------------
# LaTeX math rendering — produces text-only (no axes) SVG/PNG for embedding
# ---------------------------------------------------------------------------

def render_math(name: str, latex: str, *, fontsize: int = 18,
                color: str = "#1F242E"):
    """Render a single LaTeX formula as SVG + transparent PNG.
    Default fontsize is 18 so that, after embedding at typical slide
    sizes, the formula reads at a comfortable in-slide size."""
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.patch.set_alpha(0)
    fig.text(0, 0, f"${latex}$", fontsize=fontsize, color=color,
             ha="left", va="bottom")
    fig.savefig(OUT / f"{name}.svg", facecolor="none",
                bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)
    print(f"wrote {name}.svg")


def fig_formulas():
    """Per-channel formulas — \\boldsymbol = bold italic (matplotlib mathtext
    equivalent of LaTeX \\bm). Multi-letter subscripts use \\mathrm so labels
    like ``strain``/``rand``/``air`` stay upright instead of being italicised
    letter-by-letter."""
    # Close-contact: rotation-aware Gaussian using the instantaneous empirical
    # covariance C_pop of the population (eigenvalues mean-normalized to 1).
    render_math("formula_contact",
                r"\boldsymbol{x}_p + \mathcal{N}(\boldsymbol{0},\, \sigma_i^{2}\, \boldsymbol{C}_{\mathrm{pop}})")
    # Droplet: DE/current-to-best/1 followed by binomial crossover with the
    # parent (rate CR, at least one coordinate forced to inherit the trial).
    render_math("formula_droplet",
                r"\boldsymbol{x}_p + F\,(\boldsymbol{x}_{\mathrm{strain}}-\boldsymbol{x}_p) "
                r"+ F\,(\boldsymbol{x}_a-\boldsymbol{x}_b)\;\;\xrightarrow{\,\mathrm{bin}(CR)\,}\;\;\boldsymbol{x}_{\mathrm{child}}")
    render_math("formula_air",
                r"\boldsymbol{x}_{\mathrm{rand}} + \mathcal{N}(\boldsymbol{0},\, \sigma_{\mathrm{air}}\, \boldsymbol{I})")


if __name__ == "__main__":
    fig_channel_contact()
    fig_channel_droplet()
    fig_channel_air()
    fig_mech_strain()
    fig_mech_host()
    fig_mech_spillover()
    fig_analogy_epidemic()
    fig_analogy_optimization()
    fig_sigma_schematic()
    fig_formulas()
