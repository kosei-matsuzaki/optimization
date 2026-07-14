"""SVG → EMF conversion for the deck, one step of the build pipeline.

Figures live in per-page subfolders: figs/<pNN_slug>/<panel>.svg. Most panels
convert cleanly with LibreOffice (soffice); the handful that are LINE charts
must go through Inkscape instead, because soffice's EMF export renders plot
lines as hairlines regardless of lw (Inkscape keeps the real line width, but it
rasterizes 3-D surfaces into huge bitmaps — so it is used ONLY for pure line
charts). Run after figs.py and before build_deck.py.
"""
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

FIGS = Path(__file__).resolve().parent / "figs"

# Inkscape CLI (Homebrew cask installs the app, not a PATH binary).
INKSCAPE = shutil.which("inkscape") or \
    "/Applications/Inkscape.app/Contents/MacOS/inkscape"

# Panels that are line charts → Inkscape (keeps line width). "<page>/<panel>".
INKSCAPE_PANELS = {
    "p13_restart_result/a_conv", "p13_restart_result/b_conv",
    "p24_best2_result/a_conv", "p24_best2_result/b_conv",
    "p17_floor_result/f10_conv", "p17_floor_result/f19_conv",
    "p21_router_conv/a_conv", "p21_router_conv/b_conv",
    # p27/p28 family-conv panels are exported directly as PNG (semi-transparent
    # ±1σ bands can't survive EMF), so they are not converted here.
}


def main():
    svgs = sorted(FIGS.glob("*/*.svg"))
    soffice_by_dir = defaultdict(list)
    n_ink = 0
    for svg in svgs:
        key = f"{svg.parent.name}/{svg.stem}"
        if key in INKSCAPE_PANELS:
            subprocess.run([INKSCAPE, str(svg), "--export-type=emf",
                            "-o", str(svg.with_suffix(".emf"))],
                           check=True, capture_output=True)
            n_ink += 1
        else:
            soffice_by_dir[svg.parent].append(svg)
    for d, files in soffice_by_dir.items():
        subprocess.run(["soffice", "--headless", "--convert-to", "emf",
                        "--outdir", str(d)] + [str(f) for f in files],
                       check=True, capture_output=True)
    n_off = sum(len(v) for v in soffice_by_dir.values())
    # drop stale flat-layout artifacts from the old (pre-subfolder) scheme
    for f in list(FIGS.glob("*.emf")) + list(FIGS.glob("*.svg")):
        f.unlink()
    print(f"converted {n_off} via soffice, {n_ink} via inkscape")


if __name__ == "__main__":
    main()
