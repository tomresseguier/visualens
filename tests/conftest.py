import os
from pathlib import Path

import pytest

# Headless Qt and writable matplotlib config for CI and local runs without a display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
MINIMAL_LENS_MODEL_PAR = FIXTURES_DIR / "minimal_model" / "minimal.par"


@pytest.fixture
def small_image():
    from visualens import Image
    from visualens.utils.utils_fits.utils_fits_image import make_default_wcs

    wcs = make_default_wcs(47.233278, 26.760495, npix=64)
    return Image(wcs=wcs, plot_image=False)
