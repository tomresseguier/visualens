# **visualens**

### A visual, fast interface to explore and manipulate astronomical images, catalogs, and lens models created using Lenstool.

`visualens` uses `PyQt` for fast handling of large astronomical images. Main lensing functionalities include:
- An interactive interface to create Lenstool input files and visualize and manipulate lens models,
- Fast, easy-to-run lensing calculations using `Lenstool`'s engine,
- Visual forward modeling based on `lenstronomy`.

---

### Installation:

Many `visualens` functions rely on `Lenstool`, which is easy to install through conda (highly recommended). We also recommend installing `lenstronomy` through conda rather than pip, before installing `visualens`.

Some IDEs like Spyder may cause conflicts with PyQt. We suggest installing your favorite IDE in your conda virtual environment before installing visualens.

**Installation steps:**

> ```
> conda create -n visualens_env -c conda-forge lenstool astropy matplotlib ipykernel ipywidgets ipyevents lenstronomy python==3.12.15
> conda activate visualens_env
> pip install visualens
> ```

---

### Starting guide:

If using Jupyter notebooks, run these lines in your first cell:
> ```python
> %gui qt5
> from PyQt5.QtWidgets import QApplication
> import sys
> app = QApplication.instance()
> if app is None:
>     app = QApplication(sys.argv)
> ```
Then
> ```python
> from visualens import Visualens
> vl = Visualens()
> ```

***A window should open, with a side panel accessible by clicking the top left button. Click "Open..." under the Image, Catalog, or Lens model tab to get started.***

More examples here:
https://github.com/tomresseguier/visualens/tree/main/examples
