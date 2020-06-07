"""
Default config for application.

This config is currently being loaded through config.py (ignored on VCS).
Any overrides can be applied there by doing:

```
from default_config import Config

# apply your overrides here, for example:
# Config.seed = 0

```
"""


class Config:

    # General configuration

    seed = 42
    """
    Random seed for all algorithms used.
    """

    image_shape = (512, 512)
    """
    Shape of images produced, in pixels.
    """

    h_bounds = (-6, -0)
    """
    Bounds of parameter `h`.
    """

    alpha_bounds = (0.5, 1)
    """
    Bounds of parameter `alpha`.
    """

    phase_shape = (-2, 2, -2, 2)
    """
    Shape of the phase plane to use.
    """

    C = complex(-0.5, 0.5)
    """
    `C` constant.
    """

    n_skip = 256
    """
    How many iterations to skip before starting an algorithm (applicable to all algorithms)
    """

    n_iter = 64
    """
    How many iterations to perform (applicable to all algorithms)
    """

    tolerance = 1e-3
    """
    Which level of tolerance to use (applicable to most algorithms)
    """

    # Parameter map configuration

    param_map_z0 = complex(0.5, 0.0)
    """
    z0 to use for parameter maps by default.
    """

    param_map_select_z0_from_phase = True
    """
    Whether to enable selecting z0 from phase plot/basins.
    """

    param_map_draw_on_select = True
    """
    Whether to redraw map upon selecting z0 from phase plot/basins.
    """

    # Phase plot configuration

    phase_grid_size = 16
    """
    Size of a grid of the initial conditions to use for phase plot.
    """

    phase_plot_select_point = False
    """
    Whether to enable selecting z0 from itself.
    """

    phase_z0 = None
    """
    Starting point to use when in single-point mode.
    """

    # basins configuration

    basins_threshold = 128
    """
    How many times should a sequence appear in order to be counted as an attractor.
    """

    attractor_plot_shape = (8, 8)

    attractor_plot_dpi = 80

    # Bifurcation tree configuration

    bif_tree_z0 = complex(0.5, 0)
    """
    z0 to use for bifurcation trees by default.
    """
