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

    default_shape = (400, 400)
    """
    Default shape of all images produced, in pixels.
    """

    h_bounds = (-6, -0)
    """
    Bounds of parameter `h`.
    """

    alpha_bounds = (0.5, 1)
    """
    Bounds of parameter `alpha`.
    """

    C = complex(-0.5, 0.5)
    """
    `C` constant.
    """

    # Parameter map configuration

    param_map_image_shape = default_shape
    """
    Image size for parameter maps.
    """

    param_map_skip = 1 << 8
    """
    How many iterations to skip before computing periods on parameter map.
    """

    param_map_iter = 1 << 6
    """
    How many iterations to look at when computing periods on parameter map.
    This affects the number of detectable periods.
    """

    param_map_tolerance = 1e-3
    """
    Precision to use when comparing points in order to find periods.
    """

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

    phase_shape = (-2, 2, -2, 2)
    """
    Region of phase space to draw.
    """

    phase_image_shape = default_shape
    """
    Image size for phase plots.
    """

    phase_skip = 1 << 10
    """
    How many iterations to skip before starting to draw the phase plot.
    """

    phase_iter = 1 << 10
    """
    How many iterations to draw on the phase plot.
    """

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

    basins_image_shape = default_shape
    """
    Image size for basins of attraction.  
    """

    basins_skip = param_map_skip
    """
    How many iterations to skip before capturing points.
    """

    basins_iter = param_map_iter
    """
    How many points to capture.
    This affects the number of detectable periods.
    """

    basins_tolerance = param_map_tolerance
    """
    Precision to use when comparing points in order to find periods.
    """

    # Bifurcation tree configuration

    bif_tree_skip = 1 << 10
    """
    How many iterations to skip before starting to draw the bifurcation tree.
    """

    bif_tree_iter = 1 << 8
    """
    How many iterations to draw.
    """

    bif_tree_z0 = complex(0.5, 0)
    """
    z0 to use for bifurcation trees by default.
    """
