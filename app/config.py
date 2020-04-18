## general parameters
seed = 42

default_shape = (768, 768)

# range for varying `h`
# h_bounds = (-0.5, 0.5)
# h_bounds = (-10, 10)
# h_bounds = (-6, 6)
# h_bounds = (-4.3114, -4.31)
h_bounds = (-6, -3)
# h_bounds = (0, 6)

# range for varying `alpha`
# alpha_bounds = (0, 2)
# alpha_bounds = (0.0, 1.000)
# alpha_bounds = (0.6044, 0.6045)
alpha_bounds = (0.5, 1)

# `c` constant
C = complex(-0.5, 0.5)


## parameter map params

# image size
param_map_image_shape = default_shape

# skip iters on param map
# param_map_skip = 1 << 4
param_map_skip = 1 << 7

# iters on param map
param_map_iter = 1 << 6

# same point detection tol
param_map_tolerance = 1e-4

# starting point for param map
param_map_z0 = complex(0.5, 0.0)

# enable phase space selection
param_map_select_z0_from_phase = True


param_map_draw_on_select = True


param_map_lossless = False


## phase plot params

# space bounds
# phase_shape = (-5, 5, -5, 5)
phase_shape = (-2, 2, -2, 2)

# phase plot image shape
phase_image_shape = default_shape

# skip iters on phase plot
phase_skip = 1 << 10

# iters on phase plot
phase_iter = 1 << 10

# grid size for phase plot
phase_grid_size = 16

phase_plot_select_point = False

# z0 to use when in single-point mode
# phase_z0 = param_map_z0
phase_z0 = None


## basins params

basins_image_shape = param_map_image_shape

basins_skip = param_map_skip

basins_iter = param_map_iter

basins_tolerance = param_map_tolerance

## bif tree

bif_tree_skip = 1 << 10
bif_tree_iter = 1 << 8
bif_tree_z0 = complex(-0.1, 0.1)
