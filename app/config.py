## general parameters

# range for varying `h`
# h_bounds = (-0.5, 0.5)
h_bounds = (-4, 4)

# range for varying `alpha`
# alpha_bounds = (0, 2)
alpha_bounds = (0.0, 1)

# `c` constant
C = complex(1, 0)


## parameter map params

# image size
param_map_image_shape = (512, 512)

# skip iters on param map
# param_map_skip = 1 << 4
param_map_skip = 1 << 8

# iters on param map
param_map_iter = 1 << 4

# same point detection tol
param_map_tolerance = 1e-2

# starting point for param map
param_map_z0 = complex(0.5, 0.0)

# enable phase space selection
param_map_select_z0_from_phase = True

# resolution
param_map_resolution = 4


param_map_draw_on_select = True


param_map_lossless = False


## phase plot params

# space bounds
# phase_shape = (-5, 5, -5, 5)
phase_shape = (-2, 2, -2, 2)

# phase plot image shape
phase_image_shape = (768, 768)

# skip iters on phase plot
phase_skip = 1 << 4#1 << 1

# iters on phase plot
phase_iter = 1 << 6

# grid size for phase plot
phase_grid_size = 2

phase_plot_select_point = True

# z0 to use when in single-point mode
phase_z0 = param_map_z0
# phase_z0 = None


## basins params

basins_image_shape = phase_image_shape

basins_resolution = 4

basins_skip = param_map_skip + param_map_iter - 1


## bif tree

bif_tree_skip = 150
bif_tree_iter = 1 << 10
bif_tree_z0 = complex(-0.1, 0.1)
