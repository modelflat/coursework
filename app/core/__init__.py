import os
import pyopencl as cl

from .utils import alloc_image, read_image, clear_image, read_file


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))

CL_SOURCE_PATH = os.path.join(ROOT_DIR, "cl")

CL_INCLUDE_PATH = os.path.join(CL_SOURCE_PATH, "include")


def build_program_from_file(ctx, file_name, root_path=CL_SOURCE_PATH, include_path=CL_INCLUDE_PATH,
                            options=tuple()):
    if isinstance(file_name, str):
        src = read_file(os.path.join(root_path, file_name))
    else:
        src = "\n".join([read_file(os.path.join(root_path, fn)) for fn in file_name])

    return cl.Program(ctx, src).build(options=[
        "-w", "-I", include_path, *options
    ])
