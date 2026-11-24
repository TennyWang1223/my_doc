import aiter.ops.custom_all_reduce as m
from aiter.jit.core import get_module  # noqa
m.lamport_ar_rms_buffer_size_gfx1250(4, 1 << 20)
print("built")
