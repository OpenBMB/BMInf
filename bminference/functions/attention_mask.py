from ..backend import create_ufunc
mask_attention_kernel = create_ufunc(
    'bms_attention_mask',
    ('?ff->f',),
    'out0 = in0 ? in1 : in2'
)