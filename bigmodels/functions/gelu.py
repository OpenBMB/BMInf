from ..backend import create_ufunc

gelu_kernel = create_ufunc('bms_gelu', ('ff->f', 'ee->e'), 'out0 = 0.5 * in0 * (1.0 + tanh(0.7978845608028654 * in0 * (1.0 + 0.044715 * in0 * in0))) * in1')
