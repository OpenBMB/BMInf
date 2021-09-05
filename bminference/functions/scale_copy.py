from ..backend import create_ufunc

elementwise_copy_scale = create_ufunc('bms_scaled_copy', ('bee->e', 'iee->e', 'iee->f', 'iff->f'), 'out0 = in0 * in1 * in2')
elementwise_copy = create_ufunc('bms_raw_copy', ('b->b', 'e->e', 'f->f', 'i->i'), 'out0 = in0')