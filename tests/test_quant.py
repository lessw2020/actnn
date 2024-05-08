import numpy as np
import torch

from actnn.ops import ext_minimax

from timeit_v2 import py_benchmark

import actnn.cpp_extension.quantization as ext_quantization


def quantize_and_pack(data, bits, mn, mx):

    # Pack to bitstream
    if isinstance(bits, int):
        print(f"single precision {bits=}")
        pack_func = ext_quantization.pack_single_precision
    else:
        print(f"mixed precision {bits=}")
        pack_func = ext_quantization.pack_mixed_precision
    output, scale = pack_func(data, mn, mx, bits, True)
    #if config.swap:
    #    output = swap_to_cpu(output)

    return output, scale


def no_scheme_compute_quantization_bits(input):
    N = input.shape[0]
    D = input.shape[1]
    input_flatten = input.view(N, -1)
    num_features = input_flatten.shape[1]
    num_pixels = num_features // D

    group_size = 256
    activation_compression_bits = [2, 8, 8]

    # Compute min, max by groups
    if num_features % group_size != 0:
        # Padding
        new_num_features = (num_features // group_size + 1) * group_size
        delta = new_num_features - num_features
        input_flatten = torch.cat([input_flatten,
                                   torch.zeros([N, delta], dtype=input.dtype, device=input.device)], 1)

    input_groups = input_flatten.view(-1, group_size)
    mn, mx = ext_minimax.minimax(input_groups)

    b = activation_compression_bits[0]
    return input_groups.view(N, -1, group_size), b, mn.view(N, -1, 1), mx.view(N, -1, 1)


def quantize_activation(input, scheme=None):

    N = input.shape[0]
    #if scheme:
    #    input_groups, q_bits, q_min, mx = scheme.compute_quantization_bits(input)
    #else:
    input_groups, q_bits, q_min, mx = no_scheme_compute_quantization_bits(input)
    print(f"{input_groups=}, {q_bits=}, {mx=}, {q_min=}")

    q_input, q_scale = quantize_and_pack(input_groups, q_bits, q_min, mx)


    # TODO convert q_bits to int8
    if input.dtype == torch.float32:
        return q_input, q_bits, q_scale.to(torch.bfloat16), q_min.to(torch.bfloat16)
    else:
        return q_input, q_bits, q_scale, q_min


def test_quant_correctness():
    print("========== Quant Correctness Test ==========")

    activation = torch.tensor([
    [-100.0070, -0.0123, -0.0019, 0.0086, -0.0028, 0.0012],
    [0.0041, 0.0119, -0.0089, -0.0102, 0.0020, -0.0017],
    [0.0105, 0.0071, 0.0150, 0.0015, -0.0062, 0.0101],
    [-0.0120, -0.0006, 0.0009, 0.0001, 0.0008, -0.0084],
    [0.0025, -0.0081, 0.0223, -0.0047, 0.0114, -0.0015],
    [-0.0093, -0.0102, -0.0139, 0.0058, 0.0023, 100.0084],
    [-110.0070, -0.0123, -0.0019, 0.0086, -0.0028, 0.0012],
    [0.0041, 0.0119, -0.0089, -0.0102, 0.0020, -0.0017],
    [0.0105, 0.0071, 0.0150, 0.0015, -0.0062, 0.0101],
    [-0.0120, -0.0006, 0.0009, 0.0001, 0.0008, -0.0084],
    [0.0025, -0.0081, 0.0223, -0.0047, 0.0114, -0.0015],
    [-0.0093, -0.0102, -0.0139, 0.0058, 0.0023, 101.0084],
    [-120.0070, -0.0123, -0.0019, 0.0086, -0.0028, 0.0012],
    [0.0041, 0.0119, -0.0089, -0.0102, 0.0020, -0.0017],
    [0.0105, 0.0071, 0.0150, 0.0015, -0.0062, 0.0101],
    [-0.0120, -0.0006, 0.0009, 0.0001, 0.0008, -0.0084],
    [0.0025, -0.0081, 0.0223, -0.0047, 0.0114, -0.0015],
    [-0.0093, -0.0102, -0.0139, 0.0058, 0.0023, 110.0084]
    ], dtype = torch.float32, device="cuda:0")

    act2 = torch.randn((768, 768), dtype=torch.float32, device="cuda:0")

    #print(f'{activation=}')

    q_inputs, q_scale, q_min, q_max = quantize_activation(activation)
    print(f'{q_inputs=}, {q_scale=}')
    print(f'{q_min=}, {q_max=}')





    '''
    for dtype in ['float32', 'float16']:
        print(f"test {dtype}...")
        data_np = np.random.randn(1024, 256).astype(dtype)

        def test_implementation(func):
            data = torch.tensor(data_np).to("cuda")

            if func == torch:
                mn, mx = torch.min(data, 1)[0], torch.max(data, 1)[0]
            else:
                mn, mx = ext_minimax.minimax(data)[:2]

            return [x.detach().cpu().numpy() for x in [mn, mx]]

        mn_ref, mx_ref =  test_implementation(torch)
        mn_us, mx_us = test_implementation(ext_minimax)

        np.testing.assert_allclose(mn_ref, mn_us)
        np.testing.assert_allclose(mx_ref, mx_us)

    '''
if __name__ == "__main__":
    test_quant_correctness()
    #test_minimax_speed()
