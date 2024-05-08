import numpy as np
import torch

from actnn.ops import ext_minimax

from timeit_v2 import py_benchmark


def test_quant_correctness():
    print("========== Quant Correctness Test ==========")

    activation = torch.tensor([
    [-0.0070, -0.0123, -0.0019, 0.0086, -0.0028, 0.0012],
    [0.0041, 0.0119, -0.0089, -0.0102, 0.0020, -0.0017],
    [0.0105, 0.0071, 0.0150, 0.0015, -0.0062, 0.0101],
    [-0.0120, -0.0006, 0.0009, 0.0001, 0.0008, -0.0084],
    [0.0025, -0.0081, 0.0223, -0.0047, 0.0114, -0.0015],
    [-0.0093, -0.0102, -0.0139, 0.0058, 0.0023, -0.0084]
    ], dtype = torch.float32, device="cuda:0")

    print(f'{activation=}')


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
