import torch

def quant_allclose(tensor1, tensor2, rtol=1e-05, atol=1e-08, equal_nan=False):
    """
    Checks if two tensors are element-wise equal within a tolerance.

    Args:
        tensor1 (torch.Tensor): The first tensor to compare.
        tensor2 (torch.Tensor): The second tensor to compare.
        rtol (float, optional): The relative tolerance. Default is 1e-05.
        atol (float, optional): The absolute tolerance. Default is 1e-08.
        equal_nan (bool, optional): Whether to compare NaN values as equal. Default is False.

    Returns:
        bool: True if the tensors are element-wise equal within the given tolerance, False otherwise.
    """
    if tensor1.shape != tensor2.shape:
        print(f"mismatched shapes {tensor1.shape=}, {tensor2.shape=}")
        return False

    if equal_nan:
        # Check equality of NaN values
        nan_mask1 = torch.isnan(tensor1)
        nan_mask2 = torch.isnan(tensor2)
        if not torch.equal(nan_mask1, nan_mask2):
            return False

        # Exclude NaN values from the comparison
        tensor1 = tensor1[~nan_mask1]
        tensor2 = tensor2[~nan_mask2]

    # Compute the absolute difference and the maximum absolute difference
    diff = torch.abs(tensor1 - tensor2)
    print(f"{diff=}")
    max_diff = torch.max(diff)
    print(f"{max_diff=}")

    # Compute the maximum absolute value among both tensors
    max_abs = torch.max(torch.abs(tensor1), torch.abs(tensor2))

    # Check if the maximum absolute difference is within the tolerance
    tol = atol + rtol * max_abs
    return max_diff <= tol
