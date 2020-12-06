# Torch-based K-Means
A torch-based implementation of K-Means, MiniBatch K-Means, K-Means++ and more with customizable distance metrics,
and similarity-based clustering.

## Notes
Please note that this repository is still in WIP phase, but feel free to jump in.

The goal is to reach the fastest and cleanest implementation of K-Means, K-Means++ and Mini-Batch K-Means using
PyTorch for CUDA-enabled clustering.


Here's the progress so far:

:white_check_mark: K-Means

:white_check_mark: Similarity-based K-Means (Spherical K-Means)

:white_check_mark: Custom metrics for K-Means

:white_check_mark: K-Means++ initialization

:white_check_mark: DISCERN initialization

:white_check_mark: Purity score

:white_check_mark: MiniBatch K-Means

:black_square_button: (Testing) MiniBatch K-Means++ initialization

:black_square_button: (In progress)MiniBatch K-Means optimized by torch.optim

&nbsp;&nbsp; Successful implementation, much faster than the previous MiniBatch K-Means implementation, but not as accurate.