"""
Checks whether FAISS and PyTorch can share one process on this machine,
and that FAISS returns the correct nearest neighbours when they do.

Run each variant as its own process, from the backend folder:
    python scripts\faiss_torch_check.py torch_first
    python scripts\faiss_torch_check.py faiss_first
    python scripts\faiss_torch_check.py torch_first --one-thread
"""

import sys

import numpy as np


def main() -> None:
    args = sys.argv[1:]
    order = args[0] if args else "torch_first"
    one_thread = "--one-thread" in args

    if order == "faiss_first":
        import faiss
        import torch
    else:
        import torch
        import faiss

    if one_thread:
        faiss.omp_set_num_threads(1)

    rng = np.random.default_rng(0)
    data = rng.standard_normal((2000, 384)).astype("float32")
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    queries = data[:5] + 0.01 * rng.standard_normal((5, 384)).astype("float32")
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    # FAISS exact search
    index = faiss.IndexFlatIP(384)
    index.add(data)
    _, faiss_ids = index.search(queries, 5)

    # Ground truth with plain numpy
    expected = np.argsort(-(queries @ data.T), axis=1)[:, :5]

    # Use torch's math too, so its OpenMP runtime is exercised as well
    chunk = torch.from_numpy(data[:256])
    torch_product = chunk @ chunk.T

    same = all(set(f) == set(e) for f, e in zip(faiss_ids.tolist(), expected.tolist()))
    top1 = bool((faiss_ids[:, 0] == np.arange(5)).all())
    print(f"order={order} one_thread={one_thread} faiss={faiss.__version__} torch={torch.__version__}")
    print("torch matmul shape:", tuple(torch_product.shape))
    print("OK: FAISS top-5 matches numpy" if same else "MISMATCH: FAISS differs from numpy")
    print("OK: each query finds its own source vector first" if top1 else "MISMATCH: top-1 wrong")


if __name__ == "__main__":
    main()