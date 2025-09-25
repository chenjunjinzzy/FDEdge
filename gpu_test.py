import torch
import time

def main():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        return

    device = torch.device("cuda")
    print(f"Using device: {torch.cuda.get_device_name(device)}")

    x = torch.randn(10000, 10000, device=device)
    torch.cuda.synchronize()

    start = time.time()
    y = torch.mm(x, x)
    torch.cuda.synchronize()
    print(f"Matrix multiply took {time.time() - start:.4f}s on GPU; result device: {y.device}")

if __name__ == "__main__":
    main()
