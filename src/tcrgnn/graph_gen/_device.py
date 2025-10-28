import torch

def get_device() -> torch.device:
    # Decide at runtime, not at import
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
