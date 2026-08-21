import torch
from pathlib import Path

def test():
    Root = Path(__file__).resolve().parent
    checkpoint_float = Root / "checkpoints/float_model_best.pth"
    checkpoint_qat = Root / "checkpoints/qat_model_best.pth"
    dict_float = torch.load(checkpoint_float)
    dict_qat = torch.load(checkpoint_qat)
    print("float model state dict\r\n")
    print(dict_float["model_state_dict"])
    print("\r\n")
    print("qat model state dict\r\n")
    print(dict_qat["model_state_dict"])

if __name__ == "__main__":
    test()