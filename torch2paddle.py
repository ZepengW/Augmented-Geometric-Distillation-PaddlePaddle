from subprocess import check_output
import numpy as np
import torch
import paddle

def torch2paddle():
    torch_path = './data/test_diff/checkpoint.pth.tar'
    paddle_path = "./data/test_diff/checkpoint_paddle.pdparams"
    checkpoint = torch.load(torch_path)
    backbone = checkpoint['backbone']
    classifier = checkpoint['classifier']
    fc_names = ["fc", "W"]
    checkpoint_paddle = {}
    backbone_paddle = {}
    classifier_paddle = {}
    for k in backbone:
        if "num_batches_tracked" in k:
            continue
        v = backbone[k].detach().cpu().numpy()
        flag = [i in k for i in fc_names]
        if any(flag) and "weight" in k: # ignore bias
            new_shape = [1, 0] + list(range(2, v.ndim))
            print(f"name: {k}, ori shape: {v.shape}, new shape: {v.transpose(new_shape).shape}")
            v = v.transpose(new_shape)
        k = k.replace("running_var", "_variance")
        k = k.replace("running_mean", "_mean")
        backbone_paddle[k] = v
    for k in classifier:
        if "num_batches_tracked" in k:
            continue
        v = classifier[k].detach().cpu().numpy()
        classifier_paddle[k] = v
    checkpoint_paddle['backbone'] = backbone_paddle
    checkpoint_paddle['classifier'] = classifier_paddle
    checkpoint_paddle['epoch'] = checkpoint['epoch']
    for k in checkpoint_paddle['backbone']:
        print(k)
    for k in checkpoint_paddle['classifier']:
        print(k)

    paddle.save(checkpoint_paddle, paddle_path)

if __name__ == "__main__":
    torch2paddle()
    """checkpoint = torch.load("./data/test_diff/checkpoint.pth.tar")
    for i in checkpoint['backbone']:
        print(i)
    for i in checkpoint['classifier']:
        print(i)"""