import os
import torch
import paddle
import numpy as np
from reprod_log import ReprodDiffHelper
from reprod_log import ReprodLogger

from reid_torch.models import ResNet, Linear, Networks
from reid_torch.utils.serialization import load_checkpoint, CheckpointManager
from reid.models import ResNet as ResNet_paddle
from reid.models import Linear as Linear_paddle
from reid.models import Networks as Networks_paddle
from reid.utils.serialization import load_checkpoint as load_checkpoint_paddle
from reid.utils.serialization import CheckpointManager as CheckpointManager_paddle


def gen_comdata():
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join('1')
    device = "gpu"  # you can also set it as "cpu"
    torch_device = torch.device("cuda:1" if device == "gpu" else "cpu")
    print(paddle.get_device())

    # load torch model
    backbone = ResNet(depth= 50, last_stride= 2, last_pooling= "avg",
                      embedding=2048)
    classifier = Linear(2048 , 1041, torch_device)
    backbone = backbone.to(torch_device)
    classifier = classifier.to(torch_device)
    backbone.eval()
    checkpoint = load_checkpoint('./data/test_diff/checkpoint.pth.tar')
    manager = CheckpointManager(logs_dir='./logs',backbone=backbone, classifier=classifier)
    manager.load(checkpoint)
    networks = Networks(backbone, classifier)

    # load paddle model
    backbone_paddle = ResNet_paddle(depth= 50, last_stride= 2, last_pooling= "avg",
                      embedding=2048)
    classifier_paddle = Linear_paddle(2048 , 1041)
    backbone_paddle = backbone_paddle.to("gpu:1")
    classifier_paddle = classifier_paddle.to("gpu:1")
    backbone_paddle.eval()
    checkpoint_paddle = paddle.load('./data/test_diff/checkpoint_paddle.pdparams')
    print("=> Loaded checkpoint '{}'".format('./data/test_diff/checkpoint_paddle.pdparams'))
    manager_paddle = CheckpointManager_paddle(logs_dir='./logs',backbone=backbone_paddle, classifier=classifier_paddle)
    manager_paddle.load(checkpoint_paddle)
    networks_paddle = Networks_paddle(backbone_paddle, classifier_paddle)

    # load data
    inputs = np.load("./data/test_diff/fake_data3.npy")

    reprod_logger = ReprodLogger()
    # save the torch output
    torch_out = networks(
        torch.tensor(
            inputs, dtype=torch.float32).to(torch_device))
    print(torch_out['maps'][0].cpu().detach().numpy().shape)
    reprod_logger.add("all_net", torch_out['preds'].cpu().detach().numpy())
    reprod_logger.save("./data/test_result/forward_ref.npy")

    # save the paddle output
    paddle_out = networks_paddle(paddle.to_tensor(inputs, dtype="float32"))
    print(paddle_out['maps'][0].cpu().detach().numpy().shape)
    reprod_logger.add("all_net", paddle_out['preds'].cpu().detach().numpy())
    reprod_logger.save("./data/test_result/forward_paddle.npy")
    """for i in range(128):
        for j in range(2048):
            print(checkpoint['classifier']['W'][i][j].cpu().numpy() == checkpoint_paddle['classifier']['W'][i][j].numpy())
            print(checkpoint_paddle['classifier']['W'][i][j].numpy()) """


def test_forward():
    # load data
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("./data/test_result/forward_ref.npy")
    paddle_info = diff_helper.load_info("./data/test_result/forward_paddle.npy")

    # compare result and produce log
    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(
        path="./data/test_result/log/all_net.log", diff_threshold=1e-5)

if __name__ == "__main__":
    gen_comdata()
    test_forward()