import torch
import paddle
import numpy as np
from reprod_log import ReprodLogger
from reprod_log import ReprodDiffHelper

from reid_torch.models import ResNet, Linear, Networks
from reid_torch.utils.serialization import load_checkpoint, CheckpointManager
from reid_torch.loss import TriHardPlusLoss
from reid.models import ResNet as ResNet_paddle
from reid.models import Linear as Linear_paddle
from reid.models import Networks as Networks_paddle
from reid.utils.serialization import load_checkpoint as load_checkpoint_paddle
from reid.utils.serialization import CheckpointManager as CheckpointManager_paddle
from reid.loss import TriHardPlusLoss as TriHardPlusLoss_paddle


def gen_comdata():
    # init loss
    criterion_paddle = paddle.nn.CrossEntropyLoss()
    criterion_torch = torch.nn.CrossEntropyLoss()


    # prepare logger & load data
    reprod_logger = ReprodLogger()
    fake_global = np.load("./data/test_diff/fake_global.npy")
    fake_preds = np.load("./data/test_diff/fake_preds.npy")
    fake_label = np.load("./data/test_diff/fake_label.npy")

    # save the paddle output
    pid_criterion_paddle = paddle.nn.CrossEntropyLoss()
    pid_loss_paddle = pid_criterion_paddle(paddle.to_tensor(fake_preds, dtype="float32"), 
                                            paddle.to_tensor(fake_label, dtype="int64"))
    triplet_criterion_paddle = TriHardPlusLoss_paddle(0.0)
    triplet_loss_paddle = triplet_criterion_paddle(paddle.to_tensor(fake_global, dtype="float32"), 
                                            paddle.to_tensor(fake_label, dtype="int64"))
    loss_paddle = pid_loss_paddle + triplet_loss_paddle
    reprod_logger.add("loss", loss_paddle.cpu().detach().numpy())
    reprod_logger.save("./data/test_result/loss_paddle.npy")

    # save the torch output
    pid_criterion_torch = torch.nn.CrossEntropyLoss()
    pid_loss_troch = pid_criterion_torch(torch.tensor(fake_preds, dtype=torch.float32),
                                                torch.tensor(fake_label, dtype=torch.int64))
    triplet_criterion_torch = TriHardPlusLoss(0.0)
    triplet_loss_torch = triplet_criterion_torch(torch.tensor(fake_global, dtype=torch.float32),
                                                torch.tensor(fake_label, dtype=torch.int64))
    loss_torch = pid_loss_troch + triplet_loss_torch
    reprod_logger.add("loss", loss_torch.cpu().detach().numpy())
    reprod_logger.save("./data/test_result/loss_ref.npy")

def test_loss():
    # load data
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("./data/test_result/loss_ref.npy")
    paddle_info = diff_helper.load_info("./data/test_result/loss_paddle.npy")

    # compare result and produce log
    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(path="./data/test_result/loss_log/loss.log")

if __name__ == "__main__":
    gen_comdata()
    #test_loss()
