from __future__ import print_function, absolute_import
import json
import os.path as osp
import shutil

import paddle
import paddle.nn as nn

from .osutils import mkdir_if_missing


def save_checkpoint(state, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    paddle.save(state, fpath)


def load_checkpoint(fpath):
    assert osp.isdir(fpath) or osp.isfile(fpath), 'previous checkpoint path not exists or not a folder'
    fpath = osp.join(fpath, 'checkpoint.pth.tar') if osp.isdir(fpath) else fpath

    if osp.isfile(fpath):
        checkpoint = paddle.load(fpath)
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))


def copy_state_dict(state_dict, model, strip=None):
    tgt_state = model.state_dict()
    copied_names = set()
    for name, param in state_dict.items():
        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
        if name not in tgt_state:
            continue
        # if isinstance(param, Parameter):
        #     param = param.data
        if param.size() != tgt_state[name].size():
            print('mismatch:', name, param.size(), tgt_state[name].size())
            continue
        tgt_state[name].copy_(param)
        copied_names.add(name)

    missing = set(tgt_state.keys()) - copied_names
    if len(missing) > 0:
        print("missing keys in state_dict:", missing)

    return model


class CheckpointManager(object):
    def __init__(self, logs_dir='./logs', **modules):
        self.logs_dir = logs_dir
        self.modules = modules

    def save(self, epoch, fpath=None, **modules):
        ckpt = {}
        modules.update(self.modules)
        for name, module in modules.items():
            ckpt[name] = module.state_dict()
        ckpt['epoch'] = epoch + 1

        fpath = osp.join(self.logs_dir, f"checkpoint-epoch{epoch}.pth.tar") if fpath is None else fpath
        save_checkpoint(ckpt, fpath)

    def load(self, ckpt):
        for name, module in self.modules.items():
            # module.load(ckpt.get(name, {}), strict=False)
            module.set_state_dict(ckpt.get(name, {}))
            # print(f"Loading {name}... \n"
            #       f"missing keys {missing_keys} \n"
            #       f"unexpected keys {unexpected_keys} \n")
        return ckpt["epoch"]
