# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================


import sys
import os
import numpy as np
import torch
from torch import nn
import random
from functools import partial
import re


def seed_everything(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_device(gpu=-1):
    if gpu >= 0 and torch.cuda.is_available():
        device = torch.device("cuda:" + str(gpu))
    else:
        device = torch.device("cpu")   
    return device

def _resolve_optimizer_class(name):
    """Case-insensitive lookup of an optimizer class in torch.optim."""
    name_lower = name.lower()
    for attr in dir(torch.optim):
        if attr.lower() == name_lower:
            cls = getattr(torch.optim, attr)
            if isinstance(cls, type):
                return cls
    raise NotImplementedError("optimizer={} is not supported.".format(name))


def get_optimizer(optimizer, params, lr, **extra_kwargs):
    """Optimizer factory.

    Args:
        optimizer: str (e.g. "adam") or dict like
            {"type": "adagrad", "lr": 5e-3, "initial_accumulator_value": 0.1}.
            When dict, keys other than ``type`` are passed verbatim to the
            torch.optim class ctor. ``lr`` inside the dict (if any) overrides
            the positional ``lr`` argument.
        params: iterable of parameters or list of param groups.
        lr: fallback lr used when ``optimizer`` is a str or when the dict
            does not specify ``lr``.
        **extra_kwargs: additional ctor kwargs (merged with lowest priority).

    Returns:
        A torch.optim.Optimizer instance.
    """
    if isinstance(optimizer, dict):
        cfg = dict(optimizer)  # shallow copy, do not mutate caller's dict
        opt_type = cfg.pop("type", None)
        if opt_type is None:
            raise ValueError("optimizer dict must contain a 'type' key")
        # extra_kwargs < cfg (cfg takes precedence)
        kwargs = {**extra_kwargs, **cfg}
        kwargs.setdefault("lr", lr)
    else:
        opt_type = optimizer
        if str(opt_type).lower() == "adam":
            opt_type = "Adam"
        kwargs = {"lr": lr, **extra_kwargs}

    opt_class = _resolve_optimizer_class(opt_type)
    try:
        return opt_class(params, **kwargs)
    except TypeError as e:
        raise TypeError(
            "Failed to build optimizer {} with kwargs={}: {}".format(
                opt_type, kwargs, e))

def get_loss(loss):
    if isinstance(loss, str):
        if loss in ["bce", "binary_crossentropy", "binary_cross_entropy"]:
            loss = "binary_cross_entropy"
    try:
        loss_fn = getattr(torch.functional.F, loss)
    except:
        try: 
            loss_fn = eval("losses." + loss)
        except:
            raise NotImplementedError("loss={} is not supported.".format(loss))       
    return loss_fn

def get_regularizer(reg):
    reg_pair = [] # of tuples (p_norm, weight)
    if isinstance(reg, float):
        reg_pair.append((2, reg))
    elif isinstance(reg, str):
        try:
            if reg.startswith("l1(") or reg.startswith("l2("):
                reg_pair.append((int(reg[1]), float(reg.rstrip(")").split("(")[-1])))
            elif reg.startswith("l1_l2"):
                l1_reg, l2_reg = reg.rstrip(")").split("(")[-1].split(",")
                reg_pair.append((1, float(l1_reg)))
                reg_pair.append((2, float(l2_reg)))
            else:
                raise NotImplementedError
        except:
            raise NotImplementedError("regularizer={} is not supported.".format(reg))
    return reg_pair

def get_activation(activation, hidden_units=None):
    if isinstance(activation, str):
        if activation.lower() in ["prelu", "dice"]:
            assert type(hidden_units) == int
        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "sigmoid":
            return nn.Sigmoid()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        elif activation.lower() == "softmax":
            return nn.Softmax(dim=-1)
        elif activation.lower() == "prelu":
            return nn.PReLU(hidden_units, init=0.1)
        elif activation.lower() == "dice":
            from fuxictr.pytorch.layers.activations import Dice
            return Dice(hidden_units)
        else:
            return getattr(nn, activation)()
    elif isinstance(activation, list):
        if hidden_units is not None:
            assert len(activation) == len(hidden_units)
            return [get_activation(act, units) for act, units in zip(activation, hidden_units)]
        else:
            return [get_activation(act) for act in activation]
    return activation

def get_initializer(initializer):
    if isinstance(initializer, str):
        try:
            initializer = eval(initializer)
        except:
            raise ValueError("initializer={} is not supported."\
                             .format(initializer))
    return initializer
