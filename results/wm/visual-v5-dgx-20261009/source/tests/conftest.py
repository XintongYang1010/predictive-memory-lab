import pytest
import torch
from pmr.upstream import activate
from pmr.data import synthetic_raw
from pmr.adapter import RealMotionAdapter


@pytest.fixture(scope="session")
def adapter():
    torch.set_num_threads(2)
    torch.manual_seed(123)
    return RealMotionAdapter("cpu",random_init=True)


@pytest.fixture
def sequence():
    _,module=activate()
    data=synthetic_raw("fixture",12)
    ds=module.Av2Dataset(".",split="train",num_historical_steps=30,split_points=[30,40,50])
    return module.collate_fn([ds.process(data)])
