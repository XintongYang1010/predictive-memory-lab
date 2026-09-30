import pytest
import torch
from pmr.compat import NeighborhoodAttention1D


@pytest.mark.parametrize("length,ksize,bias",[(8,3,False),(8,5,True),(5,5,False),(30,3,True)])
def test_shifted_neighborhood_matches_dense_reference(length,ksize,bias):
    torch.manual_seed(0)
    model=NeighborhoodAttention1D(8,2,ksize,rel_pos_bias=bias).double().eval()
    if bias:
        with torch.no_grad(): model.rpb.copy_(torch.randn_like(model.rpb))
    x=torch.randn(2,length,8,dtype=torch.float64,requires_grad=True)
    actual=model(x)
    q,k,v=model.qkv(x).reshape(2,length,3,2,4).permute(2,0,3,1,4)
    logit=q @ k.transpose(-1,-2) * .5
    mask=torch.full((2,length,length),-torch.inf,dtype=x.dtype)
    for i in range(length):
        start=max(0,min(i-ksize//2,length-ksize))
        for j in range(start,start+ksize):
            mask[:,i,j]=model.rpb[:,j-i+ksize-1] if bias else 0.
    expected=model.proj(((logit+mask).softmax(-1)@v).transpose(1,2).reshape(2,length,8))
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
    ga=torch.autograd.grad(actual.square().sum(),x,retain_graph=True)[0]
    ge=torch.autograd.grad(expected.square().sum(),x)[0]
    torch.testing.assert_close(ga,ge,atol=1e-12,rtol=1e-12)


def test_non_supported_dilation_rejected():
    with pytest.raises(ValueError): NeighborhoodAttention1D(8,2,3,dilation=2)
