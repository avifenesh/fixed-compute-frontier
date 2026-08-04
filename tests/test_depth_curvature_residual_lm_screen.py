import torch

from experiments.depth_curvature_residual_lm_screen import D, GROUP, ResidualWrapper, UpdateBus


class Add(torch.nn.Module):
    def __init__(self,value):super().__init__();self.value=value
    def forward(self,x,**kwargs):return x+self.value


def test_zero_endpoint_and_curvature_equation():
    bus=UpdateBus();first=ResidualWrapper(Add(2),bus,"depth_curvature",0);second=ResidualWrapper(Add(5),bus,"depth_curvature",1)
    x=torch.zeros(1,1,D);y=first(x);z=second(y)
    assert torch.equal(y,x+2);assert torch.equal(z,y+5)
    second.raw_beta.data.fill_(torch.atanh(torch.tensor(.5)).item())
    bus.previous=torch.full_like(x,2);z=second(y)
    assert torch.allclose(z,y+5+.25*(5-2))


def test_group_ledger():assert D%GROUP==0 and D//GROUP==12
