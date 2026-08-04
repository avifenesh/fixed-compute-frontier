#!/usr/bin/env python3
"""Algebra and local-capacity gate for peak-lifted attention."""

from __future__ import annotations

import hashlib, json, math
from pathlib import Path
from typing import Any, Callable, NamedTuple

import torch


OUTPUT = Path("results/peak-lifted-attention-stage0.json")
PREREGISTRATION = Path("results/peak-lifted-attention-stage0-preregistration.md")
SEED = 20260730
PROBES, LENGTH, D, QD, VD, OD, CHUNK = 48, 8, 4, 2, 3, 4, 2
ALPHA_SCALE = 0.5


class State(NamedTuple):
    maximum: torch.Tensor
    mass: torch.Tensor
    output: torch.Tensor


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024): digest.update(chunk)
    return digest.hexdigest()


def unpack(theta: torch.Tensor) -> tuple[torch.Tensor, ...]:
    cursor = 0
    norm = theta[cursor:cursor+D]; cursor += D
    query = theta[cursor:cursor+D*QD].reshape(D,QD); cursor += D*QD
    key = theta[cursor:cursor+D*QD].reshape(D,QD); cursor += D*QD
    value = theta[cursor:cursor+D*VD].reshape(D,VD); cursor += D*VD
    out = theta[cursor:cursor+VD*OD].reshape(VD,OD); cursor += VD*OD
    if cursor != theta.numel(): raise ValueError("parameter ledger drift")
    return norm, query, key, value, out


def pack(parts: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.cat([part.reshape(-1) for part in parts])


def alpha(norm: torch.Tensor, arm: str) -> torch.Tensor:
    return norm.new_zeros(()) if arm == "ordinary" else ALPHA_SCALE * torch.tanh(norm[0]-1.0)


def score_value(theta: torch.Tensor, inputs: torch.Tensor, arm: str):
    norm, query, key, value, out = unpack(theta)
    hidden = inputs * norm if arm == "ordinary" else inputs
    q, k, v = hidden[:,-1] @ query, hidden @ key, hidden @ value
    scores = torch.einsum("bd,bnd->bn", q, k) / math.sqrt(QD)
    return norm, scores, v, out


def effective_logits(scores: torch.Tensor, coefficient: torch.Tensor, arm: str) -> torch.Tensor:
    if arm == "ordinary": return scores
    if arm == "temperature": return (1.0 + coefficient) * scores
    pieces = []
    for start in range(0, scores.shape[-1], CHUNK):
        local = scores[:,start:start+CHUNK]
        pieces.append(local + coefficient * local.max(dim=-1).values[:,None])
    return torch.cat(pieces, dim=-1)


def forward(theta: torch.Tensor, inputs: torch.Tensor, arm: str) -> torch.Tensor:
    norm, scores, values, out = score_value(theta, inputs, arm)
    weights = torch.softmax(effective_logits(scores, alpha(norm, arm), arm), dim=-1)
    return torch.einsum("bn,bnv->bv", weights, values) @ out


def gauge_convert(theta: torch.Tensor) -> torch.Tensor:
    norm, query, key, value, out = (part.clone() for part in unpack(theta))
    query *= norm[:,None]; key *= norm[:,None]; value *= norm[:,None]; norm[:] = 1.0
    return pack((norm,query,key,value,out))


def atomic_states(scores: torch.Tensor, values: torch.Tensor, coefficient: torch.Tensor) -> list[State]:
    states=[]
    for start in range(0,scores.shape[-1],CHUNK):
        local=scores[:,start:start+CHUNK]; local_values=values[:,start:start+CHUNK]
        maximum=local.max(dim=-1).values
        probability=torch.exp(local-maximum[:,None]); mass=probability.sum(dim=-1)
        numerator=torch.einsum("bn,bnv->bv",probability,local_values)
        states.append(State((1.0+coefficient)*maximum,mass,numerator))
    return states


def compose(left: State, right: State) -> State:
    maximum=torch.maximum(left.maximum,right.maximum)
    ls=torch.exp(left.maximum-maximum); rs=torch.exp(right.maximum-maximum)
    return State(maximum,ls*left.mass+rs*right.mass,ls[:,None]*left.output+rs[:,None]*right.output)


def reduce_left(states: list[State]) -> State:
    state=states[0]
    for item in states[1:]: state=compose(state,item)
    return state


def reduce_right(states: list[State]) -> State:
    state=states[-1]
    for item in reversed(states[:-1]): state=compose(item,state)
    return state


def reduce_balanced(states: list[State]) -> State:
    if len(states)==1:return states[0]
    middle=len(states)//2
    return compose(reduce_balanced(states[:middle]),reduce_balanced(states[middle:]))


def normalized(state: State) -> torch.Tensor: return state.output/state.mass[:,None]


def state_error(a: State,b: State)->float:
    return max(float((a.maximum-b.maximum).abs().max()),float((a.mass-b.mass).abs().max()),float((a.output-b.output).abs().max()))


def rank_record(matrix: torch.Tensor)->dict[str,Any]:
    singular=torch.linalg.svdvals(matrix); tolerance=max(matrix.shape)*torch.finfo(matrix.dtype).eps*singular[0]
    rank=int((singular>tolerance).sum())
    return {"shape":list(matrix.shape),"rank":rank,"nullity":matrix.shape[1]-rank,
            "tolerance":float(tolerance),"singular_values":[float(x) for x in singular],
            "smallest_counted_over_tolerance":float(singular[rank-1]/tolerance)}


def jacobian(function: Callable[[torch.Tensor],torch.Tensor],theta:torch.Tensor)->torch.Tensor:
    return torch.autograd.functional.jacobian(lambda x:function(x).reshape(-1),theta,vectorize=True)


def derivative(function:Callable[[torch.Tensor],torch.Tensor],theta:torch.Tensor,index:int,reference:torch.Tensor)->dict[str,float]:
    direction=torch.zeros_like(theta);direction[index]=1
    _,analytic=torch.autograd.functional.jvp(lambda x:function(x).reshape(-1),theta,direction)
    eps=1e-6; finite_difference=(function(theta+eps*direction).reshape(-1)-function(theta-eps*direction).reshape(-1))/(2*eps)
    delta=analytic-finite_difference
    left,singular,_=torch.linalg.svd(reference,full_matrices=False)
    tolerance=max(reference.shape)*torch.finfo(reference.dtype).eps*singular[0]
    basis=left[:,singular>tolerance];residual=analytic-basis@(basis.T@analytic)
    return {"analytic_norm":float(torch.linalg.vector_norm(analytic)),
            "relative_error":float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(finite_difference).clamp_min(1e-30)),
            "outside_reference_span_norm":float(torch.linalg.vector_norm(residual)),
            "outside_reference_span_fraction":float(torch.linalg.vector_norm(residual)/torch.linalg.vector_norm(analytic).clamp_min(1e-30))}


def algebra_probe(generator:torch.Generator)->dict[str,Any]:
    scores=torch.randn(23,LENGTH,generator=generator,dtype=torch.float64)*3
    values=torch.randn(23,LENGTH,5,generator=generator,dtype=torch.float64); coefficient=scores.new_tensor(0.37)
    states=atomic_states(scores,values,coefficient);left=reduce_left(states);right=reduce_right(states);balanced=reduce_balanced(states)
    permuted=reduce_left([states[i] for i in (2,0,3,1)])
    logits=effective_logits(scores,coefficient,"peak");weights=torch.softmax(logits,-1);direct=torch.einsum("bn,bnv->bv",weights,values)
    shifted_scores=scores+torch.linspace(-1000,1000,scores.shape[0])[:,None]
    shifted=normalized(reduce_balanced(atomic_states(shifted_scores,values,coefficient)))
    return {"left_vs_right_state_max_abs":state_error(left,right),"left_vs_balanced_state_max_abs":state_error(left,balanced),
            "left_vs_permuted_state_max_abs":state_error(left,permuted),"left_vs_direct_output_max_abs":float((normalized(left)-direct).abs().max()),
            "common_shift_output_max_abs":float((shifted-direct).abs().max()),"weight_minimum":float(weights.min()),
            "weight_sum_max_abs_error":float((weights.sum(-1)-1).abs().max())}


def local_distribution_probe()->dict[str,float]:
    scores=torch.tensor([[1.2,0.1,-0.7,0.4]],dtype=torch.float64); coefficient=scores.new_tensor(0.6)
    ordinary=torch.softmax(scores.reshape(2,2),-1)
    peak=torch.softmax(effective_logits(scores,coefficient,"peak").reshape(2,2),-1)
    temperature=torch.softmax(effective_logits(scores,coefficient,"temperature").reshape(2,2),-1)
    return {"peak_vs_ordinary_max_abs":float((peak-ordinary).abs().max()),"temperature_vs_ordinary_max_abs":float((temperature-ordinary).abs().max())}


def iia_probe()->dict[str,float]:
    base=torch.tensor([0.4,-0.2,0.1,-0.5,0.2,-0.4,0.7,0.0],dtype=torch.float64)
    ordinary=[];peak=[]
    for distractor in torch.linspace(-1,4,257,dtype=torch.float64):
        scores=base.clone();scores[7]=distractor
        wo=torch.softmax(scores,-1);wp=torch.softmax(effective_logits(scores[None],scores.new_tensor(.7),"peak")[0],-1)
        ordinary.append(torch.log(wo[0]/wo[6]));peak.append(torch.log(wp[0]/wp[6]))
    ordinary=torch.stack(ordinary);peak=torch.stack(peak)
    return {"ordinary_log_ratio_range":float(ordinary.max()-ordinary.min()),"peak_log_ratio_range":float(peak.max()-peak.min())}


def gradient_probe(generator:torch.Generator)->dict[str,float]:
    rows,chunks,width=4096,4,64
    scores=torch.randn(rows,chunks*width,generator=generator,dtype=torch.float64,requires_grad=True)
    values=torch.randn(rows,chunks*width,generator=generator,dtype=torch.float64)
    ordinary=(torch.softmax(scores,-1)*values).sum(-1).sum(); go=torch.autograd.grad(ordinary,scores,retain_graph=True)[0]
    chunked=scores.reshape(rows,chunks,width); maxima,indices=chunked.max(-1)
    peak_logits=(chunked+0.5*maxima[...,None]).reshape(rows,chunks*width)
    peak=(torch.softmax(peak_logits,-1)*values).sum(-1).sum()
    g_logits=torch.autograd.grad(peak,peak_logits,retain_graph=True)[0]
    gp=torch.autograd.grad(peak,scores)[0]
    correction=gp-g_logits
    argmax_mask=torch.zeros_like(chunked,dtype=torch.bool)
    argmax_mask.scatter_(-1,indices[...,None],True);argmax_mask=argmax_mask.reshape_as(scores)
    correction_norm=torch.linalg.vector_norm(correction)
    argmax_correction_norm=torch.linalg.vector_norm(correction[argmax_mask])
    ratio=torch.linalg.vector_norm(gp,dim=-1)/torch.linalg.vector_norm(go,dim=-1).clamp_min(1e-30)
    peak_abs=gp.abs()[argmax_mask];nonpeak_abs=gp.abs()[~argmax_mask]
    return {"rows":rows,"chunks":chunks,"chunk_width":width,"median_norm_ratio":float(torch.quantile(ratio,.5)),
            "p99_norm_ratio":float(torch.quantile(ratio,.99)),"maximum_norm_ratio":float(ratio.max()),
            "correction_nonargmax_max_abs":float(correction[~argmax_mask].abs().max()),
            "correction_norm_on_argmax_fraction":float(argmax_correction_norm/correction_norm.clamp_min(1e-30)),
            "argmax_gradient_abs_median":float(torch.quantile(peak_abs,.5)),"argmax_gradient_abs_p99":float(torch.quantile(peak_abs,.99)),
            "nonargmax_gradient_abs_median":float(torch.quantile(nonpeak_abs,.5)),"nonargmax_gradient_abs_p99":float(torch.quantile(nonpeak_abs,.99))}


def stress_probe(generator:torch.Generator)->dict[str,Any]:
    scores=torch.randn(31,512,generator=generator,dtype=torch.float64)*30+torch.linspace(-10000,10000,31)[:,None]
    values=torch.randn(31,512,7,generator=generator,dtype=torch.float64);states=atomic_states(scores,values,scores.new_tensor(.49))
    a,b=normalized(reduce_left(states)),normalized(reduce_balanced(states))
    return {"all_finite":bool(torch.isfinite(a).all() and torch.isfinite(b).all()),"left_vs_balanced_max_abs":float((a-b).abs().max())}


def finite(value:Any)->bool:
    if isinstance(value,bool) or value is None or isinstance(value,str):return True
    if isinstance(value,(int,float)):return math.isfinite(float(value))
    if isinstance(value,dict):return all(finite(x) for x in value.values())
    if isinstance(value,list):return all(finite(x) for x in value)
    return True


def run()->dict[str,Any]:
    generator=torch.Generator().manual_seed(SEED)
    inputs=torch.randn(PROBES,LENGTH,D,generator=generator,dtype=torch.float64);inputs/=torch.sqrt(inputs.square().mean(-1,keepdim=True)+1e-6)
    count=D+D*QD+D*QD+D*VD+VD*OD;theta=torch.randn(count,generator=generator,dtype=torch.float64)/3;theta[:D]=1;theta=theta.requires_grad_()
    outputs={arm:forward(theta,inputs,arm) for arm in ("ordinary","temperature","peak")}
    endpoints={arm:float((outputs[arm]-outputs["ordinary"]).abs().max()) for arm in ("temperature","peak")}
    parts=list(unpack(theta.detach().clone()));parts[0]=torch.tensor([.7,1.1,.9,1.3],dtype=torch.float64);source=pack(tuple(parts)).requires_grad_();converted=gauge_convert(source.detach()).requires_grad_()
    source_output=forward(source,inputs,"ordinary");gauge={arm:float((source_output-forward(converted,inputs,arm)).abs().max()) for arm in ("temperature","peak")}
    jacobians={arm:jacobian(lambda x,a=arm:forward(x,inputs,a),theta) for arm in ("ordinary","temperature","peak")};ranks={arm:rank_record(j) for arm,j in jacobians.items()}
    derivatives={"temperature":derivative(lambda x:forward(x,inputs,"temperature"),theta,0,jacobians["ordinary"]),"peak":derivative(lambda x:forward(x,inputs,"peak"),theta,0,jacobians["ordinary"])}
    result={"schema":"peak-lifted-attention-stage0-v1","seed":SEED,"torch_version":torch.__version__,"device":str(theta.device),"dtype":str(theta.dtype),
            "shape":{"probes":PROBES,"length":LENGTH,"input":D,"qk":QD,"value":VD,"output":OD,"chunk":CHUNK},
            "parameter_ledger":{"all_arms_stored_scalars":count,"qkvo_mac_change":0,"per_token_cache_fields_added":0,"parallel_partial_scalars_added":0,"exp_added":0,"score_reductions_added":0,"scalar_fma_per_query_head_chunk_added":1},
            "endpoints":endpoints,"gauge_conversion":gauge,"algebra":algebra_probe(generator),"local_distribution":local_distribution_probe(),"iia":iia_probe(),
            "rank":ranks,"derivatives":derivatives,"gradient_diagnostic":gradient_probe(generator),"stress":stress_probe(generator),
            "source_sha256":sha256_file(Path(__file__)),"preregistration_sha256":sha256_file(PREREGISTRATION)}
    a=result["algebra"];d=result["local_distribution"]
    result["gates"]={"zero_endpoint":max(endpoints.values())<=1e-12,"gauge_conversion":max(gauge.values())<=1e-11,
        "associative_commutative_direct":max(a["left_vs_right_state_max_abs"],a["left_vs_balanced_state_max_abs"],a["left_vs_permuted_state_max_abs"],a["left_vs_direct_output_max_abs"])<=1e-11,
        "common_shift_invariant":a["common_shift_output_max_abs"]<=1e-11,
        "local_distribution_separation":d["peak_vs_ordinary_max_abs"]<=1e-12 and d["temperature_vs_ordinary_max_abs"]>=1e-3,
        "iia_separation":result["iia"]["ordinary_log_ratio_range"]<=1e-12 and result["iia"]["peak_log_ratio_range"]>=1e-3,
        "positive_unit_weights":a["weight_minimum"]>0 and a["weight_sum_max_abs_error"]<=1e-12,
        "rank_gain_beyond_temperature":ranks["temperature"]["rank"]==ranks["ordinary"]["rank"] and ranks["peak"]["rank"]>ranks["ordinary"]["rank"] and ranks["peak"]["smallest_counted_over_tolerance"]>=1e4,
        "derivatives_match":max(x["relative_error"] for x in derivatives.values())<=1e-6,
        "temperature_in_baseline_span":derivatives["temperature"]["outside_reference_span_fraction"]<=1e-8,
        "peak_outside_baseline_span":derivatives["peak"]["outside_reference_span_fraction"]>=1e-3,
        "stress_finite":result["stress"]["all_finite"],"ledger_exact":count==44}
    result["stage0_pass"]=all(result["gates"].values()) and finite(result);return result


def main()->None:
    result=run();OUTPUT.parent.mkdir(parents=True,exist_ok=True);OUTPUT.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"stage0_pass":result["stage0_pass"],"gates":result["gates"],"rank":{k:v["rank"] for k,v in result["rank"].items()},"derivatives":result["derivatives"],"algebra":result["algebra"],"local_distribution":result["local_distribution"],"iia":result["iia"],"gradient_diagnostic":result["gradient_diagnostic"],"stress":result["stress"]},indent=2,sort_keys=True))


if __name__=="__main__":main()
