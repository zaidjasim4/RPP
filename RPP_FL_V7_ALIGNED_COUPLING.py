# -*- coding: utf-8 -*-
"""Aligned RPP-FL mechanism-level experiment; code-only until executed.

Run with the unmodified RPP_FL_V6_ALL_REVIEWERS_ORIGINAL_AUDITED.py in the same
folder. Generated results are **new** and are not substitutes for previous
185-run or unaligned Reviewer-3 tables. Original source SHA-256:
3da58795bd2b38f05710869c53bbaaec4d35336b6a9116f5ca32c6bfb2709dcb
"""
from __future__ import annotations

import csv
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import statistics
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
import pandas as pd

HERE=Path(__file__).resolve().parent
ORIGINAL=HERE/'RPP_FL_V6_ALL_REVIEWERS_ORIGINAL_AUDITED.py'
EXPECTED_ORIGINAL_SHA256='3da58795bd2b38f05710869c53bbaaec4d35336b6a9116f5ca32c6bfb2709dcb'
actual_sha=hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()
if actual_sha != EXPECTED_ORIGINAL_SHA256:
    raise RuntimeError(f'Original V6 source hash mismatch: {actual_sha}. Refusing to run a different implementation.')
spec=importlib.util.spec_from_file_location('rpp_v6_original_audited',ORIGINAL)
v6=importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name]=v6
spec.loader.exec_module(v6)
# V6 helper used by the deliberately copied ablation-only aggregation body.
cosine = v6.cosine

# The exact original V6 RPP algorithm is inherited without editing. Full calls it.
@dataclass
class MechanismConfig(v6.Config):
    mechanism_variant: str='full'

class AlignedServer(v6.Server):
    def _aggregate_rpp(self, updates, sample_counts):
        variant=getattr(self.cfg,'mechanism_variant','full')
        if variant=='full':
            return super()._aggregate_rpp(updates,sample_counts)
        return self._aggregate_rpp_variant(updates,sample_counts)

    def _aggregate_rpp_variant(self, updates: Dict[int, torch.Tensor],
                       sample_counts: Dict[int, int]) -> Tuple[torch.Tensor, Dict[int, float], Dict[int, float]]:
        """Original V6 aggregation with isolated variant switches (FULL uses super())."""
        variant = self.cfg.mechanism_variant
        if variant not in {"simple_temporal", "uncoupled", "partial"}:
            raise ValueError(f"Unexpected mechanism variant: {variant}")
        cids = list(updates.keys())
        stack = torch.stack([updates[c].detach().cpu() for c in cids], dim=0)
        sketches = torch.stack([self._sketch(updates[c].detach().cpu()) for c in cids], dim=0)
        m = len(cids)
        center, rho, phi = self._robust_center_scale(sketches)

        # D_i: robust coordinate-deviation trust.
        standardized = torch.clamp(torch.abs(sketches - center) / rho, 0.0, 20.0)
        R = torch.median(standardized, dim=1).values.numpy()
        Dscore = np.exp(-np.maximum(R - self.cfg.rpp_deviation_R0, 0.0) / self.cfg.rpp_deviation_tau)

        # C_i and H_i: current-center and historical-momentum consistency.
        Cscore = np.asarray([max(0.0, cosine(s, center)) for s in sketches], dtype=np.float64)
        hist = self._sketch(self.V)
        hist_valid = float(hist.norm()) > 1e-12
        Hscore = np.asarray([
            max(0.0, cosine(s, hist)) if hist_valid else Cscore[i]
            for i, s in enumerate(sketches)
        ], dtype=np.float64)

        # P_i: median pairwise cosine agreement mapped from [-1,1] to [0,1].
        Pscore = np.ones(m, dtype=np.float64)
        if m > 1:
            for i in range(m):
                sims = [cosine(sketches[i], sketches[j]) for j in range(m) if j != i]
                Pscore[i] = float(np.clip((1.0 + float(np.median(sims))) / 2.0, 0.0, 1.0))

        if variant == "simple_temporal":
            # No collusion-dependent evidence and no event-coupled actions.
            flags = np.zeros(m, dtype=bool)
            Gscore = np.ones(m, dtype=np.float64)
            cdiag = {"RPPKappa": float("nan"), "RPPChi": float("nan"), "RPPFlagged": 0}
            components = [
                (0.40 if self.cfg.use_center_alignment else 0.0, Cscore),
                (0.20 if self.cfg.use_momentum else 0.0, Hscore),
                (0.40 if self.cfg.use_robust_deviation else 0.0, Dscore),
            ]
        else:
            flags, Gscore, cdiag = self._collusion_guard(sketches, center, rho)
            components = [
                (self.cfg.lambda_center if self.cfg.use_center_alignment else 0.0, Cscore),
                (self.cfg.lambda_history if self.cfg.use_momentum else 0.0, Hscore),
                (self.cfg.lambda_deviation if self.cfg.use_robust_deviation else 0.0, Dscore),
                (self.cfg.lambda_peer if self.cfg.use_peer_agreement else 0.0, Pscore),
                (self.cfg.lambda_collusion if self.cfg.use_collusion_guard else 0.0, Gscore),
            ]
        denom = sum(w for w, _ in components)
        if denom <= 0:
            trust_arr = np.ones(m, dtype=np.float64)
        else:
            trust_arr = sum(w * v for w, v in components) / denom
        if variant == "partial" and self.cfg.use_collusion_guard:
            trust_arr[flags] *= self.cfg.rpp_collusion_trust_gate
        trust_arr = np.clip(trust_arr, 0.0, 1.0)
        trust = {cid: float(trust_arr[i]) for i, cid in enumerate(cids)}

        # Asymmetric persistent reputation: fast punishment, slow recovery.
        for i, cid in enumerate(cids):
            if self.cfg.use_reputation:
                old = float(self.reputation.get(cid, self.cfg.rpp_rep_init))
                if variant == "simple_temporal":
                    # Symmetric memory update; unlike RPP asymmetric punishment/recovery.
                    alpha = 0.80
                else:
                    alpha = self.cfg.rpp_rep_alpha_decay if trust_arr[i] < old else self.cfg.rpp_rep_alpha_recover
                rep = alpha * old + (1.0 - alpha) * float(trust_arr[i])
                if variant == "partial" and flags[i] and self.cfg.use_collusion_guard:
                    rep *= self.cfg.rpp_rep_collusion_penalty
                self.reputation[cid] = float(np.clip(rep, self.cfg.rpp_rep_min, 1.0))
            else:
                self.reputation[cid] = 1.0

        counts = np.asarray([max(sample_counts[c], 1) for c in cids], dtype=np.float64)
        med_n = float(np.median(counts))
        capped = np.minimum(counts, self.cfg.rpp_sample_cap_multiplier * med_n)
        q = np.sqrt(capped)
        if self.cfg.use_reputation:
            q *= np.asarray([self.reputation[c] for c in cids], dtype=np.float64) ** self.cfg.rpp_reputation_power
        q *= np.maximum(trust_arr, self.cfg.rpp_trust_floor) ** self.cfg.rpp_trust_power
        if (not np.isfinite(q).all()) or q.sum() <= 0:
            q = np.ones(m, dtype=np.float64)
        w = q / q.sum()
        weights = {cid: float(w[i]) for i, cid in enumerate(cids)}

        # Full-dimensional winsorization around coordinate median ±2.5 robust scales.
        full_center, full_rho, _ = self._robust_center_scale(stack)
        if self.cfg.use_winsorization:
            lo = full_center - self.cfg.rpp_winsor_z * full_rho
            hi = full_center + self.cfg.rpp_winsor_z * full_rho
            bounded = torch.maximum(torch.minimum(stack, hi), lo)
        else:
            bounded = stack
        robust_mean = torch.sum(bounded * torch.tensor(w, dtype=bounded.dtype).view(-1, 1), dim=0)

        flagged_fraction = float(flags.mean()) if self.cfg.use_collusion_guard else 0.0
        anchor = 0.0
        if self.cfg.use_median_anchor:
            # All ablations retain a fixed 0.10 anchor; FULL alone uses adaptive anchoring.
            anchor = float(self.cfg.rpp_anchor_base)
        delta = (1.0 - anchor) * robust_mean + anchor * full_center

        self.last_agg_diagnostics = {
            **cdiag,
            "RPPScaleFloorPhi": float(phi),
            "RPPMeanDeviationR": float(np.mean(R)),
            "RPPMeanDeviationTrust": float(np.mean(Dscore)),
            "RPPMeanCenterTrust": float(np.mean(Cscore)),
            "RPPMeanHistoryTrust": float(np.mean(Hscore)),
            "RPPMeanPeerTrust": float(np.mean(Pscore)),
            "RPPMeanCollusionTrust": float(np.mean(Gscore)),
            "RPPFlaggedFraction": flagged_fraction,
            "RPPAnchorCoeff": anchor,
        }
        return self._apply_central_dp(delta, weights), trust, weights

# No silent IID fallback in NEW experiments: use exact original partition routine
# and abort on its warning rather than mislabel a fallback as Dirichlet non-IID.
_original_partition=v6.dirichlet_label_partition
def _strict_partition(*args,**kwargs):
    buffer=io.StringIO()
    with contextlib.redirect_stdout(buffer):
        out=_original_partition(*args,**kwargs)
    message=buffer.getvalue()
    if 'IID fallback' in message:
        raise RuntimeError('Non-IID Dirichlet partition failed; refusing silent IID fallback: '+message.strip())
    if message:
        print(message,end='')
    return out

v6.dirichlet_label_partition=_strict_partition
v6.Server=AlignedServer

# Use original *prepared* cache to save preprocessing; NEVER overwrite original 185-run results.
v6.RESULT_ROOT=f'{v6.DRIVE_ROOT}/RPP_FL_RESULTS_V7_ALIGNED_COUPLING'
v6.STATE_ROOT=f'{v6.RESULT_ROOT}/_STATE'
v6.CACHE_ROOT=f'{v6.DRIVE_ROOT}/RPP_FL_CACHE_V6_REVIEW'
VARIANTS=('simple_temporal','uncoupled','partial','full','p4p')
SEEDS=(42,2026,31415)
DATASETS=('CICIoT2023','TON_IoT')
CONDITIONS=(('clean',0.5,False),('alie20_a05',0.5,True),('alie20_a01',0.1,True))
METRICS=('Test_F1','Test_MCC','Test_Accuracy','MeanClientAttackDetectAUC','MeanMaliciousWeightShare')

# Student t(0.975, df) for n=2 and n=3. The planned design uses n=3.
T975={1:12.706204736,2:4.302652729}
def stats(values):
    vals=[float(x) for x in values if x is not None and np.isfinite(float(x))]
    if not vals: return {'N':0,'Mean':None,'SD':None,'CI95Low':None,'CI95High':None}
    n=len(vals);mean=statistics.mean(vals)
    sd=statistics.stdev(vals) if n>1 else None
    ci=(T975[n-1]*sd/math.sqrt(n)) if n>1 and n-1 in T975 else None
    return {'N':n,'Mean':mean,'SD':sd,'CI95Low':mean-ci if ci is not None else None,'CI95High':mean+ci if ci is not None else None}

def write_csv(path,rows,cols):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    tmp=path+'.tmp'
    with open(tmp,'w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows)
    os.replace(tmp,path)

def audit_full_identity():
    # Fixed-input two-round identity against original V6 server, no training.
    cfg=MechanismConfig(seed=42,mechanism_variant='full')
    def model_ctor(): return torch.nn.Linear(8,2)
    v6.set_seed(42);a=v6.Server.__bases__[0](10,v6.Config(seed=42),model_ctor,18)
    v6.set_seed(42);b=AlignedServer(10,cfg,model_ctor,18)
    gen=torch.Generator().manual_seed(888)
    for rnd in range(2):
        updates={i:torch.randn(18,generator=gen)*0.03 for i in range(8)}
        counts={i:100+13*i for i in range(8)}
        xa,ta,wa=a.aggregate(updates,counts)
        xb,tb,wb=b.aggregate(updates,counts)
        if not torch.equal(xa,xb) or any(ta[k]!=tb[k] or wa[k]!=wb[k] for k in ta):
            raise AssertionError(f'Full aggregation differs from original at round {rnd+1}')
        a.apply_update(xa);b.apply_update(xb)
    print('[PASS] two-round fixed-input FULL == original V6 bitwise (CPU)')

def write_tables(completed):
    root=Path(v6.RESULT_ROOT)/'ALIGNED_TABLES'
    root.mkdir(parents=True,exist_ok=True)
    # Only completed, explicit Scenario + Variant keys, no ablation pooling.
    keys=['Dataset','ProtocolSeed','MechanismVariant','Condition','DirichletAlpha','AttackType','MaliciousFraction']
    results=[dict(r) for r in completed]
    write_csv(str(root/'MECHANISM_PER_SEED.csv'),results,list(dict.fromkeys(keys+[k for r in results for k in r])))
    grouped={}
    paired={}
    for r in results:
        group=(r['Dataset'],r['Condition'],float(r['DirichletAlpha']),r['MechanismVariant'])
        grouped.setdefault(group,[]).append(r)
        paired[(r['Dataset'],r['Condition'],float(r['DirichletAlpha']),int(r['ProtocolSeed']),r['MechanismVariant'])]=r
    summaries=[]
    for (ds,cond,alpha,variant),rows in sorted(grouped.items()):
        for metric in METRICS:
            values=[]
            for r in rows:
                try: values.append(float(r[metric]))
                except (TypeError,ValueError,KeyError):pass
            summaries.append(dict(Dataset=ds,Condition=cond,DirichletAlpha=alpha,MechanismVariant=variant,Metric=metric,**stats(values)))
    cols=['Dataset','Condition','DirichletAlpha','MechanismVariant','Metric','N','Mean','SD','CI95Low','CI95High']
    write_csv(str(root/'MECHANISM_MEAN_SD_CI95.csv'),summaries,cols)
    differences=[]
    for (ds,cond,alpha,seed,variant),r in sorted(paired.items()):
        if variant!='full':continue
        for other in VARIANTS:
            if other=='full':continue
            b=paired.get((ds,cond,alpha,seed,other))
            if b is None:continue
            for m in METRICS:
                try:
                    x=float(r[m]);y=float(b[m]);diff=x-y
                    if not (math.isfinite(x) and math.isfinite(y)):continue
                except (KeyError,TypeError,ValueError):continue
                differences.append({'Dataset':ds,'Condition':cond,'DirichletAlpha':alpha,'Seed':seed,'Comparator':other,'Metric':m,'Full':x,'ComparatorValue':y,'DifferenceFullMinusComparator':diff})
    cols=['Dataset','Condition','DirichletAlpha','Seed','Comparator','Metric','Full','ComparatorValue','DifferenceFullMinusComparator']
    write_csv(str(root/'FULL_VS_COMPARATORS_PAIRED.csv'),differences,cols)
    diffgroups={}
    for r in differences:
        key=(r['Dataset'],r['Condition'],r['DirichletAlpha'],r['Comparator'],r['Metric'])
        diffgroups.setdefault(key,[]).append(r['DifferenceFullMinusComparator'])
    diff_summary=[]
    for (ds,cond,alpha,comp,metric),vals in sorted(diffgroups.items()):
        diff_summary.append(dict(Dataset=ds,Condition=cond,DirichletAlpha=alpha,Comparator=comp,Metric=metric,**stats(vals)))
    write_csv(str(root/'PAIRED_DIFFERENCE_SUMMARY.csv'),diff_summary,['Dataset','Condition','DirichletAlpha','Comparator','Metric','N','Mean','SD','CI95Low','CI95High'])


def main():
    audit_full_identity()
    v6.set_seed(42)
    v6.mount_drive_if_needed();v6.ensure_dirs()
    try: v6.pip_install_if_missing('opacus','opacus')
    except Exception as e: print('[WARN] Opacus optional installation:',e)
    if v6.AUTO_DOWNLOAD_DATASETS:
        for name in DATASETS:v6.ensure_dataset_downloaded(name)
    partial=Path(v6.RESULT_ROOT)/'all_results_partial.csv'
    completed=[]
    if partial.exists() and v6.AUTO_RESUME:
        completed=pd.read_csv(partial).to_dict(orient='records')
    seen={(str(r.get('Dataset')),str(r.get('Scenario'))) for r in completed}
    wanted=0
    errors=[]
    for ds in DATASETS:
        for seed in SEEDS:
            base=MechanismConfig(seed=seed,split_mode=v6.DEFAULT_SPLIT_MODE,enable_early_stopping=False)
            try: data=v6.load_or_prepare_dataset(ds,v6.DATASET_PATHS[ds],base)
            except Exception as e:
                errors.append(f'{ds} seed={seed}: preprocessing {e}');print('[PREP ERROR]',errors[-1]);continue
            for cond,alpha,attacked in CONDITIONS:
                for variant in VARIANTS:
                    wanted+=1
                    name=f'v7_{variant}_{cond}_seed{seed}'
                    cfg=replace(base,dirichlet_alpha=alpha,aggregator='p4p' if variant=='p4p' else 'rpp',mechanism_variant=variant,attack_type='alie' if attacked else 'none',frac_malicious=0.2 if attacked else 0.0,enable_early_stopping=False)
                    if (ds,name) in seen:
                        existing_file=Path(v6.RESULT_ROOT)/ds/name/'final_result.json'
                        matched=False
                        if existing_file.exists():
                            try:
                                previous=json.loads(existing_file.read_text(encoding='utf-8'))
                                matched=(previous.get('RunFingerprint')==v6.experiment_fingerprint(data,cfg,attacked,name))
                            except (OSError,ValueError):
                                matched=False
                        if matched:
                            print('[RESUME SKIP VERIFIED]',ds,name)
                            continue
                        seen.discard((ds,name))
                        completed=[r for r in completed if not (str(r.get('Dataset'))==ds and str(r.get('Scenario'))==name)]
                    try:
                        result,_=v6.run_federated_experiment(data,cfg,attacked,name)
                        result=dict(result);result.update(MechanismVariant=variant,Condition=cond)
                        completed=[r for r in completed if not (str(r.get('Dataset'))==ds and str(r.get('Scenario'))==name)]+[result]
                        seen.add((ds,name))
                        v6.atomic_dataframe_csv(pd.DataFrame(completed),str(partial))
                        write_tables(completed)
                    except Exception as e:
                        msg=f'{ds}/{name}: {type(e).__name__}: {e}'
                        errors.append(msg);print('[RUN ERROR]',msg)
            del data
    expected=len(DATASETS)*len(SEEDS)*len(CONDITIONS)*len(VARIANTS)
    print('[STATUS]',len(completed),'/',expected,'cases; errors=',len(errors))
    run_cfg={'source_sha256':actual_sha,'result_root':v6.RESULT_ROOT,'cache_root':v6.CACHE_ROOT,'aligned_runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'seeds':SEEDS,'datasets':DATASETS,'conditions':CONDITIONS,'variants':VARIANTS,'status':'COMPLETED' if len(completed)==expected and not errors else 'PARTIAL','errors':errors,'note':'These are new training results, not the previously reported R3 figures. FULL delegates to unmodified original V6._aggregate_rpp; other variants share original geometry and differ only in specified control-path gates.'}
    v6.atomic_json_dump(run_cfg,str(Path(v6.RESULT_ROOT)/'ALIGNED_PROTOCOL_MANIFEST.json'))
    if len(completed)!=expected or errors:
        raise SystemExit('INCOMPLETE: inspect error log and rerun; checkpoint resume supported.')

if __name__=='__main__':main()
