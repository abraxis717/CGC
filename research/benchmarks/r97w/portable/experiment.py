#!/usr/bin/env python3
"""CGC standalone R97W reproduction.

Derived from the archived R97W computational protocol. This is NOT the
original frozen experiment package; authority checks and output destinations
are adapted for an independent repository. Test and report separately.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import statistics
import sys
import time
import uuid
import torch
from torch.nn import functional as F

PHASE = 'R97W_ECSG_FORGETTING_RATCHET_R0'
SHA_E0 = '92c987e0fb54d80193fd60ecc9a3e31088f0f9b7cd2cfe8e08c4636a198d7084'
SHA_ARCHIVE = '576922cb74ba4aa13f824ac399a389bb215e12cc4e4b936315a279386169c0db'

class Hold(RuntimeError):
    pass

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(262144), b''):
            h.update(block)
    return h.hexdigest()

def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')

def atomic_json(path, document):
    tmp = path.with_name('.' + path.name + '.tmp')
    with open(tmp, 'xb') as f:
        f.write(canon(document) + b'\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def repo_root():
    return pathlib.Path(__file__).resolve().parents[4]

def verify_authority(root, spec):
    source = root / 'src' / 'cgc' / 'ecsg_e0.py'
    if not source.is_file() or source.is_symlink() or sha(source) != SHA_E0:
        raise Hold('PINNED_E0_SOURCE_MISMATCH')
    # Historical predecessors are retained as digest records only. The
    # original R97W environment validated them; they are not in this repo.
    expected = {'R97O', 'R97Q', 'R97V'}
    parents = spec['authority']['parents']
    if {entry['id'] for entry in parents} != expected:
        raise Hold('HISTORICAL_PARENT_RECORDS_MISSING')
    if any(set(entry) != {'id', 'sha256'} for entry in parents):
        raise Hold('HISTORICAL_PARENT_RECORD_FORMAT')
    loaded = importlib.util.spec_from_file_location('cgc_r97w_verified_e0', source)
    if loaded is None or loaded.loader is None:
        raise Hold('PINNED_E0_IMPORT_FAILURE')
    mod = importlib.util.module_from_spec(loaded)
    loaded.loader.exec_module(mod)
    for name in ('forward', 'project_s3', 'gd_step', 'ECSGE0Cell'):
        if not hasattr(mod, name):
            raise Hold('PINNED_E0_API_MISSING')
    g = torch.Generator().manual_seed(307)
    w = torch.randn(5, 8, generator=g, dtype=torch.float64, requires_grad=True)
    x = torch.randn(3, 5, generator=g, dtype=torch.float64)
    y = mod.forward(w, x)
    z = x @ w
    expected_y = (0.5*z + 0.5*z**2 + 0.5*z**3).sum(-1)
    if y.shape != (3,) or not torch.allclose(y, expected_y, atol=1e-12, rtol=1e-12):
        raise Hold('E0_FORWARD_PARITY')
    y.square().mean().backward()
    if w.grad is None or not torch.isfinite(w.grad).all():
        raise Hold('E0_GRADIENT_PARITY')
    return mod

def seed(*parts):
    b=('R97W|'+ '|'.join(map(str,parts))).encode('utf8')
    return int.from_bytes(hashlib.sha256(b).digest()[:8],'little') % (2**31-1)

def rand(shape,*parts):
    g=torch.Generator(device='cpu').manual_seed(seed(*parts))
    return torch.randn(*shape,generator=g,dtype=torch.float64)

def randperm(n,*parts):
    return torch.randperm(n,generator=torch.Generator(device='cpu').manual_seed(seed(*parts)))

def teacher(family,s,episode,split,spec):
    a=spec['architecture'];d=a['features'];out=a['outputs'];p=spec['protocol']
    # Structural teacher parameters share a fixed distribution within family and seed,
    # independent of individual episodes. They are never available to learners.
    a1=rand((d,out),'shared',family,s,'A')/math.sqrt(d)
    a2=rand((d,out),'shared',family,s,'B')/math.sqrt(d)
    ep=rand((d,out),'episode',family,s,split,episode,'slope')/math.sqrt(d)
    nonce=torch.where(rand((1,out),'episode',family,s,split,episode,'nonce')>0,1.,-1.)
    def target(z):
        if family=='smooth':
            y=.35*torch.sin(1.5*z@a1)+.15*torch.tanh(z@a2)+.09*(z[:,0:1]*z[:,1:2])
        elif family=='conditional':
            gate=torch.sigmoid(3*z[:,0:1])
            y=.40*(gate*torch.sin(z@a1)+(1-gate)*torch.tanh(z@a2))
        else:raise Hold('INVALID_FAMILY')
        return y+.28*nonce+.12*(z@ep)
    def data(kind,n):
        z=rand((n,d),'inputs',family,s,split,episode,kind)
        if kind=='ood':z=z*1.4
        x=torch.cat([z,torch.ones(n,1,dtype=torch.float64)],dim=-1)
        return x,target(z)
    return data('support',p['support']),data('query',p['query']),data('ood',p['ood'])

def init_state(model,s,spec):
    d=spec['architecture']['features']+1;out=spec['architecture']['outputs']
    if model=='ECS':
        n=spec['architecture']['micro_units'];return (rand((out,d,n),'init','ECS',s)*(.17/math.sqrt(d)),)
    h=spec['architecture']['ffn_width'];return (
      rand((h,d),'init','FFN',s,'in')*(.22/math.sqrt(d)),
      torch.zeros(h,dtype=torch.float64),
      rand((out,h),'init','FFN',s,'out')*(.22/math.sqrt(h)),
      torch.zeros(out,dtype=torch.float64))

def predict(model,state,x,api):
    if model=='ECS':return torch.stack([api.forward(w,x) for w in state[0]],dim=1)
    a,b,c,d=state
    return F.linear(F.gelu(F.linear(x,a,b)),c,d)

def mse(pred,target):return ((pred-target)**2).mean()

def adapt(model,state,support,steps,rate,api,record=None,query=None,ood=None,clip=2.0):
    cur=tuple(p.detach().requires_grad_(True) for p in state)
    trajectory={}
    if record is not None:
        def record_at(t):
            with torch.no_grad():
                trajectory[t]=[float(mse(predict(model,cur,xy[0],api),xy[1])) for xy in (query,ood)]
        record_at(0)
    for t in range(1,steps+1):
        loss=mse(predict(model,cur,support[0],api),support[1])
        grad=torch.autograd.grad(loss,cur)
        squared=sum(float(g.detach().square().sum()) for g in grad)
        scale=min(1., clip/(math.sqrt(squared)+1e-12))
        cur=tuple((p-rate*scale*g).detach().requires_grad_(True) for p,g in zip(cur,grad))
        if record is not None and t in record:record_at(t)
    return tuple(p.detach() for p in cur),trajectory

def meta_step(model,slow,support,spec,api):
    conf=spec['protocol'];fast,_=adapt(model,slow,support,conf['inner_steps'],conf['inner_lr'],api)
    eta=conf['outer_interpolation']
    next_slow=tuple((a+eta*(b-a)).detach() for a,b in zip(slow,fast))
    return fast,next_slow

def eval_episode(model,state,data,spec,api):
    conf=spec['protocol'];cur,tra=adapt(model,state,data[0],conf['inner_steps'],conf['inner_lr'],api,
         record=set(conf['eval_steps']),query=data[1],ood=data[2]);return tra,cur

def summarize_episode_rows(rows,model,variant,spec):
    d=sorted([r for r in rows if r['model']==model and r['variant']==variant],key=lambda r:r['episode'])
    if not d:return None
    metrics={}
    for step in spec['protocol']['eval_steps']:
        losses=[r['learning_curve'][str(step)][0] for r in d]
        metrics[str(step)]={'median':statistics.median(losses),'mean':statistics.mean(losses)}
    aucs=[sum(r['learning_curve'][str(step)][0] for step in spec['protocol']['eval_steps'])/len(spec['protocol']['eval_steps']) for r in d]
    gain=[(r['learning_curve']['0'][0]-r['learning_curve'][str(spec['protocol']['inner_steps'])][0]) / max(1e-12,r['learning_curve']['0'][0]) for r in d]
    return {'n':len(d),'steps':metrics,'auc_median':statistics.median(aucs),'auc_mean':statistics.mean(aucs),
            'fractional_gain_median':statistics.median(gain),
            'ood_final_median':statistics.median(r['learning_curve'][str(spec['protocol']['inner_steps'])][1] for r in d)}

def evaluate_group(family,s,states,spec,api,split,episodes,include_shuffled=True):
    rows=[]
    keys=['ECS_META','ECS_COLD','ECS_SHUFFLED','ECS_CONTINUAL','FFN_META','FFN_COLD']
    if not include_shuffled: keys.remove('ECS_SHUFFLED')
    for episode in range(episodes):
        data=teacher(family,s,episode,split,spec)
        for key in keys:
            model=key.split('_')[0]
            trajectory,_=eval_episode(model,states[key],data,spec,api)
            rows.append({'episode':episode,'model':model,'variant':key.split('_',1)[1],
                         'learning_curve':{str(step):trajectory[step] for step in spec['protocol']['eval_steps']}})
    summaries={key:summarize_episode_rows(rows,*key.split('_',1),spec) for key in keys}
    return rows,summaries

def compare(rows,key_a,key_b,spec):
    a={r['episode']:r for r in rows if r['model']+'_'+r['variant']==key_a}
    b={r['episode']:r for r in rows if r['model']+'_'+r['variant']==key_b}
    if set(a)!=set(b):raise Hold('PAIRED_EPISODE_MISMATCH')
    step=str(spec['protocol']['inner_steps'])
    ratios=[a[k]['learning_curve'][step][0]/max(1e-12,b[k]['learning_curve'][step][0]) for k in sorted(a)]
    return {'n':len(ratios),'geomean_ratio':math.exp(statistics.mean(math.log(max(1e-12,r)) for r in ratios)),
            'better_n':sum(r<1 for r in ratios),'median_ratio':statistics.median(ratios)}

def training_run(family,s,spec,api):
    cfg=spec['protocol'];init_ecs=init_state('ECS',s,spec);init_ffn=init_state('FFN',s,spec)
    states={'ECS_META':init_ecs,'ECS_COLD':init_ecs,'ECS_SHUFFLED':init_ecs,
            'ECS_CONTINUAL':init_ecs,'FFN_META':init_ffn,'FFN_COLD':init_ffn}
    trace=[];forget=[]
    for episode in range(cfg['train_episodes']+1):
        if episode in cfg['dev_checkpoints']:
            _,summary=evaluate_group(family,s,states,spec,api,'DEV',cfg['dev_episodes'],include_shuffled=False)
            trace.append({'after_train_episodes':episode,'summary':summary})
        if episode==cfg['train_episodes']:break
        support,query,_=teacher(family,s,episode,'TRAIN',spec)
        ec_fast,states['ECS_META']=meta_step('ECS',states['ECS_META'],support,spec,api)
        ff_fast,states['FFN_META']=meta_step('FFN',states['FFN_META'],support,spec,api)
        fake=support[1][randperm(len(support[1]),'shuffle',family,s,episode)]
        _,states['ECS_SHUFFLED']=meta_step('ECS',states['ECS_SHUFFLED'],(support[0],fake),spec,api)
        states['ECS_CONTINUAL'],_=adapt('ECS',states['ECS_CONTINUAL'],support,cfg['inner_steps'],cfg['inner_lr'],api)
        # Read-only forgetting test: compare the fit of old task using the just-adapted
        # fast state versus the retained slow state after forced fast-state reset.
        if episode >= cfg['train_episodes']-cfg['forget_probe_episodes']:
            with torch.no_grad():
                old_fast=float(mse(predict('ECS',ec_fast,query[0],api),query[1]))
                old_reset=float(mse(predict('ECS',states['ECS_META'],query[0],api),query[1]))
            forget.append({'episode':episode,'old_fast_loss':old_fast,'old_reset_loss':old_reset,
                           'loss_ratio_reset_to_fast':old_reset/max(old_fast,1e-12)})
        if episode % cfg['progress_every']==0:
            print(f'TRAIN_PROGRESS family={family} seed={s} episode={episode+1}/{cfg["train_episodes"]}',flush=True)
    return states,trace,forget

def summarize_results(family,s,rows,summaries,forget,spec):
    comps={c:compare(rows,'ECS_META',c,spec) for c in ('ECS_COLD','ECS_SHUFFLED','ECS_CONTINUAL')}
    comps['FFN_META_vs_FFN_COLD']=compare(rows,'FFN_META','FFN_COLD',spec)
    comps['ECS_META_vs_FFN_META']=compare(rows,'ECS_META','FFN_META',spec)
    medratio=statistics.median(r['loss_ratio_reset_to_fast'] for r in forget)
    thr=spec['qualification'];e=summaries['ECS_META'];cold=summaries['ECS_COLD']
    utility=(comps['ECS_COLD']['geomean_ratio']<=thr['max_geomean_meta_vs_cold']
       and comps['ECS_COLD']['better_n']/comps['ECS_COLD']['n']>=thr['min_paired_positive_fraction']
       and e['auc_median']<=thr['max_auc_ratio']*cold['auc_median']
       and medratio>=thr['min_old_reset_to_fast_loss_ratio'])
    relative_adv=comps['ECS_META_vs_FFN_META']['geomean_ratio'] <= thr['max_ecs_to_ffn_ratio']
    return {'family':family,'seed':s,'summaries':summaries,'comparisons':comps,
            'forget_probes':forget,'forget_median_reset_to_fast':medratio,
            'learning_after_forgetting_utility':utility,'comparative_advantage':relative_adv}

def run(root,bundle,spec,api,out,quick=False):
    print('PHASE='+PHASE,flush=True)
    print('METHOD=FIRST_ORDER_REPTILE_SLOW_INITIALIZATION_FAST_RESET',flush=True)
    print('SCIENCE_SCOPE=BOUNDED_META_LEARNING_NOT_CAUSAL_EMERGENCE',flush=True)
    print('PHASE_PARENT_PINS='+','.join(f"{p['id']}:{p['sha256']}" for p in spec['authority']['parents']),flush=True)
    print('E0_SOURCE_SHA256='+SHA_E0,flush=True)
    conf=spec['protocol'];families=spec['families'];seeds=spec['seeds']
    train_trace={};state_map={};forget_map={}
    for family in families:
        for s in seeds:
            print(f'GROUP_START family={family} seed={s}',flush=True)
            states,trace,forget=training_run(family,s,spec,api)
            name=family+'__'+str(s)
            state_map[name]=states;train_trace[name]=trace;forget_map[name]=forget
            print(f'GROUP_DEV family={family} seed={s} checkpoints='+','.join(map(str,conf['dev_checkpoints'])),flush=True)
    # Safe numeric-only checkpoint, no pickle or full DEV source dump.
    checkpoint={name:{kind:[t.tolist() for t in value] for kind,value in states.items()}
                for name,states in state_map.items()}
    atomic_json(out/'slow_state_checkpoint.json',checkpoint)
    checkpoint_hash=sha(out/'slow_state_checkpoint.json')
    dev={'phase':PHASE,'status':'DEV_COMPLETE','spec_sha256':sha(bundle/'SPEC_PORTABLE.json'),
         'checkpoint_sha256':checkpoint_hash,
         'train_trajectory':train_trace,
         'dev_source':{'family_keys':families,'seeds':seeds,'namespace':'DEV'},
         'closed_predecessor':spec['authority']['parents']}
    atomic_json(out/'dev_frozen.json',dev)
    devhash=sha(out/'dev_frozen.json')
    print('PHASE_DEV='+devhash,flush=True)
    # TEST namespace is not accessed, sampled, or scored until the DEV record is materialized.
    all_results=[]
    for family in families:
        for s in seeds:
            name=family+'__'+str(s)
            rows,summaries=evaluate_group(family,s,state_map[name],spec,api,('FIXTURE_EVAL' if quick else conf['test_namespace']),conf['test_episodes'])
            r=summarize_results(family,s,rows,summaries,forget_map[name],spec)
            all_results.append(r)
            print('LEARNING_CURVE family='+family+' seed='+str(s)+' ECS_META='+','.join(str(round(summaries['ECS_META']['steps'][str(step)]['median'],6)) for step in conf['eval_steps'])
                  +' ECS_COLD='+','.join(str(round(summaries['ECS_COLD']['steps'][str(step)]['median'],6)) for step in conf['eval_steps'])
                  +' STEPS='+','.join(map(str,conf['eval_steps'])),flush=True)
            print('FAMILY_RESULT family='+family+' seed='+str(s)
                  +' ECS_META='+str(round(summaries['ECS_META']['steps'][str(conf['inner_steps'])]['median'],8))
                  +' ECS_COLD='+str(round(summaries['ECS_COLD']['steps'][str(conf['inner_steps'])]['median'],8))
                  +' FFN_META='+str(round(summaries['FFN_META']['steps'][str(conf['inner_steps'])]['median'],8))
                  +' META_vs_COLD='+str(round(r['comparisons']['ECS_COLD']['geomean_ratio'],6))
                  +' META_vs_FFN='+str(round(r['comparisons']['ECS_META_vs_FFN_META']['geomean_ratio'],6))
                  +' META_vs_SHUFFLE='+str(round(r['comparisons']['ECS_SHUFFLED']['geomean_ratio'],6))
                  +' META_vs_CONTINUAL='+str(round(r['comparisons']['ECS_CONTINUAL']['geomean_ratio'],6))
                  +' FFN_META_vs_COLD='+str(round(r['comparisons']['FFN_META_vs_FFN_COLD']['geomean_ratio'],6))
                  +' ECS_META_AUC='+str(round(summaries['ECS_META']['auc_median'],6))
                  +' ECS_COLD_AUC='+str(round(summaries['ECS_COLD']['auc_median'],6))
                  +' ECS_META_OOD='+str(round(summaries['ECS_META']['ood_final_median'],6))
                  +' ECS_COLD_OOD='+str(round(summaries['ECS_COLD']['ood_final_median'],6))
                  +' PAIRED_POSITIVE='+str(r['comparisons']['ECS_COLD']['better_n'])+'/'+str(r['comparisons']['ECS_COLD']['n'])
                  +' FORGET_RESET_RATIO='+str(round(r['forget_median_reset_to_fast'],6))
                  +' UTILITY='+str(r['learning_after_forgetting_utility'])
                  +' COMPARATIVE='+str(r['comparative_advantage']),flush=True)
    per_family={family:[r for r in all_results if r['family']==family] for family in families}
    family_pass={family: all(r['learning_after_forgetting_utility'] for r in group) for family,group in per_family.items()}
    comparative=all(r['comparative_advantage'] for r in all_results)
    pass_utility=all(family_pass.values())
    disposition=('SCIENTIFIC_PASS_BOUNDED_POST_FORGETTING_META_LEARNING_ECS_NO_CAUSAL_EMERGENCE_CLAIM' if pass_utility
                 else 'SCIENTIFIC_NONPASS_POST_FORGETTING_ECS_RATCHET_TARGETS_NOT_MET')
    result={'phase':PHASE,'version':'r0','status':'TEST_COMPLETE','disposition':disposition,
            'spec_sha256':sha(bundle/'SPEC_PORTABLE.json'),'e0_sha256':SHA_E0,'dev_sha256':devhash,
            'checkpoint_sha256':checkpoint_hash,
            'evaluation':all_results,'family_targets':family_pass,'utility_all':pass_utility,
            'ecs_beats_ffn_all':comparative,'science_note':
            'Reptile outer optimization explicitly trains a slow initialization. This does not establish natural causal emergence, autonomous self-improvement, or neural general intelligence.'}
    atomic_json(out/'result.json',result)
    print('PHASE_TEST=COMPLETE test_episodes='+str(len(families)*len(seeds)*conf['test_episodes']),flush=True)
    print('PHASE_CHECKPOINT_SHA256='+checkpoint_hash,flush=True)
    print('PHASE_COMPARATIVE_COUNTS=ECS_VS_FFN_TRUE:'+str(sum(r['comparative_advantage'] for r in all_results))+'/'+str(len(all_results)),flush=True)
    print('PHASE_CONTROLS=ECS_COLD,ECS_SHUFFLED,ECS_CONTINUAL,FFN_META,FFN_COLD,FAST_STATE_RESET',flush=True)
    print('PHASE_TARGETS='+str(pass_utility),flush=True)
    print('PHASE_ECS_VS_FFN_ADVANTAGE='+str(comparative),flush=True)
    print('PHASE_DISPOSITION='+disposition,flush=True)
    print('PHASE_RESULT_SHA256='+sha(out/'result.json'),flush=True)
    print('PHASE_RESULT_PATH='+str(out/'result.json'),flush=True)
    print('PHASE_RUNNER_RC=0',flush=True)
    return 0

def compare_historical(output, repo, *, rtol=1e-6, atol=1e-8):
    archive = repo / 'research' / 'benchmarks' / 'r97w' / 'evidence' / 'run_20261025T191037_af40e7f2' / 'result.json'
    if not archive.is_file() or sha(archive) != SHA_ARCHIVE:
        raise Hold('HISTORICAL_EVIDENCE_DIGEST_MISMATCH')
    historical = json.loads(archive.read_text(encoding='utf-8'))
    current = json.loads((output / 'result.json').read_text(encoding='utf-8'))
    errors = []
    if historical['disposition'] != current['disposition']:
        errors.append('disposition')
    if historical['utility_all'] != current['utility_all'] or historical['ecs_beats_ffn_all'] != current['ecs_beats_ffn_all']:
        errors.append('aggregate qualification booleans')
    historic_groups = {(v['family'], v['seed']): v for v in historical['evaluation']}
    current_groups = {(v['family'], v['seed']): v for v in current['evaluation']}
    if historic_groups.keys() != current_groups.keys():
        errors.append('evaluation group identities')
    def numeric_equal(a, b, label):
        if isinstance(a, bool) or isinstance(b, bool):
            if a != b: errors.append(label)
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if not math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
                errors.append(label)
        elif isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                errors.append(label + ' keys')
            for k in a.keys() & b.keys():
                numeric_equal(a[k], b[k], label + '/' + str(k))
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                errors.append(label + ' length')
            for i, (u, v) in enumerate(zip(a, b)):
                numeric_equal(u, v, label + '/' + str(i))
        elif a != b:
            errors.append(label)
    for key in historic_groups.keys() & current_groups.keys():
        a = historic_groups[key]
        b = current_groups[key]
        for field in ('summaries', 'comparisons', 'forget_probes', 'forget_median_reset_to_fast', 'learning_after_forgetting_utility', 'comparative_advantage'):
            numeric_equal(a[field], b[field], f'{key}/{field}')
    report = {
        'run_type': 'PORTABLE_R97W_REEXECUTION',
        'archived_result_sha256': SHA_ARCHIVE,
        'portable_spec_sha256': sha(repo / 'research' / 'benchmarks' / 'r97w' / 'portable' / 'SPEC_PORTABLE.json'),
        'portable_result_sha256': sha(output / 'result.json'),
        'rel_tol': rtol, 'abs_tol': atol,
        'historical_numeric_agreement': not errors,
        'mismatch_count': len(errors),
        'mismatch_paths_first_25': errors[:25],
        'note': 'This tests numerical agreement with archived summaries. It does not recreate the frozen original spec hash or historical parent attestations.'
    }
    atomic_json(output / 'historical_comparison.json', report)
    print('HISTORICAL_NUMERIC_AGREEMENT=' + str(not errors), flush=True)
    print('HISTORICAL_MISMATCH_COUNT=' + str(len(errors)), flush=True)
    if errors:
        raise Hold('HISTORICAL_NUMERIC_COMPARISON_FAILED')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New output directory, outside repository')
    parser.add_argument('--fixture', action='store_true', help='Small mechanics smoke test; NOT scientific qualification')
    parser.add_argument('--compare-archive', action='store_true', help='After a full run, compare to archived qualified results')
    args = parser.parse_args()
    if args.fixture and args.compare_archive:
        parser.error('Historical comparison requires a full run, not a fixture.')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    root = repo_root()
    bundle = pathlib.Path(__file__).resolve().parent
    output = pathlib.Path(args.output).expanduser().resolve()
    if output == root or root in output.parents:
        parser.error('Output directory must be outside the repository.')
    if output.exists() or output.is_symlink():
        parser.error('Output directory already exists; choose a fresh directory.')
    try:
        spec = json.loads((bundle / 'SPEC_PORTABLE.json').read_text(encoding='utf-8'))
        api = verify_authority(root, spec)
        if args.fixture:
            spec = copy.deepcopy(spec)
            spec['protocol'].update(train_episodes=5, dev_checkpoints=[0, 5], dev_episodes=2,
                test_episodes=2, forget_probe_episodes=3, progress_every=5,
                inner_steps=4, eval_steps=[0, 1, 2, 4])
        output.mkdir(parents=True, exist_ok=False)
        rc = run(root, bundle, spec, api, output, args.fixture)
        if rc == 0 and args.compare_archive:
            compare_historical(output, root)
        if rc == 0:
            print('REEXECUTION_STATUS=' + ('FIXTURE_COMPLETED' if args.fixture else 'FULL_COMPLETED'), flush=True)
        return rc
    except (Hold, ValueError, RuntimeError, KeyError, AssertionError, OSError) as exc:
        print('PORTABLE_RUN_HOLD=' + type(exc).__name__, flush=True)
        print('PORTABLE_RUN_REASON=' + str(exc), flush=True)
        return 2

if __name__ == '__main__':
    sys.exit(main())
