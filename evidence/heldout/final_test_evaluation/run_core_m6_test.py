#!/usr/bin/env python3
"""Isolated final held-out M6 evaluation; never mutates frozen worktrees."""
import csv, hashlib, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT=Path(__file__).resolve().parents[1]
M6=ROOT/'drl-selective-maintenance-m6'
sys.path.insert(0,str(M6))
from src.envs.scenario_bank import Scenario, ScenarioBank, save_scenario_bank, load_scenario_bank
from src.envs.config import get_default_config
from src.envs.selective_maintenance_env import SelectiveMaintenanceEnv
from src.agents.ddqn.agent import DDQNAgent, DDQNAgentConfig
from src.m6.context import build_planner_context_h1, build_planner_context_h2, load_R1_hat
from src.m6.contract import M5_PREDICTION_CACHE_MANIFEST_SHA256
from src.m6.h1_adapter import H1Adapter
from src.m6.h2_planner import H2Planner

REGIMES=('failure-light-no-waste','failure-heavy-no-waste','failure-light-waste-aware','failure-heavy-waste-aware')
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def make_banks(out):
 cache=M6/'data/processed/fd001/v2/06_PREDICTIONS/fd001_prediction_cache_v2.parquet'
 d=pd.read_parquet(cache,filters=[('split','==','rl_test')]).sort_values(['unit_id','cycle'])
 keys=set(zip(d.unit_id,d.cycle)); cyc={}
 for r in d.itertuples(index=False):
  if r.true_rul>0 and all((r.unit_id,r.cycle+x) in keys for x in range(6)):cyc.setdefault(int(r.unit_id),[]).append(int(r.cycle))
 assert len(cyc)==15, 'expected exactly 15 eligible test engines'
 underlying={}
 for k in (1,2):
  xs=[]; seen=set(); units=sorted(cyc)
  for off in range(len(units)*40):
   us=tuple(units[(off*3+s)%len(units)] for s in range(5))
   cs=tuple(cyc[u][(off+s*7+k*11)%len(cyc[u])] for s,u in enumerate(us)); ident=(us,cs)
   if len(set(us))==5 and ident not in seen:
    seen.add(ident); xs.append((us,cs,652100+k*1000+len(xs)))
   if len(xs)==20:break
  assert len(xs)==20
  underlying[k]=xs
  for reg in REGIMES:
   bank=ScenarioBank(f'rl_test_K{k}_{reg}_final_v1','rl_test',tuple(Scenario(f'finaltest_k{k}_{i:03d}__{reg}','rl_test',u,c,s,s,100,k,reg) for i,(u,c,s) in enumerate(xs)))
   save_scenario_bank(bank,out/f'rl_test_K{k}_{reg}_final_v1.json')
 manifest={'schema_version':'rl_test_m6_bank_v1','source_split':'rl_test','source_cache_sha256':sha(cache),'scenario_count_per_cell':20,'underlying':{str(k):[{'units':list(u),'cycles':list(c),'seed':s} for u,c,s in v] for k,v in underlying.items()},'bank_sha256':{p.name:sha(p) for p in sorted(out.glob('*.json'))}}
 (out/'scenario_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
 return manifest
def chooser(method,k,reg,ck=None):
 if method=='h2':
  r=load_R1_hat(M5_PREDICTION_CACHE_MANIFEST_SHA256); return H2Planner(build_planner_context_h2(k,reg,r['R1_hat_cycles'],{'predictor_train_manifest_sha256':r['predictor_train_manifest_sha256'],'computed_at_utc':'2026-07-28T00:00:00Z','n_cycle1_records':r['n_cycle1_records']})).plan
 if method=='m4': return H1Adapter(build_planner_context_h1(k,reg)).plan
 data=torch.load(ck['checkpoint_best_path'],map_location='cpu',weights_only=False); cfg=data['config']; md=data['metadata']
 assert md['maintenance_capacity']==k and md['cost_regime_id']==reg and cfg['gamma']==.95 and md['observation_schema_id']=='m5_point_v1'
 a=DDQNAgent(DDQNAgentConfig(observation_dim=10,num_actions=6 if k==1 else 16,gamma=.95,hidden_dim=int(cfg['hidden_dim']),num_hidden_layers=int(cfg['num_hidden_layers']),explicit_device='cpu'),seed=int(ck['training_seed']))
 a.online_network.load_state_dict(data['online_network_state_dict']);a.online_network.eval();return lambda o:a.evaluate_action(o)
def main():
 tag=os.environ.get('FINAL_TAG','full');out=ROOT/'final_test_evaluation'/f'worker_core_{tag}';out.mkdir(parents=True,exist_ok=False)
 os.chdir(M6)  # frozen H=2 context intentionally resolves its cache relative to M6
 banks=make_banks(out)
 inv=json.loads((M6/'docs/milestone6/M6_DEPENDENCY_INVENTORY_FULL.json').read_text())['m5']['all_40_checkpoints']
 rows=[]
 for k in (1,2):
  if os.environ.get('FINAL_K') and k!=int(os.environ['FINAL_K']):continue
  for reg in REGIMES:
   if os.environ.get('FINAL_REG') and reg!=os.environ['FINAL_REG']:continue
   bank=load_scenario_bank(out/f'rl_test_K{k}_{reg}_final_v1.json')
   policies=[('h2',None),('m4',None)]+[('m5',x) for x in inv if int(x['k'])==k and x['cost_regime']=={'failure-light-no-waste':'light','failure-heavy-no-waste':'heavy','failure-light-waste-aware':'light_waste','failure-heavy-waste-aware':'heavy_waste'}[reg]]
   assert len(policies)==7
   for method,ck in policies:
    if os.environ.get('FINAL_METHOD') and method!=os.environ['FINAL_METHOD']:continue
    choose=chooser(method,k,reg,ck)
    for sc in bank.scenarios:
     env=SelectiveMaintenanceEnv(get_default_config(split='rl_test',cost_regime_id=reg,maintenance_capacity=k,scenario_bank_path=str(out/f'rl_test_K{k}_{reg}_final_v1.json'),prediction_cache_path=str(M6/'data/processed/fd001/v2/06_PREDICTIONS'),seed=sc.environment_seed,info_mode='normal'),scenario_bank=bank)
     obs,_=env.reset(seed=sc.environment_seed,options={'scenario_id':sc.scenario_id}); tot=dict(preventive_cost=0.,failure_cost=0.,wasted_life_cost=0.,total_cost=0.,preventive_count=0,failure_count=0)
     term=trunc=False
     while not(term or trunc):
      q=choose(obs); act=q.action_id if method in ('h2','m4') else q
      obs,_,term,trunc,info=env.step(act)
      for z in ('preventive_cost','failure_cost','wasted_life_cost','total_cost'):tot[z]+=float(info[z])
      tot['preventive_count']+=int(info['num_preventive']);tot['failure_count']+=int(info['num_failures'])
     rows.append({'method':method,'training_seed':'' if ck is None else ck['training_seed'],'K':k,'cost_regime':reg,'scenario_id':sc.scenario_id,'checkpoint_sha256':'' if ck is None else ck['checkpoint_best_sha256'],**tot})
 with open(out/'core_episode_metrics.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 (out/'run_manifest.json').write_text(json.dumps({'split':'rl_test','rows':len(rows),'banks':banks,'m6_head':'07c40697e4e80f2fc897cfa10583942ec06bf770','no_training':True},indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
