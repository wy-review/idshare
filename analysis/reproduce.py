#!/usr/bin/env python3
"""Recompute reference tables and draw L2/robustness/frequency figures offline."""
from pathlib import Path
import argparse
import csv
import json
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run import sha
from readouts import intervention_summary


def read(name):
    path=ROOT/'reference'/name
    sources=json.loads((ROOT/'provenance/SOURCES.json').read_text())
    record=next(r for r in sources if r['path']==f'reference/{name}')
    if sha(path)!=record['public_sha256']: raise ValueError(f'Reference hash mismatch: {name}')
    return json.loads(path.read_text())


def regularizers(out):
    audit=read('generated_regularization_baselines.json')
    rows=[]
    for setting,result in audit['results'].items():
        for method in ['continuous','tailshare','frozenhash','adamar','idshare']:
            values=[r[method] for r in result['per_seed'].values()]
            mean,sd=statistics.mean(values),statistics.stdev(values)
            if abs(mean-result['summary'][method]['mean'])>1e-12: raise ValueError('Mean audit failed')
            rows.append(dict(dataset=setting,method=method,seeds=len(values),auc=mean,auc_percent=mean*100,sd=sd))
    with (out/'regularizers.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    return rows


def figures(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    blue,gray='#2B5E74','#777777'
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(7,2.6))
    for setting,ax in zip(['taac','kuairand'],axes):
        doses=read(setting+'_main.json')['per_dose']
        keys=sorted(doses,key=float)
        for method,color,marker in [('continuous',gray,'o'),('idshare',blue,'D')]:
            means=[doses[k][method]['mean']*100 for k in keys]
            sd=[doses[k][method]['sample_std']*100 for k in keys]
            ax.errorbar(range(len(keys)),means,yerr=sd,color=color,marker=marker,capsize=2,label=method)
        ax.set_xticks(range(len(keys)),keys,rotation=30);ax.set_xlabel('Global L2 coefficient')
        ax.set_title('TAAC' if setting=='taac' else 'KuaiRand');ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('AUC (%)');axes[0].legend(frameon=False)
    fig.tight_layout();fig.savefig(out/'l2_response.pdf');plt.close(fig)
    r=read('generated_taac_robustness.json')
    fig,axes=plt.subplots(1,2,figsize=(7,2.6),sharey=True)
    for ax,key,xfield,xlabel in zip(axes,['k_rows','depth_rows'],['k','depth'],['Codebook size K','Backbone depth']):
        rows=r[key];xs=range(len(rows))
        for method,color,marker in [('continuous',gray,'o'),('idshare',blue,'D')]:
            ax.errorbar(xs,[v[method+'_mean']*100 for v in rows],yerr=[v[method+'_sd']*100 for v in rows],
                        color=color,marker=marker,capsize=2,label=method)
        ax.set_xticks(list(xs),[v[xfield] for v in rows]);ax.set_xlabel(xlabel);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('AUC (%)');axes[0].legend(frameon=False)
    fig.tight_layout();fig.savefig(out/'capacity_depth.pdf');plt.close(fig)
    freq=read('frequency.json')['summary']
    buckets=['0','1','2-5','6-10','11-50','51-100','101-500','501+']
    fig,ax=plt.subplots(figsize=(7,2.4))
    ax.errorbar(range(8),[freq[k]['paired']['auc']['mean']*100 for k in buckets],
                yerr=[freq[k]['paired']['auc']['sample_sd']*100 for k in buckets],fmt='D-',color=blue,capsize=3)
    ax.axhline(0,color=gray,linewidth=.7);ax.set_xticks(range(8),buckets)
    ax.set(xlabel='Training lookup count',ylabel='Paired AUC difference (pp)')
    fig.tight_layout();fig.savefig(out/'frequency.pdf');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--figures',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    rows=regularizers(a.output)
    evidence=read('continuation.json')
    reports={str(v['seed']):v for v in evidence['per_seed'].values()}
    summary=intervention_summary(reports)
    for step,splits in summary.items():
        for split,buckets in splits.items():
            for bucket,metrics in buckets.items():
                for metric,row in metrics.items():
                    if row is not None:
                        for arm in ['shared','untied']:
                            assert abs(row[arm]['mean']-evidence['summary'][step][split][bucket][metric][arm]['mean'])<1e-12
    (a.output/'intervention.json').write_text(json.dumps(summary,indent=2))
    (a.output/'verification.json').write_text(json.dumps(dict(regularizer_rows=len(rows),intervention_recomputed=True,
        metrics=['auc','logloss'],auc_bucket_attribution=False),indent=2))
    if a.figures: figures(a.output)
    print('Reference table and intervention summaries verified. No datasets loaded.')


if __name__=='__main__': main()
