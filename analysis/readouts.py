"""Offline readers: structure first, then equal-seed metric summaries."""
import math
import statistics


def stats(values):
    if not values or any(v is None for v in values): return {'mean':None, 'sample_sd':None}
    if not all(math.isfinite(v) for v in values): raise ValueError('Non-finite metric')
    return {'mean':statistics.mean(values), 'sample_sd':statistics.stdev(values) if len(values)>1 else None}


def check_intervention(reports, seeds=(2021,42,2024), steps=(0,1024,4096)):
    if set(reports) != {str(s) for s in seeds}: raise ValueError('Incomplete/duplicate seed set')
    for seed, report in reports.items():
        assert report['status']=='completed' and report['seed']==int(seed)
        assert report['frozen_parameters_unchanged'] and report['same_batches_for_both_arms']
        assert not report['best_step_selected'] and not report.get('uses_test_dataset', False)
        assert report['initial']['pair_prediction_max_error'] <= 1e-7
        assert report['initial']['native_embedding_max_error'] <= 1e-6
        assert set(report['stages']) == {str(s) for s in steps}
        for stage in report['stages'].values():
            for split in ['train_probe','validation']:
                pair = stage[split]['arms']
                a,b = (pair[n]['metrics'] for n in ['shared','untied'])
                assert a['overall']['n']==b['overall']['n']==stage[split]['rows']
                assert set(a['buckets'])==set(b['buckets'])
                for bucket in a['buckets']:
                    assert all(a['buckets'][bucket][k]==b['buckets'][bucket][k] for k in ['n','positive','negative'])
    return {'structural_pass':True, 'metrics_inspected':False}


def intervention_summary(reports, seeds=(2021,42,2024), steps=(0,1024,4096)):
    check_intervention(reports,seeds,steps)
    summary = {}
    for step in steps:
        summary[str(step)] = {}
        for split in ['train_probe','validation']:
            first=reports[str(seeds[0])]['stages'][str(step)][split]['arms']['shared']['metrics']
            rows = {}
            for bucket in ['overall',*first['buckets']]:
                row={}
                for metric in ['auc','logloss']:
                    values={}
                    for arm in ['shared','untied']:
                        vals=[]
                        for seed in seeds:
                            m=reports[str(seed)]['stages'][str(step)][split]['arms'][arm]['metrics']
                            vals.append((m['overall'] if bucket=='overall' else m['buckets'][bucket])[metric])
                        values[arm]=vals
                    if any(v is None for vs in values.values() for v in vs): row[metric]=None; continue
                    diff=[a-b for a,b in zip(values['shared'],values['untied'])]
                    row[metric]={a:stats(v) for a,v in values.items()}
                    row[metric]['paired_shared_minus_untied']={**stats(diff),'per_seed':dict(zip(map(str,seeds),diff)),
                        'shared_better_count':sum(v>0 if metric=='auc' else v<0 for v in diff)}
                rows[bucket]=row
            summary[str(step)][split]=rows
    return summary
