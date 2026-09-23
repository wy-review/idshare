from pathlib import Path
import ast
import importlib.util
import json
import statistics
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from make_catalog import build_catalog
from run import sha


class ReleaseTests(unittest.TestCase):
    def test_sources_are_traceable_and_parse(self):
        for record in json.loads((ROOT/'provenance/SOURCES.json').read_text()):
            path=ROOT/record['path']
            self.assertEqual(sha(path),record['public_sha256'],str(path.relative_to(ROOT)))
        for path in ROOT.rglob('*.py'):
            ast.parse(path.read_text(),filename=str(path))

    def test_catalog(self):
        rows=build_catalog()
        self.assertEqual(len({r['id'] for r in rows}),len(rows))
        self.assertEqual(rows,json.loads((ROOT/'configs/experiments.json').read_text()))
        self.assertEqual(sum(r['family'] in ['l2','additional_backbone'] for r in rows),168)
        extra=[r for r in rows if r['family']=='additional_backbone' and r['setting']=='kuairand']
        self.assertEqual({r['seed'] for r in extra},{42,2024,933888})
        for row in rows:
            if row['reuse']:
                reused=next(r for r in rows if r['id']==row['reuse'])
                for k in ['method','setting','seed','backbone','k','depth','global_l2','alpha']:
                    self.assertEqual(row[k],reused[k])

    def test_l2_reference_means(self):
        for setting in ['taac','kuairand']:
            data=json.loads((ROOT/'reference'/f'{setting}_main.json').read_text())
            self.assertEqual(len(data['per_dose']),7)
            for row in data['per_dose'].values():
                for method in ['continuous','idshare']:
                    values=[x[method+'_validation_auc'] for x in row['per_seed'].values()]
                    self.assertAlmostEqual(statistics.mean(values),row[method]['mean'],places=13)
                    self.assertAlmostEqual(statistics.stdev(values),row[method]['sample_std'],places=13)

    def test_taac_split_and_metadata_synthetic(self):
        import pickle
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
        sys.path.insert(0,str(ROOT/'preprocessing/taac'))
        import prepare
        import prepare_time_split as original
        seq=[dict(item_id=1,action_type=0,timestamp=1747929599),
             dict(item_id=2,action_type=1,timestamp=1747929600),
             dict(item_id=3,action_type=0,timestamp=1748448000),
             dict(item_id=4,action_type=1,timestamp=1748534400)]
        kept,rows=prepare.split_sequence(seq)
        self.assertEqual(len(kept),3)
        self.assertEqual(rows,{'train':[(2,1,1)],'val':[(3,0,2)]})
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);raw=root/'raw';(raw/'seq').mkdir(parents=True)
            for name in ['item_feat','user_feat']: (raw/name).mkdir()
            pq.write_table(pa.table(dict(user_id=[1],seq=[seq]),schema=original.USER_SEQ_SCHEMA),raw/'seq/part.parquet')
            fids=[100,101,102,103,104,105,106,107,108,109,110,112,114,115,116,117,118,119,120,121,122]
            with (raw/'indexer.pkl').open('wb') as f:
                pickle.dump(dict(u={'u':1},i={str(i):i for i in range(1,5)},f={str(i):{'v':1} for i in fids}),f)
            self.assertEqual(prepare.prepare(raw,root/'prepared'),{'train':1,'val':1})
            report=prepare.metadata(root/'prepared',root/'features',allow_small=True)
            mask=np.load(report['seen_mask']['path'])
            self.assertEqual(mask.tolist(),[False,True,True,False,False])
            self.assertFalse((root/'prepared/test').exists())

    def test_kuairand_train_fitted_vocabulary(self):
        path=ROOT/'preprocessing/kuairand/prepare_k1_nonseq_mmap.py'
        spec=importlib.util.spec_from_file_location('kuai_preprocess',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertEqual(len(module.FIELD_NAMES),37)
        expected=module.load_expected_split(str(ROOT/'preprocessing/kuairand/SHRED_ZA_P37_ROLLING_BACKTEST_SPLIT.json'))
        self.assertEqual(expected['train_rows'],243480365)
        self.assertEqual(expected['val_rows'],12080526)
        self.assertFalse(set(expected['train_days']) & set(expected['val_days']))

    def test_kuairand_preprocessing_synthetic(self):
        import csv
        from datetime import datetime, timezone
        sys.path.insert(0,str(ROOT/'preprocessing/kuairand'))
        spec=importlib.util.spec_from_file_location('kuai_public',ROOT/'preprocessing/kuairand/prepare.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        old=module.original
        expected=old.load_expected_split(str(module.SPEC))
        def write_csv(path, rows):
            with path.open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);raw=root/'raw';raw.mkdir()
            write_csv(raw/'user_features_27k.csv',[dict(user_id=1,**{k:'1' for k in module.audit.USER_CATEGORICAL_FIELDS})])
            write_csv(raw/'video_features_basic_27k.csv',[
                dict(video_id=i,**{k:'1' for k in module.audit.VIDEO_CATEGORICAL_FIELDS},
                     **{k:'10' for k in old.VIDEO_NUMERIC_FIELDS}) for i in [1,2]])
            rows=[]
            for day,vid in [(expected['train_days'][0],1),(expected['val_days'][0],2),('2030-01-01',2)]:
                ms=int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp()*1000)
                rows.append(dict(user_id=1,video_id=vid,time_ms=ms,
                            **{k:1 for k in module.audit.BINARY_LABELS},play_time_ms=1000,duration_ms=1000))
            rows[-1]['is_click']='not-read-outside-selected-dates'
            write_csv(raw/'log_standard_fixture.csv',rows)
            manifest=json.loads(module.prepare(raw,root/'processed',min_free_gb=0,
                                              shard_rows=2,allow_small=True).read_text())
            self.assertEqual(set(manifest['splits']),{'train','val'})
            self.assertEqual(manifest['splits']['train']['rows'],1)
            self.assertEqual(manifest['splits']['val']['rows'],1)
            self.assertEqual(manifest['cardinalities'][manifest['field_names'].index('video_id')],5)
            self.assertEqual(len(manifest['field_names']),37)
            self.assertFalse(list((root/'processed').glob('test*')))


if __name__=='__main__': unittest.main()
