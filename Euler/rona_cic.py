'''
CIC-IDS-2017 (dataset 3) entry layer: reuses the LANL code path (rona_lanl) with CIC defaults.
No existing file is modified. Data is produced by a CIC-IDS2017 preparation script into
./data/cic (fl_lanl/ folder name kept so loaders/load_lanl.py runs unchanged).

    import rona_cic; rona_cic.patch(); import rona_run as R; R.main()

Differences to LANL: tag prefix C_ (instead of L_), cluster file cic_<K>_<K>.json, cache dir
./data/rona_cache_cic/, delta=600 s, train = Monday (tr_end=30000 in the compressed time axis),
val = last 3600 s of Monday, test = Tue-Fri. All values come from cic_meta.json and can be overridden
with the --lanl_* flags (or env CIC_DATA).
'''
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rona_lanl

CIC_DATA = os.environ.get('CIC_DATA', './data/cic')
_meta = json.load(open(os.path.join(CIC_DATA, 'cic_meta.json')))

rona_lanl.LC.update({'data': CIC_DATA, 'delta': _meta['delta'], 'tr_end': _meta['tr_end'],
                     'val': _meta['val'], 'te_start': _meta['te_start'], 'te_end': _meta['te_end'],
                     'chunk': 50, 'cache': './data/rona_cache_cic/'})


def prefix_tag(argv, flags=('--tag', '--transfer_from')):
    argv = list(argv)
    for f in flags:
        if f in argv:
            i = argv.index(f) + 1
            if not argv[i].startswith('C_'):
                argv[i] = 'C_' + argv[i]
    return argv


_orig_prep = rona_lanl._prep


def _prep(argv):
    argv = list(argv)
    if '--cluster_fname' not in argv:
        K = int(argv[argv.index('--K') + 1]) if '--K' in argv else 3
        argv += ['--cluster_fname', 'cic_%d_%d.json' % (K, K)]
    return _orig_prep(argv)


def patch():
    rona_lanl.prefix_tag = prefix_tag
    rona_lanl._prep = _prep
    return rona_lanl.patch()
