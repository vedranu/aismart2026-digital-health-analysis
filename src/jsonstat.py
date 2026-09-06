import json, itertools
import pandas as pd

def load(path):
    d = json.load(open(path))
    ids, size, dims, val = d['id'], d['size'], d['dims'], d['value']
    # inverse index per dimension
    inv = {k: {v: kk for kk, v in dims[k].items()} for k in ids}
    rows = []
    for pos, v in val.items():
        pos = int(pos); idx = []
        for s in reversed(size):
            idx.append(pos % s); pos //= s
        idx = list(reversed(idx))
        rows.append({ids[i]: inv[ids[i]][idx[i]] for i in range(len(ids))} | {'value': v})
    return pd.DataFrame(rows)

if __name__ == '__main__':
    import sys
    for p in sys.argv[1:]:
        df = load(p)
        print(p, df.shape)
        print(df[df.geo == 'HR'].to_string())
