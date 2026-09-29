"""Cross-country analysis EU-27 + Croatian budget-impact scenario.
All inputs fetched 2026-09-06 from Eurostat / DESI / OECD APIs (see data/)."""
import json, warnings
import numpy as np, pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from jsonstat import load

warnings.filterwarnings('ignore')
rng = np.random.default_rng(2026)
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, 'data', ''); OUT = os.path.join(ROOT, 'results', ''); FIG = os.path.join(ROOT, 'figures', '')
EU27 = ['BE','BG','CZ','DK','DE','EE','IE','EL','ES','FR','HR','IT','CY','LV','LT','LU','HU','MT','NL','AT','PL','PT','RO','SI','SK','FI','SE']

def wide(df, filt, name):
    d = df.copy()
    for k, v in filt.items(): d = d[d[k] == v]
    d = d[d.geo.isin(EU27)][['geo', 'time', 'value']].rename(columns={'value': name})
    d['time'] = d.time.astype(int)
    return d

isoc1 = load(D + 'raw/raw_isoc_iuapr.json'); isoc2 = load(D + 'raw/raw_isoc_iumapp_ihif.json')
silc = load(D + 'raw/raw_silc08.json'); mort = load(D + 'raw/raw_cd_apr.json'); sha = load(D + 'raw/raw_sha11_hc.json')
prs = load(D + 'raw/raw_rs_prs2.json'); gdp = load(D + 'raw/raw_gdp.json'); age = load(D + 'raw/raw_age65.json')
desi = pd.read_csv(D + 'desi_aehr.csv').melt(id_vars='geo', var_name='time', value_name='desi_aehr')
desi['time'] = desi.time.astype(int); desi = desi[desi.geo.isin(EU27)]

parts = {
 'iuapr': wide(isoc1, {'indic_is': 'I_IUAPR'}, 'iuapr'),
 'iumapp': wide(isoc2, {'indic_is': 'I_IUMAPP'}, 'iumapp'),
 'ihif': wide(isoc2, {'indic_is': 'I_IHIF'}, 'ihif'),
 'unmet': wide(silc, {}, 'unmet'),
 'prev_mort': wide(mort, {'mortalit': 'PRVT'}, 'prev_mort'),
 'treat_mort': wide(mort, {'mortalit': 'TRT'}, 'treat_mort'),
 'hc6_share': wide(sha, {'unit': 'PC_CHE', 'icha11_hc': 'HC6'}, 'hc6_share'),
 'che_pps': wide(sha, {'unit': 'PPS_HAB', 'icha11_hc': 'TOT_HC'}, 'che_pps'),
 'phys': wide(prs, {'med_spec': 'PHYS'}, 'phys'),
 'nurses': wide(prs, {'med_spec': 'NRS'}, 'nurses'),
 'gdp_pps': wide(gdp, {}, 'gdp_pps'),
 'age65': wide(age, {}, 'age65'),
 'desi_aehr': desi,
}
long = None
for k, d in parts.items():
    long = d if long is None else long.merge(d, on=['geo', 'time'], how='outer')
long = long.sort_values(['geo', 'time'])
long.to_csv(OUT + 'panel_long.csv', index=False)

# ---------- latest cross-section ----------
def latest(name, year):
    d = parts[name]; d = d[d.time == year].set_index('geo')[name]; return d
cs = pd.DataFrame({
 'iuapr': latest('iuapr', 2024), 'iumapp': latest('iumapp', 2024), 'ihif': latest('ihif', 2024),
 'desi_aehr': latest('desi_aehr', 2024), 'unmet': latest('unmet', 2024),
 'prev_mort': latest('prev_mort', 2023), 'treat_mort': latest('treat_mort', 2023),
 'hc6_share': latest('hc6_share', 2023), 'che_pps': latest('che_pps', 2023),
 'phys': latest('phys', 2023), 'nurses': latest('nurses', 2023), 'gdp_pps': latest('gdp_pps', 2024), 'age65': latest('age65', 2024),
}).loc[EU27]
# fill few gaps in physicians/nurses 2023 with 2022
for v in ['phys', 'nurses', 'hc6_share', 'che_pps']:
    prev = latest(v, 2022); cs[v] = cs[v].fillna(prev)
cs.to_csv(OUT + 'cross_section_latest.csv')
print('missing per var:\n', cs.isna().sum())

labels = {'iuapr': 'Access to own health records online (% ind., 2024)', 'iumapp': 'Online appointment with practitioner (% ind., 2024)',
          'ihif': 'Seeking health information online (% ind., 2024)', 'desi_aehr': 'Digital Decade e-health records score (0-100, 2024)',
          'unmet': 'Unmet medical needs, cost/distance/waiting (% , 2024)', 'prev_mort': 'Preventable mortality (per 100 000, 2023)',
          'treat_mort': 'Treatable mortality (per 100 000, 2023)', 'hc6_share': 'Preventive care share of CHE (%, 2023)',
          'che_pps': 'Current health expenditure (PPS per inhabitant, 2023)', 'phys': 'Practising physicians (per 100 000, 2023)',
          'nurses': 'Practising nurses (per 100 000, 2023)', 'gdp_pps': 'GDP per capita (PPS, 2024)', 'age65': 'Population aged 65+ (%, 2024)'}

# Table 1 descriptives
rows = []
for v in cs.columns:
    s = cs[v].dropna(); hr = cs.loc['HR', v]
    rank = int((s > hr).sum() + 1) if v not in ['unmet', 'prev_mort', 'treat_mort'] else int((s < hr).sum() + 1)
    rows.append([labels[v], f'{s.median():.1f}', f'{s.quantile(.25):.1f}–{s.quantile(.75):.1f}', f'{s.min():.1f}–{s.max():.1f}', f'{hr:.1f}', f'{rank}/{len(s)}'])
t1 = pd.DataFrame(rows, columns=['Indicator', 'Median', 'IQR', 'Min–max', 'Croatia', 'Croatia rank*'])
t1.to_csv(OUT + 'table1_descriptives.csv', index=False); print(t1.to_string())

# ---------- Spearman with bootstrap CI and Holm ----------
core = ['iuapr', 'iumapp', 'desi_aehr', 'hc6_share', 'che_pps', 'gdp_pps', 'age65', 'prev_mort', 'treat_mort', 'unmet']
def boot_spearman(x, y, B=2000):
    m = ~(np.isnan(x) | np.isnan(y)); x, y = x[m], y[m]; n = len(x)
    r, p = stats.spearmanr(x, y)
    idx = rng.integers(0, n, (B, n))
    rs = np.array([stats.spearmanr(x[i], y[i])[0] for i in idx])
    return r, p, np.nanpercentile(rs, 2.5), np.nanpercentile(rs, 97.5), n
outcomes = ['prev_mort', 'treat_mort', 'unmet']; preds = ['iuapr', 'iumapp', 'desi_aehr', 'hc6_share']
res = []
for o in outcomes:
    for pr in preds:
        r, p, lo, hi, n = boot_spearman(cs[pr].values.astype(float), cs[o].values.astype(float))
        res.append({'outcome': o, 'predictor': pr, 'rho': r, 'p': p, 'lo': lo, 'hi': hi, 'n': n})
res = pd.DataFrame(res)
# Holm
ps = res.p.values; order = np.argsort(ps); m = len(ps); adj = np.empty(m)
for k, i in enumerate(order): adj[i] = min(1, ps[i] * (m - k))
adj = np.maximum.accumulate(adj[order])[np.argsort(order)]
res['p_holm'] = adj
# partial Spearman controlling GDP (residual approach on ranks)
def partial_sp(x, y, z):
    df = pd.DataFrame({'x': x, 'y': y, 'z': z}).dropna().rank()
    rx = df.x - np.polyval(np.polyfit(df.z, df.x, 1), df.z); ry = df.y - np.polyval(np.polyfit(df.z, df.y, 1), df.z)
    return stats.pearsonr(rx, ry)[0]
res['rho_partial_gdp'] = [partial_sp(cs[r.predictor], cs[r.outcome], cs['gdp_pps']) for r in res.itertuples()]
res.to_csv(OUT + 'table2_spearman.csv', index=False); print(res.round(3).to_string())

# heatmap (Figure 2)
short = {'iuapr': 'Health records online', 'iumapp': 'Online appointment', 'desi_aehr': 'e-Health records score', 'hc6_share': 'Prevention share CHE',
         'che_pps': 'Health exp. per capita', 'gdp_pps': 'GDP per capita', 'age65': 'Population 65+', 'prev_mort': 'Preventable mortality',
         'treat_mort': 'Treatable mortality', 'unmet': 'Unmet medical needs'}
CM = cs[core].corr(method="spearman")
fig, ax = plt.subplots(figsize=(7.2, 6))
im = ax.imshow(CM.values, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(core))); ax.set_yticks(range(len(core)))
ax.set_xticklabels([short[c] for c in core], rotation=45, ha='right', fontsize=8); ax.set_yticklabels([short[c] for c in core], fontsize=8)
for i in range(len(core)):
    for j in range(len(core)):
        ax.text(j, i, f"{CM.values[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if abs(CM.values[i, j]) > .55 else 'black')
cb = plt.colorbar(im, ax=ax, fraction=0.046); cb.set_label("Spearman's ρ", fontsize=8); cb.ax.tick_params(labelsize=7)
plt.tight_layout(); plt.savefig(FIG + 'FigureS1_correlation_heatmap.png', dpi=300); plt.savefig(FIG + 'FigureS1_correlation_heatmap.tif', dpi=300, pil_kwargs={'compression': 'tiff_lzw'}); plt.close()

# ---------- PCA + clustering ----------
cl_vars = ['iuapr', 'iumapp', 'desi_aehr', 'hc6_share', 'prev_mort', 'unmet']
X = cs[cl_vars].dropna(); Z = StandardScaler().fit_transform(X)
pca = PCA(n_components=3).fit(Z); S = pca.transform(Z)
print('PCA explained variance:', pca.explained_variance_ratio_.round(3))
load = pd.DataFrame(pca.components_.T, index=cl_vars, columns=['PC1', 'PC2', 'PC3']); print(load.round(2))
L = linkage(Z, method='ward')
sil = {k: silhouette_score(Z, fcluster(L, k, 'maxclust')) for k in range(2, 7)}; print('silhouette:', {k: round(v, 3) for k, v in sil.items()})
k_best = max(sil, key=sil.get); k_use = 4  # interpretive solution; silhouette reported for all k
cl = fcluster(L, k_use, 'maxclust'); X['cluster'] = cl
# bootstrap Jaccard stability
def jaccard_stability(Z, L_ref, k, B=200):
    ref = fcluster(L_ref, k, 'maxclust'); n = len(Z); scores = []
    for _ in range(B):
        idx = rng.choice(n, n, replace=True); idxu = np.unique(idx)
        lab = fcluster(linkage(Z[idxu], 'ward'), k, 'maxclust')
        # best-match Jaccard per ref cluster
        js = []
        for c in np.unique(ref):
            A = set(np.where(ref[idxu] == c)[0]); best = 0
            for c2 in np.unique(lab):
                Bc = set(np.where(lab == c2)[0]); j = len(A & Bc) / len(A | Bc) if A | Bc else 0; best = max(best, j)
            js.append(best)
        scores.append(js)
    return np.array(scores).mean(axis=0)
jac = jaccard_stability(Z, L, k_use); print('Jaccard stability per cluster:', jac.round(2))
prof = X.groupby('cluster').agg(['mean']).round(1); prof.columns = [c[0] for c in prof.columns]
prof['n'] = X.groupby('cluster').size(); prof['countries'] = X.groupby('cluster').apply(lambda d: ', '.join(sorted(d.index)))
prof.to_csv(OUT + 'table3_clusters.csv'); print(prof.to_string())
X.to_csv(OUT + 'clusters.csv')
json.dump({'explained': pca.explained_variance_ratio_.tolist(), 'silhouette': {int(k): float(v) for k, v in sil.items()}, 'k': int(k_use), 'jaccard': jac.tolist(),
           'loadings': load.round(3).to_dict()}, open(OUT + 'pca_cluster_meta.json', 'w'), indent=1)

# Figure 3: PCA biplot with clusters
fig, ax = plt.subplots(figsize=(7.2, 5.6))
markers = ['o', 's', '^', 'D', 'v', 'P']; cols = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#8c564b']
for c in sorted(np.unique(cl)):
    m = cl == c; ax.scatter(S[m, 0], S[m, 1], s=48, marker=markers[c - 1], c=cols[c - 1], label=f'Cluster {c}', edgecolor='k', linewidth=.4, zorder=3)
for i, g in enumerate(X.index):
    off = {'ES': (6, -9), 'SE': (6, -8), 'FR': (-13, 4), 'SI': (4, 5), 'PT': (5, -8), 'MT': (-14, 3), 'BE': (-12, 5), 'CY': (-13, -3), 'SK': (-13, 3)}.get(g, (4, 3))
    ax.annotate(g, (S[i, 0], S[i, 1]), xytext=off, textcoords='offset points', fontsize=7.5, fontweight='bold' if g == 'HR' else 'normal', color='black' if g != 'HR' else '#b30000')
sc = 2.6
for v in cl_vars:
    ax.arrow(0, 0, load.loc[v, 'PC1'] * sc, load.loc[v, 'PC2'] * sc, color='grey', width=.004, head_width=.08, alpha=.8, zorder=2)
    lx, ly = load.loc[v, 'PC1'] * sc * 1.15, load.loc[v, 'PC2'] * sc * 1.15
    dx, dy, ha = {'iumapp': (0.35, -0.12, 'left'), 'hc6_share': (-0.8, -0.14, 'left'), 'iuapr': (0.25, 0.08, 'left'), 'unmet': (0.2, 0.0, 'left'), 'desi_aehr': (0.15, 0.05, 'left')}.get(v, (0, 0, 'center'))
    ax.text(lx + dx, ly + dy, short[v], fontsize=7, color='dimgrey', ha=ha, bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8))
ax.axhline(0, color='lightgrey', lw=.6); ax.axvline(0, color='lightgrey', lw=.6)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% of variance)', fontsize=9); ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% of variance)', fontsize=9)
ax.legend(fontsize=8, loc='best', frameon=False); ax.tick_params(labelsize=8)
plt.tight_layout(); plt.savefig(FIG + 'Figure2_pca_clusters.png', dpi=300); plt.savefig(FIG + 'Figure2_pca_clusters.tif', dpi=300, pil_kwargs={'compression': 'tiff_lzw'}); plt.close()

# ---------- Panel FE (waves 2020, 2022, 2024; mortality lagged to 2023 for 2024 wave) ----------
pan = long[long.time.isin([2020, 2022, 2024])].copy()
m23 = parts['prev_mort'][parts['prev_mort'].time == 2023].set_index('geo').prev_mort
t23 = parts['treat_mort'][parts['treat_mort'].time == 2023].set_index('geo').treat_mort
h23 = parts['hc6_share'][parts['hc6_share'].time == 2023].set_index('geo').hc6_share
c23 = parts['che_pps'][parts['che_pps'].time == 2023].set_index('geo').che_pps
for col, ser in [('prev_mort', m23), ('treat_mort', t23), ('hc6_share', h23), ('che_pps', c23)]:
    pan.loc[pan.time == 2024, col] = pan.loc[pan.time == 2024, 'geo'].map(ser).values
pan['ln_gdp'] = np.log(pan.gdp_pps); pan['ln_prev'] = np.log(pan.prev_mort); pan['ln_che'] = np.log(pan.che_pps)
pan = pan.dropna(subset=['iuapr', 'prev_mort', 'unmet', 'gdp_pps', 'age65', 'hc6_share'])
print('panel obs:', len(pan), 'countries:', pan.geo.nunique())
fe = []
for y in ['ln_prev', 'unmet']:
    for x in ['iuapr', 'iumapp']:
        f = f'{y} ~ {x} + ln_gdp + age65 + hc6_share + C(geo) + C(time)'
        mdl = smf.ols(f, data=pan).fit(cov_type='cluster', cov_kwds={'groups': pan.geo})
        pooled = smf.ols(f'{y} ~ {x} + ln_gdp + age65 + hc6_share + C(time)', data=pan).fit(cov_type='cluster', cov_kwds={'groups': pan.geo})
        fe.append({'outcome': y, 'predictor': x, 'beta_FE': mdl.params[x], 'se_FE': mdl.bse[x], 'lo': mdl.conf_int().loc[x, 0], 'hi': mdl.conf_int().loc[x, 1], 'p_FE': mdl.pvalues[x],
                   'beta_pooled': pooled.params[x], 'p_pooled': pooled.pvalues[x], 'r2_within': mdl.rsquared, 'n': int(mdl.nobs)})
fe = pd.DataFrame(fe); fe.to_csv(OUT + 'table4_panel.csv', index=False); print(fe.round(4).to_string())

# ---------- Monte Carlo budget impact for Croatia ----------
P = json.load(open(D + 'params_hr.json'))
pop15 = P['eurostat_demo_pjanbroad_hr_2024']['Y15-64'] + P['eurostat_demo_pjanbroad_hr_2024']['Y_GE65']
rates = P['oecd_avoidable_admissions_hr_2021_per100k_15plus']
base_adm = sum(rates[k] for k in ['ADMRCHFL_congestive_heart_failure', 'ADMRDBUC_diabetes', 'ADMRCOPD_copd', 'ADMRHYPT_hypertension']) * pop15 / 1e5
cost_2019 = P['eurostat_hlth_sha11_hc_hr_mio_eur']['HC11_2019'] * 1e6 / P['eurostat_hlth_co_disch1_hr']['all_causes_2019']
infl = P['eurostat_hlth_sha11_hc_hr_mio_eur']['HC11_2023'] / P['eurostat_hlth_sha11_hc_hr_mio_eur']['HC11_2019']
cost_base = cost_2019 * infl
print(f'pop15+={pop15:,.0f} base_adm={base_adm:,.0f} cost2019={cost_2019:,.0f} infl={infl:.3f} cost_base={cost_base:,.0f}')
N = 10000
params = {
 'Baseline avoidable admissions (n/yr)': rng.uniform(base_adm * .8, base_adm * 1.2, N),
 'Admissions per high-risk patient (n/yr)': rng.uniform(1.0, 1.5, N),
 'High-risk patients enrolled (share)': rng.uniform(.20, .60, N),
 'Relative reduction in admissions (enrolled)': rng.triangular(.05, .20, .35, N),
 'Cost per avoided admission (EUR)': rng.triangular(cost_base * .7, cost_base, cost_base * 1.6, N),
 'Platform cost per enrolled patient (EUR/yr)': rng.triangular(150, 300, 600, N),
}
def model(p):
    patients = p['Baseline avoidable admissions (n/yr)'] / p['Admissions per high-risk patient (n/yr)']
    enrolled = patients * p['High-risk patients enrolled (share)']
    avoided = enrolled * p['Admissions per high-risk patient (n/yr)'] * p['Relative reduction in admissions (enrolled)']
    gross = avoided * p['Cost per avoided admission (EUR)']
    prog = enrolled * p['Platform cost per enrolled patient (EUR/yr)']
    breakeven = p['Relative reduction in admissions (enrolled)'] * p['Admissions per high-risk patient (n/yr)'] * p['Cost per avoided admission (EUR)']
    return enrolled, avoided, gross, prog, gross - prog, breakeven
enrolled, avoided, gross, prog, net, be = model(params)
def q(a): return np.percentile(a, [2.5, 50, 97.5])
mc = {'enrolled': q(enrolled).tolist(), 'avoided_admissions': q(avoided).tolist(), 'gross_savings_meur': (q(gross) / 1e6).tolist(),
      'programme_cost_meur': (q(prog) / 1e6).tolist(), 'net_meur': (q(net) / 1e6).tolist(), 'p_net_positive': float((net > 0).mean()),
      'breakeven_price_eur': q(be).tolist(), 'base_adm': base_adm, 'cost_base': cost_base, 'cost_2019': cost_2019, 'infl': infl, 'pop15': pop15,
      'param_summary': {k: [float(np.percentile(v, 10)), float(np.median(v)), float(np.percentile(v, 90))] for k, v in params.items()}}
print(json.dumps(mc, indent=1))
med = {k: np.median(v) for k, v in params.items()}
def net_of(p): return model({k: np.array([v]) for k, v in p.items()})[4][0] / 1e6
base_net = net_of(med)
torn = []
for k, v in params.items():
    lo_p, hi_p = dict(med), dict(med); lo_p[k] = np.percentile(v, 10); hi_p[k] = np.percentile(v, 90)
    torn.append((k, net_of(lo_p), net_of(hi_p)))
torn = sorted(torn, key=lambda t: abs(t[2] - t[1]))
mc['tornado'] = torn; mc['base_net'] = base_net
json.dump(mc, open(OUT + 'mc_results.json', 'w'), indent=1)
fig, ax = plt.subplots(figsize=(7.2, 3.6))
for i, (k, lo, hi) in enumerate(torn):
    ax.barh(i, hi - base_net, left=base_net, color='#2c7fb8', height=.6)
    ax.barh(i, lo - base_net, left=base_net, color='#2c7fb8', height=.6)
    ax.text(max(lo, hi) + .05, i, f'{min(lo,hi):.2f} to {max(lo,hi):.2f}', va='center', fontsize=7.5)
ax.set_yticks(range(len(torn))); ax.set_yticklabels([t[0] for t in torn], fontsize=8)
ax.axvline(base_net, color='k', lw=.8); ax.axvline(0, color='grey', lw=.6, ls='--')
ax.set_xlabel(f'Net annual budget impact, million EUR (base case {base_net:.2f})', fontsize=8)
ax.tick_params(labelsize=8); ax.set_xlim(min(t[1] for t in torn + [(0, 0, 0)]) - .15, max(t[2] for t in torn) + 0.35)
plt.tight_layout(); plt.savefig(FIG + 'Figure3_tornado.png', dpi=300, bbox_inches='tight'); plt.savefig(FIG + 'Figure3_tornado.tif', dpi=300, bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'}); plt.close()
print('done')

# alternative k profiles
for k in [3, 4]:
    lab = fcluster(L, k, 'maxclust'); Xk = X[cl_vars].copy(); Xk['cluster'] = lab
    pk = Xk.groupby('cluster').mean().round(1); pk['n'] = Xk.groupby('cluster').size(); pk['countries'] = Xk.groupby('cluster').apply(lambda d: ', '.join(sorted(d.index)))
    print(f'--- k={k}'); print(pk.to_string())
