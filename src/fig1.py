import os, matplotlib; matplotlib.use('Agg')
FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures', '')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
fig = plt.figure(figsize=(7.4, 4.4)); ax = fig.add_axes([0.01, 0.01, 0.98, 0.98]); ax.set_xlim(0, 10); ax.set_ylim(0.7, 6.55); ax.axis('off')
def box(x, y, w, h, title, body, fc, ts=7.6, bs=6.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12', fc=fc, ec='#333333', lw=.9))
    ax.text(x + w/2, y + h - .27, title, ha='center', va='center', fontsize=ts, fontweight='bold')
    ax.text(x + w/2, y + (h - .55)/2, body, ha='center', va='center', fontsize=bs, linespacing=1.4)
ax.add_patch(FancyBboxPatch((0.2, 5.75), 9.6, .65, boxstyle='round,pad=0.02,rounding_size=0.1', fc='#f2f2f2', ec='#555555', lw=.8, ls='--'))
ax.text(5, 6.07, 'System context: demographic ageing  ·  health workforce shortages  ·  rising demand for timely care  ·  EHDS and AI Act', ha='center', va='center', fontsize=6.9, style='italic')
box(0.2, 3.35, 2.25, 2.2, '1. Detection layer', 'AI models for early\ndetection (imaging,\nvoice, biomarkers);\nrisk stratification of\nchronic populations', '#dbe9f6')
box(2.65, 3.35, 2.25, 2.2, '2. Interaction layer', 'triage and symptom\nassessment support;\npersonalised preventive\ncommunication; remote\npatient monitoring', '#dbe9f6')
box(5.1, 3.35, 2.25, 2.2, '3. Organisational layer', 'integration with\nprimary care; workflow\nand skill-mix redesign;\ntargeting and enrolment;\ndata governance, trust', '#fde9d9')
box(7.55, 3.35, 2.25, 2.2, '4. Value layer', 'outcomes (preventable\nmortality, avoidable\nadmissions); access\n(unmet needs); efficiency\n(cost per outcome)', '#e2f0d9')
for x in [2.45, 4.9, 7.35]:
    ax.add_patch(FancyArrowPatch((x - .02, 4.45), (x + .22, 4.45), arrowstyle='-|>', mutation_scale=11, color='#333333', lw=1.0))
box(0.2, 0.85, 4.7, 1.75, 'Enablers (supply side)', 'online access to electronic health records\n(Digital Decade e-health score); interoperability\nand patient portals; platform business model\nand financing; professional and digital skills', '#ffffff')
box(5.1, 0.85, 4.7, 1.75, 'Adoption (demand side)', 'citizens accessing records and booking\nappointments online; health-information seeking;\ndigital divide by age, education and income;\nengagement and adherence over time', '#ffffff')
for x0 in [2.55, 7.45]:
    ax.add_patch(FancyArrowPatch((x0, 2.62), (x0, 3.32), arrowstyle='-|>', mutation_scale=11, color='#333333', lw=1.0))
ax.add_patch(FancyArrowPatch((9.1, 3.33), (9.1, 2.62), arrowstyle='-|>', mutation_scale=11, color='#7f7f7f', lw=1.0, ls='--'))
ax.text(9.2, 2.98, 'evidence,\nreimbursement', fontsize=5.8, color='#555555', va='center')
plt.savefig(FIG + 'Figure1_framework.png', dpi=300); plt.savefig(FIG + 'Figure1_framework.tif', dpi=300, pil_kwargs={'compression': 'tiff_lzw'})
