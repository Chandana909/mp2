"""Write the complete ML Platform dashboard to serving/static/index.html"""
from pathlib import Path

OUT = Path(__file__).parent.parent / "serving" / "static" / "index.html"

CSS = """
:root{--bg:#0d1117;--bg2:#161b22;--bg3:#21262d;--bd:rgba(255,255,255,.08);--bdh:rgba(255,255,255,.18);--tx:#e6edf3;--mu:#7d8590;--gr:#3fb950;--grb:rgba(63,185,80,.12);--grd:rgba(63,185,80,.3);--ye:#d29922;--yeb:rgba(210,153,34,.12);--yed:rgba(210,153,34,.3);--re:#f85149;--reb:rgba(248,81,73,.12);--red:rgba(248,81,73,.3);--bl:#58a6ff;--blb:rgba(88,166,255,.12);--bld:rgba(88,166,255,.3);--pu:#bc8cff;--pub:rgba(188,140,255,.12);--r:12px;--sh:0 8px 32px rgba(0,0,0,.4)}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--tx);font-family:'Inter',sans-serif;min-height:100vh;background-image:radial-gradient(circle at 0% 0%,rgba(88,166,255,.06) 0%,transparent 50%),radial-gradient(circle at 100% 100%,rgba(63,185,80,.04) 0%,transparent 50%);background-attachment:fixed}
a{color:inherit;text-decoration:none}
.layout{display:flex;min-height:100vh}
.sidebar{width:220px;background:var(--bg2);border-right:1px solid var(--bd);display:flex;flex-direction:column;position:sticky;top:0;height:100vh;overflow:hidden;flex-shrink:0;transition:width .25s}
.brand{padding:1.25rem 1rem;border-bottom:1px solid var(--bd);display:flex;align-items:center;gap:10px;font-weight:700;font-size:1rem}
.brand-icon{width:30px;height:30px;background:linear-gradient(135deg,var(--bl),var(--pu));border-radius:8px;display:flex;align-items:center;justify-content:center;flex-shrink:0}
.nav{display:flex;flex-direction:column;gap:2px;padding:.75rem .5rem;flex:1;overflow-y:auto}
.nav-item{display:flex;align-items:center;gap:10px;padding:.55rem .75rem;border-radius:8px;cursor:pointer;font-size:.88rem;font-weight:500;color:var(--mu);transition:all .15s;border:1px solid transparent;white-space:nowrap}
.nav-item:hover{background:var(--bg3);color:var(--tx)}
.nav-item.active{background:var(--blb);color:var(--bl);border-color:var(--bld)}
.nav-icon{width:16px;height:16px;flex-shrink:0}
.main{flex:1;display:flex;flex-direction:column;overflow:hidden}
.topbar{padding:.9rem 1.5rem;border-bottom:1px solid var(--bd);display:flex;align-items:center;justify-content:space-between;background:rgba(13,17,23,.9);backdrop-filter:blur(16px);position:sticky;top:0;z-index:50}
.page-title{font-size:1.05rem;font-weight:600}
.tbar-r{display:flex;align-items:center;gap:.6rem}
.status-pill{display:flex;align-items:center;gap:6px;padding:4px 12px;border-radius:99px;font-size:.78rem;font-weight:700;border:1px solid;letter-spacing:.3px;text-transform:uppercase}
.dot{width:6px;height:6px;border-radius:50%}
.content{padding:1.5rem;overflow-y:auto;flex:1}
.tab-pane{display:none;animation:fadeIn .25s ease}
.tab-pane.active{display:block}
@keyframes fadeIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
.btn{display:inline-flex;align-items:center;gap:5px;padding:5px 12px;font-size:.83rem;font-weight:500;border-radius:8px;border:1px solid var(--bd);background:var(--bg3);color:var(--tx);cursor:pointer;transition:all .15s}
.btn:hover{background:var(--bd);border-color:var(--bdh)}
.btn.pri{background:var(--bl);border-color:transparent;color:#fff;font-weight:600}
.btn.pri:hover{opacity:.88}
.btn.dan{background:var(--reb);border-color:var(--red);color:#ff7b72}
.btn.dan:hover{background:rgba(248,81,73,.22)}
.btn.suc{background:var(--grb);border-color:var(--grd);color:var(--gr)}
.btn.yel{background:var(--yeb);border-color:var(--yed);color:var(--ye)}
.grid4{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:1rem}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:1.25rem}
.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:1.25rem}
@media(max-width:900px){.grid2,.grid3{grid-template-columns:1fr}.sidebar{width:56px}.nav-item span{display:none}.brand span:last-child{display:none}}
.card{background:var(--bg2);border:1px solid var(--bd);border-radius:var(--r);padding:1.25rem;transition:border-color .2s}
.card:hover{border-color:var(--bdh)}
.card-hdr{display:flex;justify-content:space-between;align-items:center;padding-bottom:.9rem;border-bottom:1px solid var(--bd);margin-bottom:.9rem}
.card-title{font-size:.9rem;font-weight:600;display:flex;align-items:center;gap:7px}
.stat-val{font-size:2rem;font-weight:800;line-height:1;font-variant-numeric:tabular-nums}
.stat-lbl{font-size:.78rem;color:var(--mu);text-transform:uppercase;letter-spacing:.5px;margin-bottom:.35rem}
.stat-sub{font-size:.78rem;color:var(--mu);margin-top:.25rem}
.c-bl .stat-val{color:var(--bl)}.c-gr .stat-val{color:var(--gr)}.c-ye .stat-val{color:var(--ye)}.c-re .stat-val{color:var(--re)}.c-pu .stat-val{color:var(--pu)}
.badge{padding:2px 9px;border-radius:99px;font-size:.7rem;font-weight:700;text-transform:uppercase;border:1px solid;white-space:nowrap}
.badge.active{background:var(--grb);color:var(--gr);border-color:var(--grd)}
.badge.candidate{background:var(--yeb);color:var(--ye);border-color:var(--yed)}
.badge.retired{background:var(--bg3);color:var(--mu);border-color:var(--bd)}
.mcard{background:var(--bg3);border:1px solid var(--bd);border-radius:10px;padding:1.1rem;display:flex;flex-direction:column;gap:10px;transition:all .2s}
.mcard:hover{border-color:var(--bdh);transform:translateY(-2px)}
.mcard.is-active{border-color:var(--grd);background:linear-gradient(145deg,rgba(63,185,80,.04),var(--bg3))}
.mcard.is-cand{border-color:var(--yed)}
.mcard-hdr{display:flex;justify-content:space-between;align-items:flex-start}
.mcard-name{font-weight:700;font-size:.95rem}
.mcard-ver{font-size:.73rem;color:var(--mu);margin-top:2px;font-family:'JetBrains Mono',monospace}
.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}
.metric{background:var(--bg2);border:1px solid var(--bd);border-radius:7px;padding:7px 9px}
.metric-lbl{font-size:.68rem;color:var(--mu);text-transform:uppercase}
.metric-val{font-size:.95rem;font-weight:700;font-family:'JetBrains Mono',monospace;margin-top:1px}
.mcard-actions{display:flex;gap:6px;flex-wrap:wrap;padding-top:8px;border-top:1px solid var(--bd)}
.sec-title{font-size:.72rem;font-weight:600;text-transform:uppercase;letter-spacing:1px;color:var(--mu);display:flex;align-items:center;gap:8px;margin-bottom:1rem}
.sec-title::after{content:'';flex:1;height:1px;background:var(--bd)}
.model-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:1rem}
.console{background:#010409;border:1px solid var(--bd);border-radius:8px;padding:1rem;font-family:'JetBrains Mono',monospace;font-size:.78rem;height:200px;overflow-y:auto;color:var(--gr);line-height:1.6}
.console::-webkit-scrollbar{width:5px}.console::-webkit-scrollbar-thumb{background:var(--bg3);border-radius:4px}
.c-err{color:var(--re)}.c-warn{color:var(--ye)}.c-info{color:var(--bl)}.c-ts{color:rgba(255,255,255,.22)}
.acc-row{display:flex;align-items:center;gap:10px;margin-bottom:.6rem}
.acc-lbl{font-size:.8rem;min-width:120px;font-family:'JetBrains Mono',monospace}
.acc-track{flex:1;height:7px;background:var(--bg3);border-radius:99px;overflow:hidden}
.acc-fill{height:100%;border-radius:99px;transition:width .8s cubic-bezier(.4,0,.2,1)}
.acc-num{font-size:.8rem;font-weight:700;min-width:45px;text-align:right;font-family:'JetBrains Mono',monospace}
.hrow{display:flex;justify-content:space-between;align-items:center;padding:.7rem 0;border-bottom:1px solid var(--bd)}
.hrow:last-child{border-bottom:none}
.hval{font-size:.8rem;font-weight:600;padding:3px 10px;border-radius:6px;border:1px solid}
.slider-wrap{display:flex;align-items:center;gap:12px}
.slider{-webkit-appearance:none;appearance:none;width:100%;height:6px;border-radius:3px;background:var(--bg3);outline:none}
.slider::-webkit-slider-thumb{-webkit-appearance:none;width:18px;height:18px;border-radius:50%;background:var(--bl);cursor:pointer;border:2px solid var(--bg2);box-shadow:0 0 8px rgba(88,166,255,.4)}
.severity-labels{display:flex;justify-content:space-between;font-size:.72rem;color:var(--mu);margin-top:4px}
.drift-result{border-radius:10px;padding:1rem;border:1px solid;margin-top:1rem;display:none}
.drift-result.show{display:block;animation:fadeIn .3s ease}
.log-table{width:100%;border-collapse:collapse;font-size:.8rem}
.log-table th{text-align:left;padding:.5rem .75rem;color:var(--mu);font-weight:600;font-size:.72rem;text-transform:uppercase;letter-spacing:.5px;border-bottom:1px solid var(--bd)}
.log-table td{padding:.5rem .75rem;border-bottom:1px solid var(--bd);font-family:'JetBrains Mono',monospace}
.log-table tr:last-child td{border-bottom:none}
.log-table tr:hover td{background:var(--bg3)}
.timeline{display:flex;flex-direction:column;gap:0}
.tl-item{display:flex;gap:1rem;position:relative}
.tl-line{display:flex;flex-direction:column;align-items:center;gap:0}
.tl-dot{width:12px;height:12px;border-radius:50%;border:2px solid;flex-shrink:0;margin-top:4px;transition:transform .2s}
.tl-item:hover .tl-dot{transform:scale(1.3)}
.tl-connector{width:2px;flex:1;background:var(--bd);min-height:24px}
.tl-content{padding-bottom:1.25rem;flex:1}
.tl-title{font-weight:600;font-size:.88rem}
.tl-meta{font-size:.77rem;color:var(--mu);margin-top:3px;font-family:'JetBrains Mono',monospace}
.upload-zone{border:2px dashed var(--bd);border-radius:var(--r);padding:2rem;text-align:center;cursor:pointer;transition:border-color .2s;color:var(--mu)}
.upload-zone:hover{border-color:var(--bl);color:var(--bl)}
input,select,textarea{background:var(--bg3);border:1px solid var(--bd);color:var(--tx);border-radius:8px;padding:.45rem .75rem;font-size:.85rem;outline:none;font-family:'Inter',sans-serif}
input:focus,select:focus{border-color:var(--bl)}
.form-row{display:flex;flex-direction:column;gap:.35rem;margin-bottom:.75rem}
.form-label{font-size:.78rem;font-weight:500;color:var(--mu)}
"""

NAV_ITEMS = [
    ("overview","M3 3h7v7H3zm11 0h7v7h-7zM3 14h7v7H3zm11 0h7v7h-7z","Overview"),
    ("models","M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z","Models"),
    ("drift","M22 12h-4l-3 9L9 3l-3 9H2","Drift Lab"),
    ("audit","M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z","Audit Logs"),
    ("pipeline","M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5","Pipeline"),
]

ICONS = {
    "refresh": "M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.59-9.14l2.12 3.51",
    "docs": "M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z",
    "star": "M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z",
    "heart": "M22 12h-4l-3 9L9 3l-3 9H2",
    "bars": "M18 20V10M12 20V4M6 20v-6",
    "rollback": "M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8 M3 3v5h5",
    "promote": "M12 19V5M5 12l7-7 7 7",
    "zap": "M13 2L3 14h9l-1 8 10-12h-9l1-8z",
    "upload": "M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12",
    "drift_icon": "M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z M12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6z",
}

def svg(d, w=16, stroke="currentColor", sw=2):
    return f'<svg width="{w}" height="{w}" fill="none" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="{d}"/></svg>'

sidebar = '<aside class="sidebar"><div class="brand"><div class="brand-icon">' + svg(NAV_ITEMS[0][1],18,"#fff") + '</div><span>ML Platform</span><span style="background:var(--bg3);border:1px solid var(--bd);border-radius:5px;padding:1px 7px;font-size:.7rem;color:var(--mu);font-family:\'JetBrains Mono\',monospace;margin-left:4px">v1.0</span></div><nav class="nav">'
for key, path, label in NAV_ITEMS:
    actcls = ' active' if key == "overview" else ''
    sidebar += f'<div class="nav-item{actcls}" onclick="showTab(\'{key}\')" id="nav-{key}">{svg(path)}<span>{label}</span></div>'
sidebar += '</nav></aside>'

topbar = '<header class="topbar"><div class="page-title" id="page-title">Overview</div><div class="tbar-r"><div class="status-pill" id="global-status" style="background:var(--grb);color:var(--gr);border-color:var(--grd)"><div class="dot" style="background:var(--gr);box-shadow:0 0 6px var(--gr)"></div>Operational</div><button class="btn" onclick="loadAll()">' + svg(ICONS["refresh"],13) + ' Refresh</button><a href="/docs" target="_blank" class="btn pri">' + svg(ICONS["docs"],13) + ' API Docs</a></div></header>'

# Overview tab
overview = '''<div class="tab-pane active" id="tab-overview">
<div class="sec-title">''' + svg("M3 3h7v7H3zm11 0h7v7h-7zM3 14h7v7H3zm11 0h7v7h-7z",12) + ''' Platform Summary</div>
<div class="grid4" style="margin-bottom:1.5rem">
<div class="card c-bl"><div class="stat-lbl">Total Versions</div><div class="stat-val" id="s-total">—</div><div class="stat-sub">All registered artifacts</div></div>
<div class="card c-gr"><div class="stat-lbl">Active</div><div class="stat-val" id="s-active">—</div><div class="stat-sub">Serving live traffic</div></div>
<div class="card c-ye"><div class="stat-lbl">Candidates</div><div class="stat-val" id="s-cand">—</div><div class="stat-sub">Awaiting promotion</div></div>
<div class="card c-re"><div class="stat-lbl">Retired</div><div class="stat-val" id="s-ret">—</div><div class="stat-sub">Kept for failover</div></div>
<div class="card c-pu"><div class="stat-lbl">Avg Accuracy</div><div class="stat-val" id="s-acc">—</div><div class="stat-sub">Across all versions</div></div>
<div class="card c-bl"><div class="stat-lbl">Model Families</div><div class="stat-val" id="s-names">—</div><div class="stat-sub">Distinct pipelines</div></div>
</div>
<div class="grid2" style="margin-bottom:1.5rem">
<div class="card" style="background:linear-gradient(135deg,var(--grb),var(--blb));border-color:var(--grd)">
<div class="card-hdr"><div class="card-title">''' + svg(ICONS["star"],16,"var(--gr)") + ''' Best Performing Model</div><div class="badge active" id="bp-badge" style="display:none"></div></div>
<div style="font-size:.75rem;text-transform:uppercase;letter-spacing:1px;color:var(--gr);margin-bottom:.25rem">Top Accuracy</div>
<div style="font-size:1.4rem;font-weight:800" id="bp-name">—</div>
<div style="font-size:.82rem;color:var(--mu);margin:.25rem 0" id="bp-ver">Loading…</div>
<div style="font-size:3rem;font-weight:800;color:var(--gr);font-family:'JetBrains Mono',monospace;line-height:1" id="bp-acc">—</div>
<div style="font-size:.8rem;color:var(--mu);margin-top:.5rem">Accuracy on held-out test set</div>
</div>
<div class="card">
<div class="card-hdr"><div class="card-title">''' + svg(ICONS["heart"],16,"var(--bl)") + ''' System Health</div><div class="badge active" id="hbadge">Checking…</div></div>
<div class="hrow"><div><div style="font-size:.88rem;font-weight:500">Registry</div><div style="font-size:.75rem;color:var(--mu)">Total model versions</div></div><div class="hval" id="h-cnt" style="background:var(--blb);color:var(--bl);border-color:var(--bld)">—</div></div>
<div class="hrow"><div><div style="font-size:.88rem;font-weight:500">Artifact Store</div><div style="font-size:.75rem;color:var(--mu)">Joblib binary files</div></div><div class="hval" id="h-mod" style="background:var(--bg3);border-color:var(--bd)">—</div></div>
<div class="hrow"><div><div style="font-size:.88rem;font-weight:500">Metadata Store</div><div style="font-size:.75rem;color:var(--mu)">JSON state files</div></div><div class="hval" id="h-meta" style="background:var(--bg3);border-color:var(--bd)">—</div></div>
<div class="hrow"><div><div style="font-size:.88rem;font-weight:500">Audit Ledger</div><div style="font-size:.75rem;color:var(--mu)">JSONL compliance logs</div></div><div class="hval" id="h-aud" style="background:var(--bg3);border-color:var(--bd)">—</div></div>
</div>
</div>
<div class="card">
<div class="card-hdr"><div class="card-title">''' + svg(ICONS["bars"],16,"var(--pu)") + ''' Version Accuracy Comparison</div><span style="font-size:.75rem;color:var(--mu)">All versions ranked</span></div>
<div id="acc-bars"><div style="color:var(--mu);font-size:.85rem">Loading…</div></div>
</div>
</div>'''

# Models tab
models_tab = '''<div class="tab-pane" id="tab-models">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem">
<div class="sec-title" style="margin-bottom:0">''' + svg(ICONS[list(ICONS.keys())[0]],12) + ''' Model Registry <span id="reg-count" style="background:var(--blb);color:var(--bl);border:1px solid var(--bld);border-radius:4px;padding:1px 8px;font-size:.73rem;margin-left:6px"></span></div>
<button class="btn" onclick="loadModels()">''' + svg(ICONS["refresh"],13) + ''' Refresh</button>
</div>
<div class="model-grid" id="model-grid"><div style="color:var(--mu)">Loading registry…</div></div>
</div>'''

# Drift Lab tab
drift_tab = '''<div class="tab-pane" id="tab-drift">
<div class="sec-title">''' + svg(ICONS["drift_icon"],12) + ''' Drift Detection Laboratory</div>
<div class="grid2">
<div class="card">
<div class="card-hdr"><div class="card-title">''' + svg(ICONS["zap"],16,"var(--ye)") + ''' Manual Drift Simulator</div></div>
<p style="font-size:.83rem;color:var(--mu);line-height:1.6;margin-bottom:1rem">Inject synthetic data drift at configurable severity. Uses KS-test, Chi-Square, Jensen-Shannon divergence and PSI algorithms to generate a realistic drift report.</p>
<div class="form-row"><div class="form-label">Target Model</div><select id="drift-model" style="width:100%"><option value="">Select model…</option></select></div>
<div class="form-row"><div class="form-label">Drift Severity: <span id="sev-val" style="color:var(--bl);font-weight:700">0.50</span></div>
<div class="slider-wrap"><span style="font-size:.72rem;color:var(--mu)">0%</span><input type="range" class="slider" id="sev-slider" min="0" max="100" value="50" oninput="document.getElementById('sev-val').innerText=(this.value/100).toFixed(2);updateSevColor(this.value)"><span style="font-size:.72rem;color:var(--mu)">100%</span></div>
<div class="severity-labels"><span>None</span><span>Low</span><span>Medium</span><span style="color:var(--re)">High</span></div>
</div>
<div style="display:flex;gap:.6rem;margin-top:.5rem">
<button class="btn pri" onclick="simulateDrift()" style="flex:1">''' + svg(ICONS["zap"],13) + ''' Simulate Drift</button>
<button class="btn yel" onclick="simulateAuto()" style="flex:1">⚡ Auto Escalation</button>
</div>
<div class="drift-result" id="drift-result"></div>
</div>
<div class="card">
<div class="card-hdr"><div class="card-title">📋 Drift Event History</div><button class="btn" onclick="loadDriftHistory()" style="font-size:.77rem;padding:3px 9px">Refresh</button></div>
<div id="drift-history" style="max-height:400px;overflow-y:auto"><div style="color:var(--mu);font-size:.85rem">No drift events yet.</div></div>
</div>
</div>
</div>'''

# Audit tab
audit_tab = '''<div class="tab-pane" id="tab-audit">
<div class="sec-title">📋 Prediction Audit Ledger</div>
<div class="card" style="margin-bottom:1rem">
<div class="card-hdr"><div class="card-title">Recent Predictions</div>
<div style="display:flex;gap:.6rem"><select id="audit-model" style="font-size:.82rem;padding:4px 8px"><option value="">All models</option></select><button class="btn" onclick="loadAudit()">''' + svg(ICONS["refresh"],13) + ''' Refresh</button></div>
</div>
<div style="overflow-x:auto"><table class="log-table"><thead><tr><th>Timestamp</th><th>Model</th><th>Version</th><th>Prediction</th><th>Confidence</th></tr></thead><tbody id="audit-body"><tr><td colspan="5" style="text-align:center;color:var(--mu)">Loading…</td></tr></tbody></table></div>
<div id="audit-meta" style="font-size:.77rem;color:var(--mu);margin-top:.75rem;text-align:right"></div>
</div>
<div class="card">
<div class="card-hdr"><div class="card-title">Make Test Prediction</div></div>
<p style="font-size:.83rem;color:var(--mu);margin-bottom:1rem">Send a prediction request to the active model and watch the audit log update.</p>
<div style="display:flex;gap:.75rem;align-items:center;flex-wrap:wrap">
<select id="pred-model" style="flex:1;min-width:160px"><option value="">Select model…</option></select>
<input id="pred-features" placeholder="e.g. 35,12,85,340,2" style="flex:2;min-width:200px">
<button class="btn suc" onclick="makePrediction()">''' + svg(ICONS["zap"],13) + ''' Predict</button>
</div>
<div id="pred-result" style="margin-top:.75rem;font-size:.83rem;font-family:'JetBrains Mono',monospace;color:var(--gr);display:none"></div>
</div>
</div>'''

# Pipeline tab
pipeline_tab = '''<div class="tab-pane" id="tab-pipeline">
<div class="sec-title">''' + svg(NAV_ITEMS[4][1],12) + ''' Automated Lifecycle Pipeline</div>
<div class="grid2" style="margin-bottom:1.5rem">
<div class="card">
<div class="card-hdr"><div class="card-title">''' + svg(ICONS["zap"],16,"var(--pu)") + ''' Trigger Retraining</div></div>
<p style="font-size:.83rem;color:var(--mu);line-height:1.6;margin-bottom:1rem">Simulate an automated pipeline that loads the current active model, trains an improved variant with additional estimators, and registers it as a new candidate version.</p>
<div class="form-row"><div class="form-label">Target Model</div><select id="retrain-model" style="width:100%"><option value="">Select model…</option></select></div>
<div style="display:flex;gap:.6rem">
<button class="btn pri" onclick="triggerRetrain()" style="flex:1">''' + svg(ICONS["zap"],13) + ''' Retrain Model</button>
<button class="btn" onclick="document.getElementById('retrain-console').innerHTML=''">Clear</button>
</div>
<div class="console" id="retrain-console" style="margin-top:.75rem">> Pipeline supervisor idle. Ready to retrain.</div>
</div>
<div class="card">
<div class="card-hdr"><div class="card-title">''' + svg(ICONS["rollback"],16,"var(--re)") + ''' Quick Rollback Test</div></div>
<p style="font-size:.83rem;color:var(--mu);line-height:1.6;margin-bottom:1rem">Simulate a 40% accuracy drop and trigger the autonomous rollback daemon. Watches for performance threshold breach and reroutes traffic with zero downtime.</p>
<div style="display:flex;gap:.6rem;margin-bottom:.75rem"><select id="rb-model" style="flex:1"><option value="">Select model…</option></select><button class="btn dan" onclick="triggerRollback()">⚠ Simulate Drop</button></div>
<div class="console" id="console">> Supervisor daemon watching pipeline…</div>
</div>
</div>
<div class="card">
<div class="card-hdr"><div class="card-title">📅 Version Timeline</div><select id="timeline-model" onchange="loadTimeline()" style="font-size:.82rem"><option value="">All models</option></select></div>
<div id="timeline"><div style="color:var(--mu);font-size:.85rem">Loading timeline…</div></div>
</div>
</div>'''

JS = """
const $ = id => document.getElementById(id);
const fmt = v => v != null ? v : '—';
const fmtPct = v => v != null ? (v*100).toFixed(1)+'%' : 'N/A';
const fmtDate = iso => {if(!iso)return '—';try{return new Date(iso).toLocaleString('en-US',{month:'short',day:'numeric',year:'numeric',hour:'2-digit',minute:'2-digit'});}catch{return iso;}};
function clog(id,msg,cls=''){const c=$(id),t=new Date().toLocaleTimeString('en-US',{hour12:false});c.innerHTML+=`<br><span class="c-ts">[${t}]</span> <span class="${cls}">${msg}</span>`;c.scrollTop=c.scrollHeight;}

function showTab(key){
  document.querySelectorAll('.tab-pane').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n=>n.classList.remove('active'));
  $('tab-'+key).classList.add('active');
  $('nav-'+key).classList.add('active');
  const titles={overview:'Overview',models:'Model Registry',drift:'Drift Detection Laboratory',audit:'Audit Logs',pipeline:'Lifecycle Pipeline'};
  $('page-title').innerText=titles[key]||key;
  if(key==='drift')loadDriftHistory();
  if(key==='audit')loadAudit();
  if(key==='pipeline')loadTimeline();
}

async function loadHealth(){
  try{
    const d=await fetch('/health').then(r=>r.json());
    const ok=d.status==='healthy';
    const gs=$('global-status');
    gs.style.background=ok?'var(--grb)':'var(--reb)';gs.style.color=ok?'var(--gr)':'var(--re)';gs.style.borderColor=ok?'var(--grd)':'var(--red)';
    gs.innerHTML=`<div class="dot" style="background:${ok?'var(--gr)':'var(--re)'};box-shadow:0 0 6px ${ok?'var(--gr)':'var(--re)'}"></div>${ok?'Operational':'Degraded'}`;
    const hb=$('hbadge');hb.className='badge '+(ok?'active':'retired');hb.innerText=ok?'Healthy':'Degraded';
    $('h-cnt').innerText=d.checks.model_count??'—';
    const s=d.checks.storage||{};
    [['h-mod',s.models_exists],['h-meta',s.metadata_exists],['h-aud',s.audit_exists]].forEach(([id,on])=>{
      const el=$(id);el.innerText=on?'Mounted':'Offline';
      el.style.background=on?'var(--grb)':'var(--reb)';el.style.color=on?'var(--gr)':'var(--re)';el.style.borderColor=on?'var(--grd)':'var(--red)';
    });
  }catch(e){console.error(e);}
}

async function loadStats(){
  try{
    const s=await fetch('/stats').then(r=>r.json());
    $('s-total').innerText=fmt(s.total_models);$('s-active').innerText=fmt(s.active_count);
    $('s-cand').innerText=fmt(s.candidate_count);$('s-ret').innerText=fmt(s.retired_count);
    $('s-acc').innerText=s.avg_accuracy!=null?fmtPct(s.avg_accuracy):'—';
    $('s-names').innerText=fmt(s.model_names?.length);
    if(s.best_performer){const bp=s.best_performer;$('bp-name').innerText=bp.model_name;$('bp-ver').innerText=`Version ${bp.version} · ${bp.status.toUpperCase()}`;$('bp-acc').innerText=fmtPct(bp.accuracy);const b=$('bp-badge');b.style.display='inline-flex';b.className='badge '+bp.status;b.innerText=bp.status.toUpperCase();}
    const bars=$('acc-bars');const all=[];
    Object.entries(s.models_by_name||{}).forEach(([n,d])=>(d.versions||[]).forEach(v=>all.push({label:`${n} ${v.version}`,acc:v.accuracy,status:v.status})));
    all.sort((a,b)=>(b.acc||0)-(a.acc||0));
    const colors={active:'var(--gr)',candidate:'var(--ye)',retired:'var(--bl)'};
    bars.innerHTML=all.length?all.map(v=>{const pct=v.acc!=null?(v.acc*100).toFixed(1):0;const c=colors[v.status]||'var(--mu)';return`<div class="acc-row"><div class="acc-lbl">${v.label}</div><div class="acc-track"><div class="acc-fill" style="width:${pct}%;background:${c}"></div></div><div class="acc-num" style="color:${c}">${v.acc!=null?pct+'%':'N/A'}</div></div>`;}).join(''):'<div style="color:var(--mu);font-size:.85rem">No accuracy data yet.</div>';
    // Populate selects
    const names=s.model_names||[];
    ['drift-model','retrain-model','rb-model','timeline-model','pred-model','audit-model'].forEach(id=>{
      const el=$(id);if(!el)return;
      const prev=el.value;
      el.innerHTML=(id==='timeline-model'?'<option value="">All models</option>':'<option value="">Select model…</option>')+names.map(n=>`<option value="${n}">${n}</option>`).join('');
      if(prev)el.value=prev;
    });
  }catch(e){console.error(e);}
}

async function loadModels(){
  try{
    const data=await fetch('/models').then(r=>r.json());
    const grid=$('model-grid');const cnt=$('reg-count');
    if(!data||!data.length){grid.innerHTML='<div style="grid-column:1/-1;text-align:center;padding:3rem;color:var(--mu);background:var(--bg2);border:1px dashed var(--bd);border-radius:var(--r)">No models registered.<br>Run <code style="color:var(--bl)">python scripts/generate_demo_data.py</code></div>';if(cnt)cnt.innerText='0 versions';return;}
    if(cnt)cnt.innerText=`${data.length} versions`;
    data.sort((a,b)=>{const w={active:3,candidate:2,retired:1};const d=(w[b.status]||0)-(w[a.status]||0);return d||b.version.localeCompare(a.version,undefined,{numeric:true});});
    const desc={active:'Routing 100% of live production traffic.',candidate:'In shadow validation. Awaiting promotion.',retired:'Preserved on disk as instant failback.'};
    grid.innerHTML=data.map(m=>{
      const acc=m.performance_metrics?.accuracy,f1=m.performance_metrics?.f1,sc=m.status==='active'?'is-active':m.status==='candidate'?'is-cand':'';
      const aColor=m.status==='active'?'var(--gr)':'var(--tx)';
      const actions=[];
      if(m.status!=='active')actions.push(`<button class="btn suc" onclick="promoteModel('${m.model_name}','${m.version}')" style="font-size:.77rem;padding:3px 8px">${svg(ICONS.promote,12)} Promote</button>`);
      if(m.status!=='active')actions.push(`<button class="btn dan" onclick="rollbackTo('${m.model_name}','${m.version}')" style="font-size:.77rem;padding:3px 8px">${svg(ICONS.rollback,12)} Set Active</button>`);
      return`<div class="mcard ${sc}"><div class="mcard-hdr"><div><div class="mcard-name">${m.model_name}</div><div class="mcard-ver">${m.version} · ${(m.model_type||'').toUpperCase()}</div></div><div class="badge ${m.status}">${m.status.toUpperCase()}</div></div><div class="metrics"><div class="metric"><div class="metric-lbl">Accuracy</div><div class="metric-val" style="color:${aColor}">${fmtPct(acc)}</div></div><div class="metric"><div class="metric-lbl">F1 Score</div><div class="metric-val">${fmtPct(f1)}</div></div><div class="metric"><div class="metric-lbl">Features</div><div class="metric-val">${m.features?.length??'?'}</div></div></div><p style="font-size:.8rem;color:var(--mu)">${desc[m.status]||''}</p><div class="mcard-actions">${actions.join('')}</div><div style="display:flex;justify-content:space-between;border-top:1px solid var(--bd);padding-top:8px;margin-top:2px"><span style="font-family:'JetBrains Mono',monospace;font-size:.7rem;color:var(--mu)">ID: ${m.model_id?.substring(0,20)}…</span><span style="font-size:.75rem;color:var(--mu)">${fmtDate(m.training_date)}</span></div></div>`;
    }).join('');
  }catch(e){console.error(e);}
}
// stub so template can reference ICONS in JS
const ICONS={promote:"M12 19V5M5 12l7-7 7 7",rollback:"M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8 M3 3v5h5"};
function svg(d,w=16){return `<svg width="${w}" height="${w}" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="${d}"/></svg>`;}

async function promoteModel(name,ver){
  try{const r=await fetch(`/models/${name}/promote`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({version:ver})});const j=await r.json();alert(j.message||'Promoted!');loadAll();}catch(e){alert('Error: '+e);}
}
async function rollbackTo(name,ver){
  try{const r=await fetch(`/models/${name}/rollback`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({target_version:ver})});const j=await r.json();alert(j.message||'Done!');loadAll();}catch(e){alert('Error: '+e);}
}

function updateSevColor(v){const sl=$('sev-slider');const c=v<15?'var(--gr)':v<35?'var(--ye)':v<60?'var(--ye)':'var(--re)';sl.style.accentColor=c;$('sev-val').style.color=c;}

async function simulateDrift(){
  const model=$('drift-model').value;if(!model){alert('Select a model');return;}
  const sev=parseFloat($('sev-slider').value)/100;
  try{
    const r=await fetch('/drift/simulate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model_name:model,severity:sev,label:'manual'})});
    const j=await r.json();
    const detected=j.drift_detected;const score=(j.drift_score*100).toFixed(1);
    const slab=j.severity_label;
    const color=detected?(sev>=0.6?'var(--red)':'var(--yed)'):'var(--grd)';
    const bg=detected?(sev>=0.6?'var(--reb)':'var(--yeb)'):'var(--grb)';
    const res=$('drift-result');
    const feats=Object.entries(j.feature_drifts||{}).map(([f,s])=>`<span style="display:inline-block;margin:.25rem .3rem .25rem 0;background:var(--bg3);border:1px solid var(--bd);border-radius:6px;padding:2px 8px;font-size:.77rem;font-family:'JetBrains Mono',monospace">${f}: ${(s*100).toFixed(1)}%</span>`).join('');
    res.style.background=bg;res.style.borderColor=color;
    res.innerHTML=`<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.6rem"><strong style="color:${color}">${detected?'⚠ DRIFT DETECTED':'✅ No Drift'}</strong><span style="font-family:'JetBrains Mono',monospace;font-size:.9rem;font-weight:700;color:${color}">${score}%</span></div><div style="font-size:.78rem;color:var(--mu);margin-bottom:.5rem">Severity: <strong style="color:${color}">${slab.toUpperCase()}</strong> · Recommendation: <strong>${j.recommendation}</strong></div><div>${feats}</div>`;
    res.classList.add('show');loadDriftHistory();
  }catch(e){alert('API error: '+e);}
}

async function simulateAuto(){
  const model=$('drift-model').value;if(!model){alert('Select a model');return;}
  try{await fetch(`/drift/simulate-auto?model_name=${model}`,{method:'POST'});loadDriftHistory();}catch(e){alert('Error: '+e);}
}

async function loadDriftHistory(){
  try{
    const d=await fetch('/drift/history?limit=20').then(r=>r.json());
    const el=$('drift-history');
    if(!d.events||!d.events.length){el.innerHTML='<div style="color:var(--mu);font-size:.85rem">No drift events yet. Use the simulator above.</div>';return;}
    el.innerHTML=d.events.map(e=>{
      const det=e.drift_detected;const c=det?'var(--ye)':'var(--gr)';
      return`<div style="border-bottom:1px solid var(--bd);padding:.6rem 0"><div style="display:flex;justify-content:space-between"><span style="font-weight:600;color:${c}">${det?'⚠ Drift':'✅ Clear'} <span class="badge" style="background:var(--bg3);border-color:var(--bd);color:var(--mu)">${e.severity_label||''}</span></span><span style="font-family:'JetBrains Mono',monospace;font-size:.72rem;color:var(--mu)">${fmtDate(e.timestamp)}</span></div><div style="font-size:.77rem;color:var(--mu);margin-top:2px">${e.model_name} · Score: ${((e.drift_score||0)*100).toFixed(1)}% · ${e.recommendation||''}</div></div>`;
    }).join('');
  }catch(e){}
}

async function loadAudit(){
  try{
    const model=$('audit-model')?.value||'';
    const url=model?`/audit/logs?model_name=${model}&limit=50`:'/audit/logs?limit=50';
    const d=await fetch(url).then(r=>r.json());
    const tb=$('audit-body');
    if(!d.entries||!d.entries.length){tb.innerHTML='<tr><td colspan="5" style="text-align:center;color:var(--mu);padding:2rem">No predictions logged yet. Make a prediction to see audit entries.</td></tr>';return;}
    tb.innerHTML=d.entries.slice(0,50).map(e=>`<tr><td>${fmtDate(e.timestamp)}</td><td>${e.model_name||'—'}</td><td>${e.model_version||'—'}</td><td style="color:var(--gr)">${e.prediction??'—'}</td><td>${e.confidence!=null?(e.confidence*100).toFixed(1)+'%':'—'}</td></tr>`).join('');
    $('audit-meta').innerText=`Showing ${d.entries.length} of ${d.count} entries`;
  }catch(e){}
}

async function makePrediction(){
  const model=$('pred-model').value;if(!model){alert('Select a model');return;}
  const raw=$('pred-features').value||'35,12,85,340,2';
  const vals=raw.split(',').map(Number).filter(v=>!isNaN(v));
  const featNames=['age','tenure_months','monthly_charge','total_charge','support_calls'];
  const features={};vals.forEach((v,i)=>features[featNames[i]||`f${i}`]=v);
  try{
    const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model_name:model,features})});
    const j=await r.json();
    const el=$('pred-result');
    el.style.display='block';
    el.innerHTML=`✅ Prediction: <strong style="color:var(--gr)">${j.prediction}</strong>  Confidence: <strong>${fmtPct(j.confidence)}</strong>  Version: ${j.model_version}`;
    setTimeout(loadAudit,500);
  }catch(e){$('pred-result').style.display='block';$('pred-result').innerText='Error: '+e;$('pred-result').style.color='var(--re)';}
}

async function loadTimeline(){
  try{
    const model=$('timeline-model')?.value||'';
    const url=model?`/models/${model}/versions`:'/models';
    const data=await fetch(url).then(r=>r.json());
    const el=$('timeline');if(!data||!data.length){el.innerHTML='<div style="color:var(--mu)">No versions found.</div>';return;}
    data.sort((a,b)=>a.version.localeCompare(b.version,undefined,{numeric:true}));
    const colors={active:'var(--gr)',candidate:'var(--ye)',retired:'var(--mu)'};
    const bdcolors={active:'var(--grd)',candidate:'var(--yed)',retired:'var(--bd)'};
    el.innerHTML='<div class="timeline">'+data.map((m,i)=>{const c=colors[m.status];const bc=bdcolors[m.status];return`<div class="tl-item"><div class="tl-line"><div class="tl-dot" style="background:${c};border-color:${bc};box-shadow:0 0 6px ${c}22"></div>${i<data.length-1?'<div class="tl-connector"></div>':''}</div><div class="tl-content"><div class="tl-title">${m.model_name} <span class="badge ${m.status}">${m.version} · ${m.status.toUpperCase()}</span></div><div class="tl-meta">${fmtDate(m.training_date)} · Accuracy: ${fmtPct(m.performance_metrics?.accuracy)}</div></div></div>`;}).join('')+'</div>';
  }catch(e){}
}

async function triggerRetrain(){
  const model=$('retrain-model').value;if(!model){clog('retrain-console','Select a model first.','c-warn');return;}
  clog('retrain-console',`[PIPELINE] Initialising retrain for [${model}]…`,'c-info');
  setTimeout(()=>{clog('retrain-console','[DATA] Generating synthetic training batch (3000 samples)…');},600);
  setTimeout(()=>{clog('retrain-console','[TRAIN] Fitting RandomForest with +10 estimators…');},1400);
  setTimeout(async()=>{
    try{
      const r=await fetch('/pipeline/retrain',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model_name:model})});
      const j=await r.json();
      if(j.status==='ok'){clog('retrain-console',`[DONE] ✅ ${j.new_version} registered. Accuracy: ${fmtPct(j.new_accuracy)} (was ${fmtPct(j.base_accuracy)})`,'');clog('retrain-console',`[NEXT] Promote via: POST /models/${model}/promote`,'c-info');loadAll();}
      else clog('retrain-console','[ERROR] '+JSON.stringify(j),'c-err');
    }catch(e){clog('retrain-console','[ERROR] API unreachable.','c-err');}
  },2400);
}

async function triggerRollback(){
  const modelName=$('rb-model').value;if(!modelName){clog('console','Select a model first.','c-warn');return;}
  clog('console',`Probing live metrics on [${modelName}]…`,'c-info');
  setTimeout(()=>{clog('console','DRIFT ALERT ⚠  Jensen-Shannon divergence > threshold','c-warn');},900);
  setTimeout(async()=>{
    clog('console','CRITICAL: accuracy 40% << 80% threshold. Engaging failover…','c-err');
    try{
      const r=await fetch(`/models/${modelName}/evaluate-performance`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({current_metric:0.40,performance_threshold:0.80,metric_name:'accuracy'})});
      const j=await r.json();
      if(j.status==='rolled_back'){clog('console',`✅ Failover complete — traffic rerouted to ${j.new_active_version}`,'');clog('console',`Restored accuracy: ${fmtPct(j.new_active_accuracy)} · Downtime: ${j.downtime_sec??0}s`,'c-info');setTimeout(()=>loadAll(),500);}
      else clog('console','Rollback not triggered: '+j.message,'c-warn');
    }catch(e){clog('console','API unreachable.','c-err');}
  },1800);
}

async function loadAll(){await Promise.all([loadHealth(),loadStats(),loadModels()]);}
loadAll();
setInterval(loadHealth,15000);
"""

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ML Reliability Platform — Operations Center</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>{CSS}</style>
</head>
<body>
<div class="layout">
{sidebar}
<div class="main">
{topbar}
<div class="content">
{overview}
{models_tab}
{drift_tab}
{audit_tab}
{pipeline_tab}
</div>
</div>
</div>
<script>{JS}</script>
</body>
</html>"""

OUT.write_text(HTML, encoding="utf-8")
print(f"Written {len(HTML):,} bytes to {OUT}")
