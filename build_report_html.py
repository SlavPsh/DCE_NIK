import base64, os
SC = "/home/rnga/vvpshenov/tmp/claude-8186/-scratch-rnga-vvpshenov/df07c4f6-69fb-4088-a73f-eea680358422/scratchpad"
def uri(p): return "data:image/png;base64,"+base64.b64encode(open(p, "rb").read()).decode() if os.path.exists(p) else ""
REAL = uri("results/realdata_nik_vs_cs_figures/figures/outcoil_report.png")
PHAN = uri("results/xcat_physical_nomotion_nik_vs_grasp/figures/outcoil_compare.png")
html = """<title>Where the Coil Goes</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{--paper:#f5f7f8;--surface:#fff;--ink:#161b22;--muted:#5a6572;--faint:#8a94a2;--line:#e2e6ec;--lineb:#cdd4dd;
--accent:#0b6e78;--asoft:#0b6e7817;--win:#2f7d54;--wsoft:#2f7d5417;--warn:#9a6212;--warnsoft:#9a621217;--shadow:0 1px 2px rgba(20,30,40,.04),0 8px 26px rgba(20,30,40,.05);}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--paper:#0e1218;--surface:#161c24;--ink:#e8ecf1;--muted:#9aa5b2;--faint:#6b7684;--line:#242c36;--lineb:#333d49;--accent:#4fb8c2;--asoft:#4fb8c220;--win:#63c68d;--wsoft:#63c68d1e;--warn:#d9a457;--warnsoft:#d9a45720;--shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35);}}
:root[data-theme="dark"]{--paper:#0e1218;--surface:#161c24;--ink:#e8ecf1;--muted:#9aa5b2;--faint:#6b7684;--line:#242c36;--lineb:#333d49;--accent:#4fb8c2;--asoft:#4fb8c220;--win:#63c68d;--wsoft:#63c68d1e;--warn:#d9a457;--warnsoft:#d9a45720;--shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35);}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:16px;line-height:1.62;-webkit-font-smoothing:antialiased}
.wrap{max-width:900px;margin:0 auto;padding:clamp(28px,5vw,60px) clamp(18px,4vw,40px) 90px}
.eyebrow{font-family:"IBM Plex Mono",monospace;font-size:12.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);margin:0 0 16px}
h1{font-family:"Fraunces",Georgia,serif;font-weight:500;font-size:clamp(32px,6vw,52px);line-height:1.05;letter-spacing:-.015em;text-wrap:balance;margin:0}
h1 em{font-style:italic;color:var(--accent)}
.dek{font-size:clamp(17px,2.4vw,20px);color:var(--muted);max-width:62ch;text-wrap:balance;margin:12px 0 0}
.meta{font-family:"IBM Plex Mono",monospace;font-size:12.5px;color:var(--faint);margin-top:20px}
.chips{display:flex;flex-wrap:wrap;gap:12px;margin:30px 0 0}
.chip{flex:1 1 180px;background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:15px 16px;box-shadow:var(--shadow)}
.chip .k{font-family:"IBM Plex Mono",monospace;font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--faint)}
.chip .v{font-family:"Fraunces",serif;font-weight:600;font-size:24px;line-height:1.1;margin-top:6px;font-variant-numeric:tabular-nums}
.chip .s{font-size:13px;color:var(--muted);margin-top:3px}.chip.win .v{color:var(--win)}
hr.r{height:1px;background:var(--line);border:0;margin:34px 0}
h2{font-family:"Fraunces",serif;font-weight:500;font-size:26px;letter-spacing:-.01em;margin:50px 0 4px;text-wrap:balance}
.sub{color:var(--muted);font-size:14.5px;margin:0 0 18px}
p{margin:0 0 15px}p.lead{max-width:66ch}strong{font-weight:600}.ac{color:var(--accent);font-weight:600}.wc{color:var(--win);font-weight:600}
figure{margin:8px 0 0;background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:14px;box-shadow:var(--shadow)}
figure img{width:100%;display:block;border-radius:8px}
figcaption{font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--faint);margin-top:12px;line-height:1.5}
.tw{overflow-x:auto;border:1px solid var(--line);border-radius:12px;box-shadow:var(--shadow);background:var(--surface);margin-top:6px}
table{border-collapse:collapse;width:100%;font-size:14.5px;min-width:460px}
th,td{padding:11px 16px;text-align:right;font-variant-numeric:tabular-nums}th:first-child,td:first-child{text-align:left}
thead th{font-family:"IBM Plex Mono",monospace;font-weight:500;font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--faint);border-bottom:1px solid var(--lineb)}
tbody tr+tr td{border-top:1px solid var(--line)}td.m{text-align:left;font-weight:500}.best{color:var(--win);font-weight:600}.dim{color:var(--muted)}
.panel{background:var(--surface);border:1px solid var(--line);border-left:3px solid var(--warn);border-radius:12px;padding:18px 22px;margin-top:8px;box-shadow:var(--shadow)}
.panel h3{margin:0 0 10px;font-size:14px;font-family:"IBM Plex Mono",monospace;letter-spacing:.06em;text-transform:uppercase;color:var(--warn)}
.panel ul{margin:0;padding-left:20px}.panel li{margin:0 0 8px;color:var(--muted);font-size:14.5px}.panel li:last-child{margin:0}.panel li strong{color:var(--ink)}
footer{margin-top:56px;padding-top:20px;border-top:1px solid var(--line);color:var(--faint);font-size:12.5px;font-family:"IBM Plex Mono",monospace}
</style>
<div class="wrap">
<p class="eyebrow">DCE-MRI · NIK architecture</p>
<h1>Where the <em>coil</em> goes</h1>
<p class="dek">Moving the coil from a network input (a learned embedding) to shared-backbone output heads. A clean win on the phantom that does not replicate in vivo, and cheaper either way.</p>
<p class="meta">NIK subspace / F0 / F2 · coil-as-input vs coil-as-output · XCAT phantom (truth) + real slice 21 (no truth) · GRASP reference</p>
<div class="chips">
<div class="chip win"><div class="k">Phantom (truth)</div><div class="v">wins</div><div class="s">output-coil beats input on every metric</div></div>
<div class="chip"><div class="k">Real (in vivo)</div><div class="v">no win</div><div class="s">held-out tied; output under-reads vs model-free ref</div></div>
<div class="chip"><div class="k">Compute</div><div class="v">~8&times;</div><div class="s">output-coil cheaper to train</div></div>
</div>
<hr class="r">
<p class="lead">The usual NIK feeds the coil index into the network as a learned <strong>embedding</strong>, so each coil is its own embedding-conditioned function &mdash; the coils fragment the object and don't factor cleanly. The alternative: drop the coil from the input and give the network <strong>C output heads</strong>, one per coil, all sharing a single k-space backbone. Coils can then differ only at the readout, which forces a common object representation. Everything else &mdash; Fourier features, Gabor backbone, temporal basis, normalization, render, scoring &mdash; is held identical, so this isolates the coil placement.</p>

<h2>Phantom, with ground truth</h2>
<p class="sub">XCAT no-motion, body-masked vs truth, val-selected checkpoints (fair vs sub16's own early stopping).</p>
<div class="tw"><table>
<thead><tr><th>Metric</th><th>input-coil (sub16)</th><th>output-coil</th><th>GRASP</th></tr></thead>
<tbody>
<tr><td class="m">PSNR &uarr;</td><td>36.12</td><td class="best">37.21</td><td class="dim">38.53</td></tr>
<tr><td class="m">SSIM &uarr;</td><td>0.922</td><td class="best">0.928</td><td class="dim">0.952</td></tr>
<tr><td class="m">HaarPSI &uarr;</td><td>0.886</td><td class="best">0.901</td><td class="dim">0.871</td></tr>
<tr><td class="m">cortex curve &darr;</td><td>0.055</td><td class="best">0.020</td><td class="dim">&ndash;</td></tr>
</tbody></table></div>
<p class="lead" style="margin-top:16px"><span class="wc">Output-coil beats input-coil on every metric</span>, most strongly on the tissue curves (~2.5&times; lower cortex error), and its HaarPSI clears GRASP. The image is visibly cleaner &mdash; less grain, a dimmer error map at the kidneys.</p>
<figure><img alt="phantom: truth, sub16 input-coil, output-coil, difference maps and ROI curves" src="__PHAN__"><figcaption>phantom double-check &mdash; truth | sub16 (input-coil) | output-coil, with difference maps and ROI curves. output-coil's error map is dimmer, its curves track truth more closely.</figcaption></figure>

<h2>Real data, in vivo</h2>
<p class="sub">Slice 21, no ground truth &mdash; the only accuracy proxy is held-out spoke NMSE. Matched setup: identical spoke split, val early stopping, for input and output across all three models.</p>
<div class="tw"><table>
<thead><tr><th>Held-out NMSE &darr;</th><th>input-coil</th><th>output-coil</th></tr></thead>
<tbody>
<tr><td class="m">subspace (16 atoms)</td><td class="best">0.294</td><td>0.352</td></tr>
<tr><td class="m">F2 (Patlak + 2 free)</td><td class="best">0.508</td><td>0.536</td></tr>
<tr><td class="m">F0 (Patlak, 3 atoms)</td><td>0.707</td><td class="best">0.684</td></tr>
</tbody></table></div>
<p class="lead" style="margin-top:16px">On held-out k-space the two are <strong>at parity</strong> &mdash; input edges subspace/F2, output edges F0, no consistent winner. The contrast curves are the more telling test, and the right reference here is the <strong>model-free NUFFT</strong> recon (thick grey below) &mdash; the density-compensated gridding of all spokes, noisy but unbiased, the closest thing to truth in vivo.</p>
<p class="lead">Against that reference the phantom story <em>does not carry over</em>: <span class="ac">input-coil tracks the enhancement amplitude</span> (cortex plateau ~3.0, aorta peak ~11, both matching the model-free reference), while <strong>output-coil is smoother but under-reads</strong> &mdash; cortex ~2.5, aorta peak ~7 &mdash; a low bias it shares with GRASP. So output-coil trades noise for an amplitude under-read; in vivo, input-coil arguably tracks the dynamics better. (The model-free reference is itself noisy, so read the systematic offset, not the jitter.)</p>
<figure><img alt="real slice 21: input vs output coil across subspace, F0, F2, plus GRASP, with contrast curves and a model-free NUFFT reference" src="__REAL__"><figcaption>real slice 21 &mdash; rows: subspace / F0 / F2; columns: input-coil | output-coil | GRASP | input&minus;output difference. bottom: aorta / cortex / medulla curves &mdash; thick grey = model-free NUFFT reference (unbiased), dashed = input-coil, solid = output-coil, dotted = GRASP. output-coil and GRASP sit below the model-free plateau; input-coil tracks it.</figcaption></figure>

<h2>Verdict</h2>
<div class="panel">
<h3>What holds, and what to be careful about</h3>
<ul>
<li><strong>Phantom: a real, verified accuracy win.</strong> Output-coil is the best NIK config we have &mdash; beats input-coil on every metric and clears GRASP on HaarPSI.</li>
<li><strong>Real: the win does not replicate.</strong> Held-out NMSE is at parity, and against the model-free NUFFT reference input-coil tracks the enhancement amplitude better while output-coil under-reads the plateau/peak (smoother, but low-biased like GRASP). The phantom advantage &mdash; which needed ground truth to see &mdash; doesn't appear in vivo.</li>
<li><strong>Efficiency is unambiguous.</strong> One shared backbone pass produces all coils, versus C passes for the input embedding &mdash; about 8&times; cheaper to train, both in vivo and on the phantom.</li>
<li><strong>Across models:</strong> the effect is clearest for the flexible subspace model; F0 (3 fixed Patlak atoms) is nearly identical input-vs-output, since there's little left for the coil placement to change.</li>
</ul>
</div>
<footer>NIK coil-placement study &middot; subspace/F0/F2 &middot; XCAT phantom + real slice 21 &middot; GRASP reference (cs f100) &middot; ground truth = evaluation only</footer>
</div>"""
html = html.replace("__PHAN__", PHAN).replace("__REAL__", REAL)
open(f"{SC}/coil_report.html", "w").write(html); print("wrote", f"{SC}/coil_report.html", round(len(html)/1e6, 2), "MB")
