"""Live KC-raster dashboard: sniff any text through the male-CNS mushroom body
and watch the Kenyon-cell tag light up, with the nearest retro, matched pattern
family, distilled guidance, and novelty.

Stdlib only. Build the pipeline once at startup; distil family guidance once
(honouring the optional LLM). Run:  python dashboard.py [app.db] [port]
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from flyretro.distill import distill_family
from flyretro.flyhash import overlap
from flyretro.llm import llm_configured
from flyretro.nwf_ingest import DEFAULT_DB
from flyretro.pipeline import build
from flyretro.plan_advice import advise_plan

DB = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DB
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8765

print(f"building pipeline from {DB} ...")
PIPE = build(DB)
GUIDANCE: dict[str, tuple[str, str]] = {f.label: distill_family(f) for f in PIPE.families}
print(f"ready: {len(PIPE.records)} retros, {len(PIPE.families)} families, "
      f"LLM={'on' if llm_configured() else 'off (fallback)'}")

PAGE = """<!doctype html><html><head><meta charset=utf-8>
<title>fly-retro-nose</title><style>
 body{font:14px/1.5 system-ui,sans-serif;margin:0;background:#0b0e14;color:#d6deeb}
 header{padding:14px 20px;background:#11161f;border-bottom:1px solid #232b3a}
 header b{color:#8fd3ff} .muted{color:#7a8699}
 main{display:grid;grid-template-columns:420px 1fr;gap:20px;padding:20px}
 textarea{width:100%;height:120px;background:#0e131c;color:#d6deeb;border:1px solid #263041;border-radius:8px;padding:10px;box-sizing:border-box}
 button{margin-top:8px;background:#1c6feb;color:#fff;border:0;border-radius:8px;padding:9px 16px;cursor:pointer;font-weight:600}
 canvas{background:#070a10;border:1px solid #232b3a;border-radius:8px;image-rendering:pixelated}
 .card{background:#11161f;border:1px solid #232b3a;border-radius:8px;padding:12px 14px;margin-bottom:12px}
 .k{color:#7a8699} .v{color:#e8eefc} .big{font-size:22px;font-weight:700}
 .bar{height:10px;background:#1a2130;border-radius:6px;overflow:hidden;margin-top:4px}
 .bar>span{display:block;height:100%;background:linear-gradient(90deg,#2ee6a6,#8fd3ff)}
 .tag{display:inline-block;background:#1a2536;color:#8fd3ff;border-radius:5px;padding:1px 7px;margin:2px 3px 0 0;font-size:12px}
 code{color:#ffcf8f}
</style></head><body>
<header><b>fly-retro-nose</b> &nbsp;<span class=muted>male-CNS mushroom body · FlyHash · live Kenyon-cell raster</span></header>
<main>
 <div>
  <div class=card>
   <div class=k>Sniff a plan / spec / retro</div>
   <textarea id=t placeholder="e.g. a service built around a shared global registry that handlers mutate..."></textarea>
   <button onclick=sniff()>Sniff &#129716;</button>
   <span id=st class=muted></span>
  </div>
  <div class=card>
   <div class=k>Novelty</div><div id=nov class=big>–</div>
   <div class=bar><span id=novbar style=width:0%></span></div>
   <div class=muted id=novhint></div>
  </div>
  <div class=card>
   <div class=k>Nearest retro (exemplar)</div>
   <div id=nn class=v>–</div>
  </div>
  <div class=card>
   <div class=k>Matched pattern family</div>
   <div id=fam class=v>–</div>
   <div id=famkw></div>
   <div class=k style=margin-top:8px>Structural guidance <span id=src class=muted></span></div>
   <div id=guid class=v>–</div>
  </div>
 </div>
 <div>
  <div class=card><span class=k>Kenyon-cell tag</span> — <span id=cap class=muted>active cells light up; similar smells share cells</span>
   <div><canvas id=c width=780 height=690></canvas></div>
  </div>
 </div>
</main>
<script>
let NKC=%NKC%, COLS=%COLS%;
const cv=document.getElementById('c'), cx=cv.getContext('2d');
function draw(active){
 const set=new Set(active), cell=12, cols=COLS, rows=Math.ceil(NKC/cols);
 cx.clearRect(0,0,cv.width,cv.height);
 for(let i=0;i<NKC;i++){const x=(i%cols)*cell, y=Math.floor(i/cols)*cell;
  if(set.has(i)){cx.fillStyle='#2ee6a6';cx.fillRect(x+1,y+1,cell-2,cell-2);}
  else{cx.fillStyle='#121a26';cx.fillRect(x+1,y+1,cell-2,cell-2);} }
}
draw([]);
async function sniff(){
 const text=document.getElementById('t').value.trim(); if(!text)return;
 document.getElementById('st').textContent=' sniffing...';
 const r=await fetch('/api/sniff?text='+encodeURIComponent(text));
 const d=await r.json(); document.getElementById('st').textContent='';
 draw(d.active);
 const nv=Math.round(d.novelty*100);
 document.getElementById('nov').textContent=nv+'% novel';
 document.getElementById('novbar').style.width=nv+'%';
 document.getElementById('novhint').textContent = nv>70?'looks new — the fly hasn\\'t smelled this before':'familiar territory';
 document.getElementById('nn').innerHTML = d.nearest? ('<code>'+d.nearest.repo+d.nearest.epic+'</code> · overlap '+d.nearest.overlap.toFixed(2)):'–';
 if(d.family){
  document.getElementById('fam').innerHTML='<span class=big>'+d.family.label+'</span> · '+d.family.repos.join(', ')+' · match '+Math.round(d.family.overlap*100)+'%';
  document.getElementById('famkw').innerHTML=(d.family.keywords||[]).map(k=>'<span class=tag>'+k+'</span>').join('');
  document.getElementById('guid').textContent=d.family.guidance;
  document.getElementById('src').textContent='('+d.family.source+')';
 } else {
  document.getElementById('fam').textContent='no matching pattern — structurally novel';
  document.getElementById('famkw').innerHTML=''; document.getElementById('guid').textContent='–'; document.getElementById('src').textContent='';
 }
}
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            html = (PAGE.replace("%NKC%", str(PIPE.con.flyhash.n_kc))
                        .replace("%COLS%", "65"))
            return self._send(200, html, "text/html; charset=utf-8")
        if u.path == "/api/sniff":
            text = (parse_qs(u.query).get("text", [""])[0]).strip()
            if not text:
                return self._send(400, json.dumps({"error": "empty"}))
            tag = PIPE.encode_text(text)
            nearest = None
            for tr in PIPE.mem.traces:
                ov = overlap(tag, tr.tag)
                if nearest is None or ov > nearest["overlap"]:
                    nearest = {"repo": tr.meta["repo"], "epic": tr.meta["epic"], "overlap": ov}
            adv = advise_plan(PIPE, text)
            fam = None
            if adv.matched:
                fam = {"label": next((f.label for f in PIPE.families
                                      if f.repos == adv.repos), "pattern"),
                       "repos": adv.repos, "overlap": adv.overlap,
                       "keywords": adv.keywords[:8], "guidance": adv.guidance,
                       "source": adv.source}
            return self._send(200, json.dumps({
                "n_kc": int(PIPE.con.flyhash.n_kc),
                "active": [int(i) for i in tag],
                "novelty": float(PIPE.mem.novelty(tag)),
                "nearest": nearest,
                "family": fam,
            }))
        return self._send(404, json.dumps({"error": "not found"}))


def main() -> None:
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"dashboard on http://127.0.0.1:{PORT}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
