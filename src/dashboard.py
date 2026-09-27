"""Small local dashboard using Python's built-in HTTP server. Run after training."""
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import json
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HTML = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>SecureNet IDS</title>
<style>body{font:16px system-ui;background:#f2f5f8;color:#182936;max-width:1050px;margin:36px auto;padding:0 24px}h1{margin-bottom:6px}section{background:white;padding:24px;border-radius:12px;margin:20px 0}.metrics{display:flex;gap:32px;flex-wrap:wrap}.metric{font-size:30px;font-weight:700}small{display:block;font-size:14px;font-weight:400}button,select{padding:10px 16px;font-size:15px}button{background:#245675;color:white;border:0;border-radius:5px;cursor:pointer}img{max-width:100%}pre{white-space:pre-wrap}footer{font-size:14px;color:#52616b}</style>
<h1>SecureNet intrusion detection</h1><p>Information Security · CLO 4 · NSL-KDD proof of concept</p>
<section><h2>Unseen test results</h2><div class="metrics" id="metrics"></div><p id="risk"></p></section>
<section><h2>Predict a recorded connection</h2><p>Simulated incoming traffic. No live packets are captured. Attack scores are not calibrated probabilities.</p><select id="row"></select> <button id="predict">Predict</button> <button id="replay">Replay ten records</button><pre id="output">Choose a record to begin.</pre></section>
<section><h2>Data exploration</h2><img src="/eda.png" alt="Training and test distribution and common attacks"></section>
<section><h2>Confusion matrix</h2><img src="/confusion_matrix.png" alt="Confusion matrix on unseen test data"></section>
<section><h2>Which attacks are missed</h2><img src="/attack_recall.png" alt="Detection recall for common test attacks"></section>
<footer>Use alongside an existing IDS and analyst review. Historical benchmark results do not prove production safety.</footer>
<script>
const out=document.querySelector('#output');
fetch('/metrics').then(r=>r.json()).then(m=>{document.querySelector('#metrics').innerHTML=[['Accuracy',m.accuracy],['Attack precision',m.attack_precision],['Attack recall',m.attack_recall]].map(([k,v])=>`<div class="metric">${(v*100).toFixed(2)}%<small>${k}</small></div>`).join('');document.querySelector('#risk').textContent=`Missed attacks: ${m.fn} · False alarms: ${m.fp} · Test connections: ${m.test_rows}`;}).catch(e=>out.textContent=e);
for(let i=0;i<10;i++)document.querySelector('#row').add(new Option(`Record ${i+1}`,i));
async function predict(i){let r=await fetch('/predict?row='+i);let d=await r.json();if(!r.ok)throw Error(d.error);return `Record ${i+1}: ${d.prediction} | attack score ${d.attack_score.toFixed(3)}`;}
document.querySelector('#predict').onclick=async()=>{try{out.textContent=await predict(Number(document.querySelector('#row').value));}catch(e){out.textContent=e;}};
document.querySelector('#replay').onclick=async function(){this.disabled=true;out.textContent='SIMULATED STREAM\\n';try{for(let i=0;i<10;i++){out.textContent+=await predict(i)+'\\n';await new Promise(r=>setTimeout(r,300));}}catch(e){out.textContent+=e;}finally{this.disabled=false;}};
</script></html>'''

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        from urllib.parse import urlparse, parse_qs
        request = urlparse(self.path)
        try:
            if request.path == '/':
                body, mime = HTML.encode(), 'text/html; charset=utf-8'
            elif request.path == '/metrics':
                body, mime = (ROOT / 'results/metrics.json').read_bytes(), 'application/json'
            elif request.path == '/predict':
                row = int(parse_qs(request.query).get('row', ['0'])[0])
                if not 0 <= row < len(self.server.rows):
                    raise ValueError('Row is outside the demo data.')
                scores = self.server.model.predict_proba(self.server.rows.iloc[[row]])[0]
                pred = int(self.server.model.classes_[scores.argmax()])
                result = {'prediction': 'ATTACK' if pred else 'NORMAL', 'attack_score': float(scores[list(self.server.model.classes_).index(1)])}
                body, mime = json.dumps(result).encode(), 'application/json'
            elif request.path in ['/eda.png', '/confusion_matrix.png', '/attack_recall.png']:
                body, mime = (ROOT / 'images' / request.path[1:]).read_bytes(), 'image/png'
            else:
                self.send_error(404); return
            self.send_response(200)
        except (ValueError, KeyError) as exc:
            body, mime = json.dumps({'error':str(exc)}).encode(), 'application/json'
            self.send_response(400)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

if __name__ == '__main__':
    if not (ROOT / 'models/ids_pipeline.joblib').exists():
        raise SystemExit('Train first: python src/train.py')
    server = HTTPServer(('127.0.0.1', 8000), Handler)
    server.model = joblib.load(ROOT / 'models/ids_pipeline.joblib')
    server.rows = pd.read_csv(ROOT / 'data/demo_features.csv')
    print('Open http://127.0.0.1:8000 — press Ctrl+C to stop.', flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
