import sqlite3

from flask import Flask, jsonify, render_template_string

app = Flask(__name__)
DB_PATH = "saathi_memory.db"

HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Saathi — Call Analytics</title>
<style>
  body {
    font-family: -apple-system, Segoe UI, Roboto, sans-serif;
    background: #0f172a;
    color: #e2e8f0;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 48px 16px;
    margin: 0;
  }
  h1 { color: #2dd4bf; font-size: 24px; margin-bottom: 4px; }
  .subtitle { color: #94a3b8; font-size: 14px; margin-bottom: 32px; }
  .cards { display: flex; gap: 20px; flex-wrap: wrap; justify-content: center; }
  .card {
    background: #1e293b;
    padding: 24px 36px;
    border-radius: 14px;
    text-align: center;
    min-width: 150px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
  }
  .card .num { font-size: 44px; font-weight: 700; }
  .total .num { color: #38bdf8; }
  .success .num { color: #4ade80; }
  .failed .num { color: #f87171; }
  .label { margin-top: 6px; font-size: 13px; color: #94a3b8; letter-spacing: 0.3px; }
  .rate { margin-top: 28px; color: #94a3b8; font-size: 14px; }
  .footer { margin-top: 40px; font-size: 12px; color: #475569; }
</style>
</head>
<body>
<h1>🩺 Saathi — Call Analytics</h1>
<div class="subtitle">Live counts from real calls. No caller data shown.</div>
<div class="cards">
  <div class="card total"><div class="num" id="total">–</div><div class="label">TOTAL CALLS</div></div>
  <div class="card success"><div class="num" id="success">–</div><div class="label">SUCCESSFUL</div></div>
  <div class="card failed"><div class="num" id="failed">–</div><div class="label">FAILED</div></div>
</div>
<div class="rate" id="rate"></div>
<div class="footer">Success = caller received safe guidance, a facility lookup, or an appropriate escalation.</div>
<script>
async function loadStats() {
  const res = await fetch('/api/stats');
  const data = await res.json();
  document.getElementById('total').innerText = data.total;
  document.getElementById('success').innerText = data.success;
  document.getElementById('failed').innerText = data.failed;
  const rate = data.total > 0 ? Math.round((data.success / data.total) * 100) : 0;
  document.getElementById('rate').innerText = data.total > 0 ? `Success rate: ${rate}%` : '';
}
loadStats();
setInterval(loadStats, 4000);
</script>
</body>
</html>"""


@app.route("/")
def dashboard():
    return render_template_string(HTML)


@app.route("/api/stats")
def stats():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("SELECT outcome, COUNT(*) FROM calls GROUP BY outcome")
    rows = dict(cur.fetchall())
    conn.close()
    success = rows.get("success", 0)
    failed = rows.get("failed", 0)
    return jsonify({"total": success + failed, "success": success, "failed": failed})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8502, debug=False)