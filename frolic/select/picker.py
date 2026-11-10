"""Browser person picker: serve the contact sheet as a local page with one checkbox per person and wait for Submit.

`select.selection_file` (one id per line) answers the stage without the page, e.g. to repeat a saved choice.
"""

from __future__ import annotations

import getpass
import html
import json
import os
import socket
import sys
import threading
import webbrowser
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from frolic.log import get_logger

log = get_logger(__name__)


def _parse_ids(text: str, ids: list[str]) -> list[str]:
    out = []
    for tok in text.replace(",", " ").split():
        tok = tok.strip()
        if tok in ids:
            out.append(tok)
        elif tok.isdigit() and int(tok) < len(ids):
            out.append(ids[int(tok)])
        elif tok:
            log.warning("unknown id %r ignored", tok)
    return list(dict.fromkeys(out))


def pick_file(path: Path, ids: list[str]) -> list[str]:
    lines = [ln.split("#")[0].strip() for ln in Path(path).read_text().splitlines()]
    return _parse_ids(" ".join(ln for ln in lines if ln), ids)


_PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FROLIC person picker</title><style>
body{{font-family:system-ui,sans-serif;background:#16181b;color:#e4e4e4;margin:16px}}
.p{{border:2px solid #333;border-radius:6px;padding:8px;margin:8px 0;cursor:pointer}}
.p.on{{border-color:#3fa34d;background:#18241a}} .p img{{height:150px;margin:2px;border-radius:3px}}
.st{{color:#9ab;font-size:13px}} label{{font-size:17px;font-weight:600;cursor:pointer}}
.bar{{position:sticky;top:0;background:#16181b;padding:8px 0;border-bottom:1px solid #333}}
button{{font-size:15px;padding:6px 14px;margin-right:6px}} #msg{{color:#fc6;margin-left:8px}}
</style></head><body>
<div class="bar"><b>Tick the people to reconstruct, then Submit.</b><br>
<button onclick="setAll(true)">All</button><button onclick="setAll(false)">None</button>
<button id="go" onclick="submitSel()">Submit</button><span id="msg"></span></div>
{rows}
<script>
const boxes=[...document.querySelectorAll('input[type=checkbox]')];
function sync(){{boxes.forEach(b=>b.closest('.p').classList.toggle('on',b.checked));
 document.getElementById('go').disabled=!boxes.some(b=>b.checked);}}
function setAll(v){{boxes.forEach(b=>b.checked=v);sync();}}
boxes.forEach(b=>b.onchange=sync);
document.querySelectorAll('.p').forEach(d=>d.onclick=e=>{{if(e.target.tagName!=='INPUT'&&e.target.tagName!=='LABEL'){{
 const b=d.querySelector('input');b.checked=!b.checked;sync();}}}});
async function submitSel(){{const ids=boxes.filter(b=>b.checked).map(b=>b.value);
 const r=await fetch('select',{{method:'POST',headers:{{'content-type':'application/json'}},body:JSON.stringify({{ids:ids}})}});
 document.getElementById('msg').textContent=await r.text();
 if(r.ok){{boxes.forEach(b=>b.disabled=true);document.getElementById('go').disabled=true;}}}}
sync();
</script></body></html>"""


def picker_page(ids: list[str], people: list[dict[str, Any]], imgs: dict[str, list[str]], checked: list[str]) -> str:
    """imgs: id -> list of <img> tags (base64 tiles from the contact sheet)."""
    info = {p["id"]: p for p in people}
    rows = []
    for k, pid in enumerate(ids):
        p = info.get(pid, {})
        st = p.get("stature_m")
        stats = (
            f"height {st:.2f} m; " if st else ""
        ) + f"tracked {p.get('duration_s', 0):.1f} s; median {p.get('median_views', 0):.0f} cameras"
        on = " checked" if pid in checked else ""
        rows.append(
            f"<div class='p'><label><input type='checkbox' value='{html.escape(pid)}'{on}> [{k}] {html.escape(pid)}</label>"
            f" <span class='st'>{stats}</span><br>{''.join(imgs.get(pid, []))}</div>"
        )
    return _PAGE.format(rows="\n".join(rows))


def tunnel_hint(host: str, port: int, env: dict[str, str] | None = None) -> list[str]:
    """How to reach the picker from another computer: ssh commands for this machine (Slurm job: via the login node)."""
    env = dict(os.environ if env is None else env)
    user = env.get("USER") or getpass.getuser()
    fqdn = socket.getfqdn()
    lines = []
    if env.get("SLURM_JOB_ID"):
        node = env.get("SLURMD_NODENAME") or socket.gethostname()
        login = env.get("SLURM_SUBMIT_HOST") or "<login-node>"
        lines.append(f"Slurm job {env['SLURM_JOB_ID']} on {node}; on your computer run:")
        if host in ("0.0.0.0", "::"):
            lines.append(f"  ssh -N -L {port}:{node}:{port} {user}@{login}")
        else:
            lines.append(f"  ssh -N -L {port}:127.0.0.1:{port} -J {user}@{login} {user}@{node}")
    elif host in ("127.0.0.1", "localhost", "::1", "0.0.0.0", "::"):
        if env.get("TERM_PROGRAM") == "vscode" or env.get("VSCODE_IPC_HOOK_CLI"):
            lines.append("VS Code forwards this port by itself (Ports tab); otherwise, on your computer run:")
        else:
            lines.append("from another computer run:")
        lines.append(f"  ssh -N -L {port}:127.0.0.1:{port} {user}@{fqdn}")
    else:
        return []
    lines.append(f"  then open http://127.0.0.1:{port}/")
    return lines


def pick_browser(
    ids: list[str],
    people: list[dict[str, Any]],
    imgs: dict[str, list[str]],
    host: str = "127.0.0.1",
    port: int = 8770,
    open_browser: bool = True,
    timeout_s: float = 0.0,
    checked: list[str] | None = None,
    on_ready: Callable[[str], None] | None = None,
) -> list[str]:
    """Serve the picker page and block until a non-empty selection is submitted (timeout_s 0 = wait for ever)."""
    page = picker_page(ids, people, imgs, list(ids) if checked is None else checked).encode()
    done = threading.Event()
    result: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a: Any) -> None:
            pass

        def _send(self, code: int, body: bytes, ctype: str = "text/plain; charset=utf-8") -> None:
            self.send_response(code)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path.split("?")[0] in ("/", "/index.html"):
                self._send(200, page, "text/html; charset=utf-8")
            else:
                self._send(404, b"not found")

        def do_POST(self) -> None:
            if self.path != "/select" or done.is_set():
                self._send(409 if done.is_set() else 404, b"selection already submitted" if done.is_set() else b"")
                return
            try:
                got = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))["ids"]
            except (ValueError, KeyError, TypeError):
                self._send(400, b"bad request")
                return
            sel = [i for i in ids if i in {str(g) for g in got}]
            if not sel:
                self._send(400, b"tick at least one person")
                return
            result.extend(sel)
            self._send(200, f"saved: {', '.join(sel)}. You can close this page.".encode())
            done.set()

    try:
        srv = ThreadingHTTPServer((host, port), Handler)
    except OSError:
        log.warning("port %d busy: using a free port", port)
        srv = ThreadingHTTPServer((host, 0), Handler)
    url = f"http://{host}:{srv.server_address[1]}/"
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    hint = tunnel_hint(host, srv.server_address[1])
    print("\n  FROLIC person picker: " + url, *("  " + h for h in hint), "", sep="\n", file=sys.stderr, flush=True)
    for h in hint:
        log.info("picker: %s", h.strip())
    log.info("person picker at %s; waiting for Submit", url)
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:  # no browser on this machine: the URL is printed
            pass
    if on_ready is not None:
        on_ready(url)
    try:
        if not done.wait(timeout_s if timeout_s > 0 else None):
            raise TimeoutError(f"no selection submitted at {url} within {timeout_s:.0f} s")
    finally:
        srv.shutdown()
        srv.server_close()
    return result


def pick(
    ids: list[str],
    people: list[dict[str, Any]],
    imgs: dict[str, list[str]],
    sc: Any,
    on_ready: Callable[[str], None] | None = None,
) -> tuple[list[str], str]:
    """selection_file when it exists, else the browser picker; returns (ids, mode used)."""
    if sc.selection_file is not None and Path(sc.selection_file).exists():
        return pick_file(Path(sc.selection_file), ids), "file"
    sel = pick_browser(
        ids,
        people,
        imgs,
        host=sc.host,
        port=sc.port,
        open_browser=sc.open_browser,
        timeout_s=sc.timeout_s,
        checked=[i for i in sc.ids if i in ids] or None,
        on_ready=on_ready,
    )
    return sel, "browser"
