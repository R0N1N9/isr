"""Local web server behind the site editor.

Runs on 127.0.0.1 only. The editor page posts the plan after every change;
the server saves it to ``site.json``, re-runs the analysis and sends back the
results and map overlays, so findings update while you draw. Nothing leaves
the machine.
"""

from __future__ import annotations

import json
import mimetypes
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import model as M
from .analysis import Analysis
from .editor import editor_html
from .report import write_report

MAX_BODY = 20 * 1024 * 1024


class _Handler(BaseHTTPRequestHandler):
    server_version = "isr-site"
    project: Path
    lock = threading.Lock()
    quiet = True

    def log_message(self, fmt, *args):  # keep the terminal clean
        if not self.quiet:
            super().log_message(fmt, *args)

    # A page on another site could otherwise reach this server through the browser.
    def _local(self) -> bool:
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
        origin = self.headers.get("Origin")
        ok_origin = origin is None or origin.split("//")[-1].rsplit(":", 1)[0].strip("[]") in ("127.0.0.1", "localhost", "::1")
        return host in ("127.0.0.1", "localhost", "::1") and ok_origin

    def _send(self, code, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode())

    def do_GET(self):
        if not self._local():
            return self._json({"error": "forbidden"}, 403)
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            return self._send(200, editor_html().encode(), "text/html; charset=utf-8")
        if path == "/api/site":
            return self._json(M.load(self.project / "site.json"))
        if path.startswith("/files/"):
            from urllib.parse import unquote
            root = self.project.resolve()
            target = (root / unquote(path[len("/files/"):])).resolve()
            if root not in target.parents or not target.is_file():
                return self._json({"error": "not found"}, 404)
            ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            return self._send(200, target.read_bytes(), ctype)
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        if not self._local() or "json" not in (self.headers.get("Content-Type") or ""):
            return self._json({"error": "forbidden"}, 403)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if not 0 < n <= MAX_BODY:
                return self._json({"error": "bad request"}, 400)
            model = M.normalise(json.loads(self.rfile.read(n)))
            if not isinstance(model.get("origin"), dict) or len(model.get("bounds") or []) != 4:
                return self._json({"error": "not a site model"}, 400)
            with self.lock:
                M.save(model, self.project / "site.json")
                an = Analysis(model)
                if self.path == "/api/analyse":
                    return self._json({"result": an.result, "overlays": an.overlays_b64()})
                if self.path == "/api/report":
                    write_report(model, self.project, an)
                    return self._json({"url": "/files/report.html", "path": str(self.project / "report.html")})
            self._json({"error": "not found"}, 404)
        except Exception as e:  # report the failure to the page instead of dropping the connection
            traceback.print_exc()
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)


def make_server(project: Path, port: int = 8765) -> ThreadingHTTPServer:
    handler = type("Handler", (_Handler,), {"project": Path(project)})
    for p in (port, 0):
        try:
            return ThreadingHTTPServer(("127.0.0.1", p), handler)
        except OSError:
            continue
    raise OSError("could not open a local port for the editor")


def serve(project: Path, port: int = 8765, open_browser: bool = True) -> None:
    srv = make_server(project, port)
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print(f"Site editor: {url}\nEverything you draw is saved to {Path(project) / 'site.json'}. Press Ctrl+C to stop.")
    if open_browser:
        import webbrowser
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        srv.server_close()
