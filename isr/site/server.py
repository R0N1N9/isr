"""Local web server behind the site editor.

Runs on 127.0.0.1 only. The editor page posts the plan after every change;
the server saves it to ``site.json``, re-runs the analysis and sends back the
results and map overlays, so findings update while you draw. Nothing leaves
the machine.

``project_get`` / ``project_post`` serve one project under any URL prefix, so
the same code backs both ``survey edit`` (one project at ``/``) and the
combined app (many projects at ``/survey/<job>/``).
"""

from __future__ import annotations

import json
import mimetypes
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from . import model as M
from .analysis import Analysis
from .editor import editor_html
from .report import write_report

MAX_BODY = 20 * 1024 * 1024
LOCK = threading.Lock()


class LocalHandler(BaseHTTPRequestHandler):
    """Request helpers shared by the editor server and the combined app."""

    server_version = "ronin-sec"
    quiet = True

    def log_message(self, fmt, *args):  # keep the terminal clean
        if not self.quiet:
            super().log_message(fmt, *args)

    # A page on another site could otherwise reach this server through the browser.
    def local(self) -> bool:
        ok = ("127.0.0.1", "localhost", "::1")
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
        origin = self.headers.get("Origin")
        return host in ok and (origin is None or origin.split("//")[-1].rsplit(":", 1)[0].strip("[]") in ok)

    def send(self, code, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def json(self, obj, code=200):
        self.send(code, json.dumps(obj, default=str).encode())

    def html(self, text: str):
        self.send(200, text.encode(), "text/html; charset=utf-8")

    def body(self):
        """The posted JSON, or None after an error response has been sent."""
        if "json" not in (self.headers.get("Content-Type") or ""):
            return self.json({"error": "forbidden"}, 403)
        n = int(self.headers.get("Content-Length") or 0)
        if not 0 < n <= MAX_BODY:
            return self.json({"error": "bad request"}, 400)
        return json.loads(self.rfile.read(n))

    def file(self, root: Path, rel: str, inject: str = ""):
        """Serve a file from inside ``root`` (never from outside it)."""
        root = root.resolve()
        target = (root / unquote(rel)).resolve()
        if root not in target.parents or not target.is_file():
            return self.json({"error": "not found"}, 404)
        data = target.read_bytes()
        if inject and target.suffix == ".html":
            data = data.replace(b"</body>", inject.encode() + b"</body>", 1)
        self.send(200, data, mimetypes.guess_type(target.name)[0] or "application/octet-stream")


def project_get(h: LocalHandler, project: Path, path: str, base: str = ""):
    if path in ("", "/", "/index.html"):
        return h.html(editor_html(base))
    if path == "/api/site":
        return h.json(M.load(project / "site.json"))
    if path.startswith("/files/"):
        return h.file(project, path[len("/files/"):])
    h.json({"error": "not found"}, 404)


def project_post(h: LocalHandler, project: Path, path: str, base: str = ""):
    body = h.body()
    if body is None:
        return
    model = M.normalise(body) if isinstance(body, dict) else {}
    if not isinstance(model.get("origin"), dict) or len(model.get("bounds") or []) != 4:
        return h.json({"error": "not a site model"}, 400)
    with LOCK:
        M.save(model, project / "site.json")
        an = Analysis(model, project)
        if path == "/api/analyse":
            return h.json({"result": an.result, "overlays": an.overlays_b64()})
        if path == "/api/report":
            write_report(model, project, an)
            return h.json({"url": f"{base}/files/report.html", "path": str(project / "report.html")})
    h.json({"error": "not found"}, 404)


def guarded(fn):
    """Reject non-local requests and report failures to the page instead of dropping the connection."""
    def wrap(self):
        if not self.local():
            return self.json({"error": "forbidden"}, 403)
        try:
            return fn(self)
        except Exception as e:
            traceback.print_exc()
            self.json({"error": f"{type(e).__name__}: {e}"}, 500)
    return wrap


class _Handler(LocalHandler):
    project: Path

    @guarded
    def do_GET(self):
        project_get(self, self.project, self.path.split("?", 1)[0])

    @guarded
    def do_POST(self):
        project_post(self, self.project, self.path)


def bind(handler, port: int) -> ThreadingHTTPServer:
    for p in (port, 0):
        try:
            return ThreadingHTTPServer(("127.0.0.1", p), handler)
        except OSError:
            continue
    raise OSError("could not open a local port")


def make_server(project: Path, port: int = 8765) -> ThreadingHTTPServer:
    return bind(type("Handler", (_Handler,), {"project": Path(project)}), port)


def run(srv: ThreadingHTTPServer, banner: str, open_browser: bool = True) -> None:
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print(f"{banner}: {url}\nPress Ctrl+C to stop.")
    if open_browser:
        import webbrowser
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        srv.server_close()


def serve(project: Path, port: int = 8765, open_browser: bool = True) -> None:
    print(f"Everything you draw is saved to {Path(project) / 'site.json'}.")
    run(make_server(project, port), "ISRStudio Survey editor", open_browser)
