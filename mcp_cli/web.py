from __future__ import annotations

import argparse
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from . import telemetry

HTML_PAGE = """<!DOCTYPE html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <title>MCP JSON-RPC Viewer</title>
    <style>
      body {
        font-family: system-ui, sans-serif;
        margin: 0;
        background: #111;
        color: #eee;
      }
      header {
        padding: 1rem 1.5rem;
        background: #222;
        border-bottom: 1px solid #333;
      }
      main {
        display: flex;
        gap: 1rem;
        padding: 1rem;
      }
      .column {
        flex: 1;
        min-width: 0;
      }
      h2 {
        margin-top: 0;
        font-size: 1rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #aaa;
      }
      .event {
        border: 1px solid #333;
        border-radius: 6px;
        padding: 0.75rem;
        margin-bottom: 0.75rem;
        background: #1b1b1b;
      }
      .event.outgoing { border-left: 3px solid #4caf50; }
      .event.incoming { border-left: 3px solid #03a9f4; }
      pre {
        margin: 0.5rem 0 0;
        white-space: pre-wrap;
        word-break: break-word;
      }
      .meta { font-size: 0.8rem; color: #8bc34a; }
      .meta span { margin-right: 0.75rem; color: #aaa; }
      footer {
        padding: 0.75rem 1.5rem;
        font-size: 0.75rem;
        color: #888;
        border-top: 1px solid #333;
      }
      button {
        background: #444;
        color: #fff;
        border: none;
        padding: 0.4rem 0.8rem;
        border-radius: 4px;
        cursor: pointer;
      }
      button:hover { background: #555; }
    </style>
  </head>
  <body>
    <header>
      <strong>MCP JSON-RPC Viewer</strong>
      <button id=\"pause\" style=\"float:right;\">Pause</button>
    </header>
    <main>
      <section class=\"column\">
        <h2>Client → Server</h2>
        <div id=\"client-out\"></div>
      </section>
      <section class=\"column\">
        <h2>Server → Client</h2>
        <div id=\"server-out\"></div>
      </section>
    </main>
    <footer>Polling every 1.5s. Use the pause button to freeze updates.</footer>
    <script>
      let lastId = -1;
      let paused = false;
      const clientCol = document.getElementById('client-out');
      const serverCol = document.getElementById('server-out');
      document.getElementById('pause').addEventListener('click', () => {
        paused = !paused;
        document.getElementById('pause').innerText = paused ? 'Resume' : 'Pause';
      });

      function renderEvent(event) {
        const container = document.createElement('div');
        container.className = `event ${event.direction}`;
        const meta = document.createElement('div');
        meta.className = 'meta';
        const ts = new Date(event.timestamp * 1000).toLocaleTimeString();
        meta.innerHTML = [
          `<span>#${event.id}</span>`,
          `<span>${ts}</span>`,
          `<span>${event.channel || ''}</span>`
        ].join('');
        const pre = document.createElement('pre');
        pre.textContent = JSON.stringify(event.payload, null, 2);
        container.appendChild(meta);
        container.appendChild(pre);
        return container;
      }

      async function poll() {
        if (paused) {
          setTimeout(poll, 1500); return;
        }
        try {
          const resp = await fetch(`/events?since=${lastId}`);
          if (!resp.ok) throw new Error('Request failed');
          const data = await resp.json();
          data.events.forEach(ev => {
            const node = renderEvent(ev);
            if (ev.role === 'client' && ev.direction === 'outgoing') {
              clientCol.prepend(node);
            } else {
              serverCol.prepend(node);
            }
            lastId = Math.max(lastId, ev.id);
          });
        } catch (err) {
          console.error(err);
        }
        setTimeout(poll, 1500);
      }
      poll();
    </script>
  </body>
</html>
"""


class _EventHandler(BaseHTTPRequestHandler):
    log_path: Optional[Path] = None
    enable_live: bool = True
    _lock = threading.Lock()

    def _load_events(self, since: int) -> List[Dict[str, object]]:
        if self.enable_live and _EventHandler.log_path is None:
            return telemetry.get_events(since)
        if _EventHandler.log_path is None:
            return []
        events = telemetry.load_events_from_file(_EventHandler.log_path)
        return [event for event in events if event.get("id", -1) > since]

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            self._serve_html()
        elif self.path.startswith("/events"):
            self._serve_events()
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def _serve_html(self) -> None:
        body = HTML_PAGE.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_events(self) -> None:
        qs = self.path.split("?", 1)
        since = -1
        if len(qs) == 2:
            for part in qs[1].split("&"):
                if part.startswith("since="):
                    try:
                        since = int(part.split("=", 1)[1])
                    except ValueError:
                        since = -1
        events = self._load_events(since)
        payload_dict = {
            "events": events,
            "last": events[-1]["id"] if events else since,
        }
        payload = json.dumps(payload_dict).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Start a simple MCP message viewer."
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to serve (default: 8765)",
    )
    parser.add_argument(
        "--events",
        type=Path,
        help=(
            "Optional path to an events JSONL file (defaults to live in-process "
            "events if available)."
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    _EventHandler.log_path = args.events.resolve() if args.events else telemetry.log_path()
    _EventHandler.enable_live = args.events is None

    server = ThreadingHTTPServer((args.host, args.port), _EventHandler)
    print(f"MCP viewer available at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping viewer...")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
