"""A loopback HTTP origin for the actual Rust executable, not a transport mock."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
from types import SimpleNamespace
from urllib.parse import parse_qs, unquote, urlparse


@contextmanager
def origin(monkeypatch, payload=b"abc", etag='"e"'):
    state = SimpleNamespace(payload=payload, etag=etag, gets=[], listings=[],
                            fail=False, wrong_range=False, ignore_range=False, gate=threading.Event(), encoding=None,
                            delay=0.0, active=0, peak=0, guard=threading.Lock())
    state.gate.set()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if "list-type" in query:
                prefix = query.get("prefix", [""])[0]
                state.listings.append(prefix)
                content = ''.join(f'<Contents><Key>{key}</Key><Size>{len(state.payload)}</Size>'
                                  '<ETag>&quot;e&quot;</ETag><LastModified>2026-10-03T21:59:00Z</LastModified></Contents>'
                                  for key in [prefix+'KAAA20261003_215500_V06'])
                body = ('<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                        + content + '<IsTruncated>false</IsTruncated></ListBucketResult>').encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            state.gets.append((unquote(parsed.path), self.headers.get("Range"), self.headers.get("If-Match")))
            # How many object GETs the executable holds open at once.
            with state.guard:
                state.active += 1
                state.peak = max(state.peak, state.active)
            try:
                time.sleep(state.delay)
            finally:
                with state.guard:
                    state.active -= 1
            state.gate.wait(30)
            if state.fail:
                self.send_error(503)
                return
            body = state.payload
            span = self.headers.get("Range")
            if span and not state.ignore_range:
                first, last = map(int, span.removeprefix('bytes=').split('-'))
                self.send_response(206)
                total = len(body) + int(state.wrong_range)
                self.send_header('Content-Range', f'bytes {first}-{last}/{total}')
                body = body[first:last+1]
            else:
                self.send_response(200)
            self.send_header('ETag', state.etag)
            if state.encoding:
                self.send_header('Content-Encoding', state.encoding)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv('GPUWM_RW_FETCH_TEST_ORIGIN', f'http://127.0.0.1:{server.server_port}/')
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
