"""D455 local browser preview. Run beside frame_protocol.py instead of test_middleware.py."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

from frame_protocol import MAX_MESSAGE, unpack_frame


PAGE = """<!doctype html><html lang="ko"><meta charset="utf-8">
<title>D455 카메라</title>
<style>body{background:#17202a;color:white;font:16px system-ui;margin:3vw}
img{width:100%;max-width:1280px}p{color:#aad9e3}</style>
<h1>D455 카메라</h1><p id="status">연결 확인 중</p>
<div>왼쪽 · 컬러 / 오른쪽 · 깊이 (검정은 측정값 없음)</div>
<img src="/stream.mjpg" alt="송신기 프레임 대기 중">
<script>async function poll(){try{const r=await fetch('/status');
const s=await r.json();document.querySelector('#status').textContent=
s.state+(s.seq===null?'':' · 마지막 수신 #'+s.seq)}
catch(e){document.querySelector('#status').textContent='미들웨어 연결 끊김'}
setTimeout(poll,700)}poll()</script></html>"""

condition = threading.Condition()
sender_gate = threading.Lock()
latest = None
version = 0
received_at = 0
connected = False
jpeg = None
jpeg_version = 0


def receive(ws):
    global latest, version, received_at, connected
    if not sender_gate.acquire(False):
        ws.close(1008, "One sender only")
        return
    try:
        with condition:
            connected = True
            received_at = 0
        for packet in ws:
            metadata, color, depth = unpack_frame(packet)
            with condition:
                latest = (metadata, color, depth)
                version += 1
                received_at = time.monotonic()
                condition.notify_all()
            ws.send(json.dumps({"seq": metadata["seq"], "depth_ok": True}))
    except ConnectionClosed:
        pass
    except Exception as exc:
        print("수신 오류:", exc, flush=True)
        ws.close(1008, "Invalid frame")
    finally:
        with condition:
            connected = False
        sender_gate.release()


def render():
    global jpeg, jpeg_version
    last = 0
    while True:
        with condition:
            condition.wait_for(lambda: version > last)
            last = version
            meta, color, depth = latest
        values = np.clip(depth.astype(np.float32) * (255 * meta["depth_scale"] / 3), 0, 255)
        visual = cv2.applyColorMap(values.astype(np.uint8), cv2.COLORMAP_JET)
        visual[depth == 0] = 0
        pair = np.hstack((color, visual))
        ok, encoded = cv2.imencode(".jpg", pair, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with condition:
                jpeg, jpeg_version = encoded.tobytes(), last
                condition.notify_all()


class Preview(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=d455")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.connection.settimeout(5)
            last = 0
            try:
                while True:
                    with condition:
                        if not condition.wait_for(lambda: jpeg_version > last, timeout=5):
                            # Harmless multipart whitespace also detects a closed idle tab.
                            data = None
                        else:
                            last, data = jpeg_version, jpeg
                    if data is None:
                        self.wfile.write(b"\r\n")
                        continue
                    header = f"--d455\r\nContent-Type: image/jpeg\r\nContent-Length: {len(data)}\r\n\r\n"
                    self.wfile.write(header.encode() + data + b"\r\n")
            except (ConnectionError, TimeoutError):
                return
        elif self.path == "/status":
            with condition:
                state = "송신기 대기" if not connected else "프레임 대기"
                if connected and received_at:
                    state = "LIVE" if time.monotonic() - received_at < 2 else "수신 지연"
                payload = {"state": state, "seq": latest[0]["seq"] if latest else None}
            self.respond("application/json", json.dumps(payload).encode())
        elif self.path in ("/", "/preview"):
            self.respond("text/html; charset=utf-8", PAGE.encode())
        else:
            self.send_error(404)

    def respond(self, mime, data):
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    try:
        with ThreadingHTTPServer(("127.0.0.1", 8088), Preview) as http:
            with serve(receive, "127.0.0.1", 8765, compression=None,
                       max_size=MAX_MESSAGE, max_queue=1, close_timeout=2) as ws:
                threading.Thread(target=render, daemon=True).start()
                threading.Thread(target=ws.serve_forever, daemon=True).start()
                print("브라우저: http://127.0.0.1:8088/preview", flush=True)
                try:
                    http.serve_forever()
                finally:
                    ws.shutdown()
    except KeyboardInterrupt:
        pass
    except OSError as exc:
        raise SystemExit(f"서버 시작 오류 (8088, 8765 포트 확인): {exc}")
