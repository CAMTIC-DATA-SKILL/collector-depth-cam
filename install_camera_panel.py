#!/usr/bin/env python3
"""D455 viewer patch. Default: validate only. --apply: backup and install.
No process restart, package installation, or network/firewall change.
"""
import argparse
import ast
import datetime
import hashlib
import json
from pathlib import Path
import shutil

PROXY = '''"""Relay the local camera receiver without decoding or re-encoding images."""
from http.client import HTTPConnection, HTTPException
import json
from fastapi import APIRouter, HTTPException as APIError
from starlette.responses import StreamingResponse
from starlette.background import BackgroundTask

router = APIRouter(prefix="/api/camera/d455", tags=["camera"])

def open_receiver(path):
    conn = HTTPConnection("127.0.0.1", 8089, timeout=10)
    try:
        conn.request("GET", path)
        response = conn.getresponse()
        if response.status != 200:
            raise OSError("Camera receiver unavailable")
        return conn, response
    except (OSError, HTTPException):
        conn.close()
        raise APIError(503, "Camera receiver unavailable") from None

@router.get("/status")
def status():
    conn, response = open_receiver("/status")
    try:
        return json.loads(response.read(65536))
    except (OSError, HTTPException, ValueError):
        raise APIError(502, "Invalid camera status") from None
    finally:
        response.close()
        conn.close()

@router.get("/stream.mjpg")
def stream():
    conn, response = open_receiver("/stream.mjpg")
    content_type = response.getheader("Content-Type", "")
    def close():
        response.close()
        conn.close()
    if not content_type.startswith("multipart/x-mixed-replace;"):
        close()
        raise APIError(502, "Invalid camera stream")
    def chunks():
        try:
            while True:
                chunk = response.read1(65536)
                if not chunk:
                    break
                yield chunk
        except (OSError, HTTPException):
            return
        finally:
            close()
    return StreamingResponse(chunks(), media_type=content_type,
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        background=BackgroundTask(close))
'''

PANEL = '''
      <!-- d455-camera-panel-v1 -->
      <div id="d455-camera-panel" style="background:#fff;margin-bottom:12px">
        <button id="d455-camera-toggle" type="button" aria-expanded="false">카메라 보기</button>
        <div id="d455-camera-image" hidden style="margin-top:8px;background:#fff"></div>
      </div>
'''
SCRIPT = '''
<script>
(() => {
  const button = document.getElementById('d455-camera-toggle');
  const box = document.getElementById('d455-camera-image');
  let image = null;
  function close() {
    if (image) { image.removeAttribute('src'); image.remove(); image = null; }
    box.hidden = true;
    button.textContent = '카메라 보기';
    button.setAttribute('aria-expanded', 'false');
  }
  button.addEventListener('click', () => {
    if (image) { close(); return; }
    image = document.createElement('img');
    image.alt = '';
    image.setAttribute('aria-label', 'D455 컬러 및 깊이 영상');
    image.style.cssText = 'display:block;width:100%;height:auto;background:white';
    image.onerror = () => { close(); button.textContent = '카메라 재시도'; };
    box.appendChild(image);
    box.hidden = false;
    button.textContent = '카메라 닫기';
    button.setAttribute('aria-expanded', 'true');
    image.src = '/api/camera/d455/stream.mjpg';
  });
  window.addEventListener('pagehide', close);
})();
</script>
'''

def build(root):
    main = root / 'app/main.py'
    viewer = root / 'app/static/viewer.html'
    proxy = root / 'app/camera_proxy.py'
    a, b = main.read_text(), viewer.read_text()
    if proxy.exists() or 'd455-camera-panel-v1' in b or 'camera_proxy' in a:
        raise ValueError('Camera patch already exists. No changes made.')
    import_anchor = 'from .api import router\n'
    route_anchor = '    app.include_router(router)\n'
    panel_anchor = '<aside class="panel detail-panel" id="detail-panel">'
    for text, anchor in ((a, import_anchor), (a, route_anchor), (b, panel_anchor), (b, '</body>')):
        if text.count(anchor) != 1:
            raise ValueError('Unexpected viewer structure; send current app/main.py and viewer.html. No changes made.')
    a = a.replace(import_anchor, import_anchor + 'from .camera_proxy import router as camera_router\n')
    a = a.replace(route_anchor, route_anchor + '    app.include_router(camera_router)\n')
    b = b.replace(panel_anchor, panel_anchor + PANEL).replace('</body>', SCRIPT + '</body>')
    ast.parse(a)
    ast.parse(PROXY)
    return {'app/main.py': a, 'app/static/viewer.html': b, 'app/camera_proxy.py': PROXY}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None

def restore(root, backup):
    manifest = json.loads((backup / 'manifest.json').read_text())
    if str(root) != manifest['root']:
        raise ValueError('Backup belongs to a different project.')
    for rel, record in manifest['files'].items():
        if digest(root / rel) != record['installed_sha256']:
            raise ValueError(f'{rel} changed since installation; automatic restore refused.')
    for rel, record in manifest['files'].items():
        if record['existed']:
            shutil.copy2(backup / rel, root / rel)
        else:
            (root / rel).unlink()
    print('RESTORED. Restart the web service using its normal start method.')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', required=True, type=Path)
    group = p.add_mutually_exclusive_group()
    group.add_argument('--apply', action='store_true')
    group.add_argument('--restore', type=Path)
    args = p.parse_args()
    root = args.project.expanduser().resolve()
    if args.restore:
        restore(root, args.restore.expanduser().resolve())
        return
    changes = build(root)
    print('CHECK OK: camera relay + right panel button; upstream 127.0.0.1:8089')
    for rel in changes:
        print('  ' + rel)
    if not args.apply:
        print('DRY RUN: no files changed. Add --apply to install.')
        return
    backup = root / ('camera-backup-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    backup.mkdir()
    manifest = {'root': str(root), 'files': {}}
    for rel, text in changes.items():
        target = root / rel
        existed = target.exists()
        manifest['files'][rel] = {'existed': existed, 'installed_sha256': hashlib.sha256(text.encode()).hexdigest()}
        if existed:
            dest = backup / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, dest)
    (backup / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    try:
        for rel, text in changes.items():
            (root / rel).write_text(text, encoding='utf-8')
    except BaseException:
        for rel, record in manifest['files'].items():
            if record['existed']:
                shutil.copy2(backup / rel, root / rel)
            else:
                (root / rel).unlink(missing_ok=True)
        raise
    print('INSTALLED. Backup: ' + str(backup))
    print('Web service restart required. This script does NOT restart it.')

if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, SyntaxError) as exc:
        raise SystemExit('ERROR: ' + str(exc))
