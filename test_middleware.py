import json
import threading
import time

import cv2
import numpy as np
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

from frame_protocol import MAX_MESSAGE, unpack_frame


def main():
    lock = threading.Lock()
    client_gate = threading.Lock()
    stop = threading.Event()
    state = {
        "latest": None,
        "connected": False,
        "received_at": 0.0,
    }

    def receive(websocket):
        # 이번 실습에서는 송신기 한 대만 받습니다.
        if not client_gate.acquire(blocking=False):
            websocket.close(1008, "One sender only")
            return

        count = total_bytes = 0
        interval_start = time.perf_counter()

        try:
            with lock:
                state["connected"] = True
            print("송신기 연결됨")

            while not stop.is_set():
                try:
                    packet = websocket.recv(timeout=0.5)
                except TimeoutError:
                    continue

                metadata, color, depth = unpack_frame(packet)

                with lock:
                    state["latest"] = (metadata, color, depth)
                    state["received_at"] = time.perf_counter()

                # 영상 복원과 깊이값 확인이 끝난 뒤 응답합니다.
                websocket.send(json.dumps({
                    "seq": metadata["seq"],
                    "depth_ok": True,
                }))

                count += 1
                total_bytes += len(packet)
                elapsed = time.perf_counter() - interval_start

                if elapsed >= 1:
                    print(
                        f"RX {count / elapsed:.1f} FPS | "
                        f"{total_bytes / elapsed / 1_000_000:.2f} MB/s | "
                        f"depth dtype={depth.dtype} | depth_OK"
                    )
                    interval_start = time.perf_counter()
                    count = total_bytes = 0

        except ConnectionClosed:
            print("송신기 연결 종료")

        except Exception as error:
            print(f"수신 오류: {error}")
            websocket.close(1008, "Invalid frame")

        finally:
            with lock:
                state["connected"] = False
            client_gate.release()

    window = "Middleware | Color / Depth | Q: quit"

    try:
        with serve(
            receive,
            "127.0.0.1",
            8765,
            compression=None,
            max_size=MAX_MESSAGE,
            max_queue=1,
            close_timeout=2,
        ) as server:
            worker = threading.Thread(
                target=server.serve_forever, daemon=True
            )
            worker.start()

            print("미들웨어 대기 중: ws://127.0.0.1:8765")
            cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

            try:
                while True:
                    with lock:
                        item = state["latest"]
                        connected = state["connected"]
                        received_at = state["received_at"]

                    if item is None:
                        preview = np.zeros((480, 1280, 3), np.uint8)
                        label = "Waiting for sender..."
                    else:
                        metadata, color, depth = item

                        # 표시용 변환. 복원된 depth 원본은 유지합니다.
                        depth_m = (
                            depth.astype(np.float32)
                            * metadata["depth_scale"]
                        )
                        display = (
                            np.clip(depth_m / 3.0, 0, 1) * 255
                        ).astype(np.uint8)

                        depth_color = cv2.applyColorMap(
                            display, cv2.COLORMAP_JET
                        )
                        depth_color[depth == 0] = 0
                        preview = np.hstack((color, depth_color))

                        fresh = (
                            connected
                            and time.perf_counter() - received_at < 2
                        )
                        status = "LIVE" if fresh else "STALE"
                        label = (
                            f"seq={metadata['seq']} | "
                            f"depth OK | {status}"
                        )

                    cv2.putText(
                        preview, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2,
                    )
                    cv2.imshow(window, preview)

                    key = cv2.waitKey(10) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):
                        break

                    if cv2.getWindowProperty(
                        window, cv2.WND_PROP_VISIBLE
                    ) < 1:
                        break

            finally:
                stop.set()
                server.shutdown()
                worker.join(timeout=2)

    except KeyboardInterrupt:
        pass

    finally:
        stop.set()
        cv2.destroyAllWindows()
        print("미들웨어 종료")


if __name__ == "__main__":
    main()