import json
import threading
import time

import numpy as np
import pyrealsense2 as rs
from websockets.sync.client import connect

from frame_protocol import pack_frame


def main():
    condition = threading.Condition()
    stop = threading.Event()
    state = {"latest": None, "error": None}

    def capture():
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(
            rs.stream.color, 640, 480, rs.format.bgr8, 30
        )
        config.enable_stream(
            rs.stream.depth, 640, 480, rs.format.z16, 30
        )
        started = False

        try:
            profile = pipeline.start(config)
            started = True
            scale = profile.get_device().first_depth_sensor().get_depth_scale()
            sequence = 0

            while not stop.is_set():
                frames = pipeline.wait_for_frames(timeout_ms=5000)
                received_ns = time.perf_counter_ns()

                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                # SDK 프레임 수명과 별개로 사용할 수 있도록 복사합니다.
                color = np.asanyarray(color_frame.get_data()).copy()
                depth = np.asanyarray(depth_frame.get_data()).copy()
                sequence += 1

                metadata = {
                    "seq": sequence,
                    "depth_scale": scale,
                    "color_frame_number": color_frame.get_frame_number(),
                    "depth_frame_number": depth_frame.get_frame_number(),
                    "color_timestamp_ms": color_frame.get_timestamp(),
                    "depth_timestamp_ms": depth_frame.get_timestamp(),
                    "color_timestamp_domain": str(
                        color_frame.get_frame_timestamp_domain()
                    ),
                    "depth_timestamp_domain": str(
                        depth_frame.get_frame_timestamp_domain()
                    ),
                }

                with condition:
                    state["latest"] = (
                        metadata, color, depth, received_ns
                    )
                    condition.notify_all()

        except Exception as error:
            with condition:
                state["error"] = str(error)
                condition.notify_all()

        finally:
            if started:
                pipeline.stop()

    worker = threading.Thread(target=capture, daemon=True)

    try:
        with connect(
            "ws://127.0.0.1:8765",
            compression=None,
            proxy=None,
            max_size=65536,
            open_timeout=5,
            close_timeout=2,
        ) as websocket:
            print("미들웨어 연결 성공. Ctrl+C로 종료합니다.")
            worker.start()

            last_sequence = 0
            count = skipped = total_bytes = 0
            encode_sum = delay_sum = 0.0
            interval_start = time.perf_counter()

            while True:
                with condition:
                    ready = condition.wait_for(
                        lambda: state["error"] is not None
                        or (
                            state["latest"] is not None
                            and state["latest"][0]["seq"] > last_sequence
                        ),
                        timeout=10,
                    )

                    if state["error"] is not None:
                        raise RuntimeError(state["error"])
                    if not ready:
                        raise RuntimeError("새 카메라 프레임 수신 시간 초과")

                    metadata, color, depth, received_ns = state["latest"]

                encode_start = time.perf_counter()
                packet = pack_frame(color, depth, metadata)
                encode_ms = (
                    time.perf_counter() - encode_start
                ) * 1000

                websocket.send(packet)

                # 수신기가 복원 완료했다는 응답을 기다립니다.
                ack = json.loads(websocket.recv(timeout=5))
                if (
                    ack.get("seq") != metadata["seq"]
                    or not ack.get("depth_ok")
                ):
                    raise RuntimeError("수신 확인 응답 오류")

                delay_ms = (
                    time.perf_counter_ns() - received_ns
                ) / 1_000_000

                skipped += max(
                    0, metadata["seq"] - last_sequence - 1
                )
                last_sequence = metadata["seq"]

                count += 1
                total_bytes += len(packet)
                encode_sum += encode_ms
                delay_sum += delay_ms

                elapsed = time.perf_counter() - interval_start

                if elapsed >= 1:
                    print(
                        f"TX {count / elapsed:.1f} FPS | "
                        f"{total_bytes / elapsed / 1_000_000:.2f} MB/s | "
                        f"frame {total_bytes / count / 1000:.1f} kB | "
                        f"encode {encode_sum / count:.1f} ms | "
                        f"host_to_ack {delay_sum / count:.1f} ms | "
                        f"skipped {skipped} | depth_OK"
                    )

                    interval_start = time.perf_counter()
                    count = skipped = total_bytes = 0
                    encode_sum = delay_sum = 0.0

    except KeyboardInterrupt:
        print("\n송신 종료")

    except Exception as error:
        print(f"\n송신 오류: {error}")

    finally:
        stop.set()
        if worker.is_alive():
            worker.join(timeout=6)


if __name__ == "__main__":
    main()