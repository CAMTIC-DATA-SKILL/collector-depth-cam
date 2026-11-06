import time

import cv2
import numpy as np
import pyrealsense2 as rs


def main():
    pipeline = rs.pipeline()
    config = rs.config()

    # 깊이: 픽셀마다 16비트 원본 깊이값
    config.enable_stream(
        rs.stream.depth, 640, 480, rs.format.z16, 30
    )

    # 컬러: OpenCV에서 바로 표시하기 위한 BGR 형식
    config.enable_stream(
        rs.stream.color, 640, 480, rs.format.bgr8, 30
    )

    started = False
    window_name = "D455 | Color / Depth | Q: quit"

    try:
        profile = pipeline.start(config)
        started = True

        device = profile.get_device()
        depth_scale = device.first_depth_sensor().get_depth_scale()

        print("장치:", device.get_info(rs.camera_info.name))
        print("깊이값 1단위에 해당하는 거리:", depth_scale, "m")
        print("영상 창을 선택한 뒤 Q 또는 ESC를 누르면 종료합니다.")

        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        interval_start = time.perf_counter()
        frame_count = 0
        first_frame = True

        while True:
            # 컬러·깊이 프레임을 기다립니다.
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # 원본 데이터를 NumPy 배열로 읽습니다.
            depth_raw = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            if first_frame:
                print(
                    "컬러 배열:",
                    color_image.shape,
                    color_image.dtype,
                )
                print(
                    "깊이 배열:",
                    depth_raw.shape,
                    depth_raw.dtype,
                )
                first_frame = False

            # 깊이 화면의 중앙 픽셀을 미터 단위로 환산합니다.
            height, width = depth_raw.shape
            center_raw = int(depth_raw[height // 2, width // 2])

            if center_raw == 0:
                center_text = "invalid"
            else:
                center_m = center_raw * depth_scale
                center_text = f"{center_m:.3f} m"

            # 표시용으로만 0~3m 범위를 색상으로 변환합니다.
            # depth_raw 원본 배열은 변경하지 않습니다.
            display_max_m = 3.0
            depth_m = depth_raw.astype(np.float32) * depth_scale

            depth_display = (
                np.clip(depth_m / display_max_m, 0, 1) * 255
            ).astype(np.uint8)

            depth_color = cv2.applyColorMap(
                depth_display, cv2.COLORMAP_JET
            )

            # 측정되지 않은 픽셀은 검정으로 표시합니다.
            depth_color[depth_raw == 0] = 0

            cv2.drawMarker(
                depth_color,
                (width // 2, height // 2),
                (255, 255, 255),
                cv2.MARKER_CROSS,
                15,
                1,
            )

            cv2.putText(
                depth_color,
                f"Center: {center_text}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            # 왼쪽 컬러, 오른쪽 깊이 시각화
            preview = np.hstack((color_image, depth_color))
            cv2.imshow(window_name, preview)

            frame_count += 1
            now = time.perf_counter()
            elapsed = now - interval_start

            if elapsed >= 1.0:
                fps = frame_count / elapsed

                # 압축 전 배열 크기이며 실제 네트워크 전송량은 아닙니다.
                raw_bytes = color_image.nbytes + depth_raw.nbytes

                print(
                    f"처리 FPS: {fps:.1f} | "
                    f"깊이 중앙값: {center_text} | "
                    f"컬러+깊이 원본: {raw_bytes / 1_000_000:.3f} MB"
                )

                interval_start = now
                frame_count = 0

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

            if cv2.getWindowProperty(
                window_name, cv2.WND_PROP_VISIBLE
            ) < 1:
                break

    except KeyboardInterrupt:
        print("\n사용자가 종료했습니다.")

    except RuntimeError as error:
        print(f"\n카메라 실행 오류: {error}")

    finally:
        if started:
            pipeline.stop()

        cv2.destroyAllWindows()
        print("카메라 사용을 종료했습니다.")


if __name__ == "__main__":
    main()