import json
import struct
import zlib

import cv2
import numpy as np


MAX_MESSAGE = 16 * 1024 * 1024


def pack_frame(color, depth, metadata):
    """컬러 JPEG + 원본 깊이 PNG + 설명정보를 묶습니다."""
    if color.dtype != np.uint8 or depth.dtype != np.uint16:
        raise ValueError("영상 자료형이 올바르지 않습니다.")

    ok_color, jpg = cv2.imencode(
        ".jpg", color, [cv2.IMWRITE_JPEG_QUALITY, 80]
    )
    ok_depth, png = cv2.imencode(
        ".png", depth, [cv2.IMWRITE_PNG_COMPRESSION, 1]
    )

    if not ok_color or not ok_depth:
        raise RuntimeError("영상 압축 실패")

    header = dict(metadata)
    header.update(
        protocol=1,
        color_bytes=int(jpg.nbytes),
        depth_bytes=int(png.nbytes),
        height=int(depth.shape[0]),
        width=int(depth.shape[1]),
        depth_crc32=zlib.crc32(
            depth.astype("<u2", copy=False).tobytes()
        ),
    )

    header_bytes = json.dumps(
        header, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")

    packet = (
        struct.pack("!I", len(header_bytes))
        + header_bytes
        + jpg.tobytes()
        + png.tobytes()
    )

    if len(packet) > MAX_MESSAGE:
        raise ValueError("전송 메시지가 크기 제한을 초과했습니다.")

    return packet


def unpack_frame(packet):
    """받은 메시지를 설명정보·컬러·원본 깊이로 복원합니다."""
    if not isinstance(packet, bytes) or not 4 <= len(packet) <= MAX_MESSAGE:
        raise ValueError("잘못된 바이너리 메시지")

    header_size = struct.unpack("!I", packet[:4])[0]
    if not 0 < header_size <= 16384 or 4 + header_size >= len(packet):
        raise ValueError("잘못된 설명정보 길이")

    offset = 4 + header_size
    metadata = json.loads(packet[4:offset])

    if metadata.get("protocol") != 1:
        raise ValueError("지원하지 않는 메시지 버전")

    color_size = metadata["color_bytes"]
    depth_size = metadata["depth_bytes"]

    if (
        not isinstance(color_size, int)
        or not isinstance(depth_size, int)
        or min(color_size, depth_size) <= 0
        or offset + color_size + depth_size != len(packet)
    ):
        raise ValueError("영상 데이터 길이 불일치")

    color = cv2.imdecode(
        np.frombuffer(packet[offset:offset + color_size], np.uint8),
        cv2.IMREAD_COLOR,
    )
    depth = cv2.imdecode(
        np.frombuffer(packet[offset + color_size:], np.uint8),
        cv2.IMREAD_UNCHANGED,
    )

    shape = (metadata["height"], metadata["width"])

    if color is None or depth is None:
        raise ValueError("영상 복원 실패")
    if depth.dtype != np.uint16 or depth.shape != shape:
        raise ValueError("깊이 자료형 또는 크기 불일치")
    if color.shape != (*shape, 3):
        raise ValueError("컬러 영상 크기 불일치")

    crc = zlib.crc32(depth.astype("<u2", copy=False).tobytes())
    if crc != metadata["depth_crc32"]:
        raise ValueError("송신 원본과 수신 깊이값 불일치")

    return metadata, color, depth