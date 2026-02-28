import struct
from typing import Tuple

HEADER_FMT = "<Q"
HEADER_SIZE = struct.calcsize(HEADER_FMT)


def unpack_message(message: bytes) -> Tuple[int, bytes]:
    if len(message) < HEADER_SIZE:
        raise ValueError("Mensaje demasiado corto para header.")

    (frame_id,) = struct.unpack(HEADER_FMT, message[:HEADER_SIZE])
    jpeg_bytes = message[HEADER_SIZE:]
    return int(frame_id), jpeg_bytes
