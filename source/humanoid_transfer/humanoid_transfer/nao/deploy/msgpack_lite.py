"""The part of MessagePack that LoLA speaks: maps of strings to arrays of numbers.

LoLA (NAOqi 2.8) exchanges one MessagePack map per cycle over a Unix socket.
Its messages hold only nil, booleans, integers, floats, strings, arrays and
maps, so this encoder and decoder cover those and nothing else; anything else
raises. Standard library only, so it runs wherever Python does, the robot
included.
"""

from __future__ import annotations

import struct


def packb(value) -> bytes:
    out = bytearray()
    _pack(value, out)
    return bytes(out)


def _pack(value, out: bytearray) -> None:
    if value is None:
        out.append(0xC0)
    elif value is True:
        out.append(0xC3)
    elif value is False:
        out.append(0xC2)
    elif isinstance(value, int):
        if 0 <= value < 0x80:
            out.append(value)
        elif -32 <= value < 0:
            out.append(value & 0xFF)
        elif -(2**31) <= value < 2**31:
            out += b"\xd2" + struct.pack(">i", value)
        else:
            out += b"\xd3" + struct.pack(">q", value)
    elif isinstance(value, float):
        # LoLA reads 32-bit floats; a double is decoded too, but is twice the size.
        out += b"\xca" + struct.pack(">f", value)
    elif isinstance(value, str):
        data = value.encode("utf-8")
        n = len(data)
        if n < 32:
            out.append(0xA0 | n)
        elif n < 256:
            out += b"\xd9" + bytes([n])
        else:
            out += b"\xda" + struct.pack(">H", n)
        out += data
    elif isinstance(value, (list, tuple)):
        n = len(value)
        if n < 16:
            out.append(0x90 | n)
        else:
            out += b"\xdc" + struct.pack(">H", n)
        for item in value:
            _pack(item, out)
    elif isinstance(value, dict):
        n = len(value)
        if n < 16:
            out.append(0x80 | n)
        else:
            out += b"\xde" + struct.pack(">H", n)
        for key, item in value.items():
            _pack(key, out)
            _pack(item, out)
    else:
        # NumPy scalars and arrays arrive here: convert them first.
        if hasattr(value, "tolist"):
            _pack(value.tolist(), out)
        else:
            raise TypeError(f"cannot pack {type(value).__name__}")


def unpackb(data: bytes):
    value, end = _unpack(memoryview(data), 0)
    if end != len(data):
        raise ValueError(f"{len(data) - end} trailing bytes")
    return value


def _unpack(data: memoryview, i: int):
    b = data[i]
    i += 1
    if b <= 0x7F:
        return b, i
    if b >= 0xE0:
        return b - 0x100, i
    if 0x80 <= b <= 0x8F:
        return _map(data, i, b & 0x0F)
    if 0x90 <= b <= 0x9F:
        return _array(data, i, b & 0x0F)
    if 0xA0 <= b <= 0xBF:
        n = b & 0x1F
        return bytes(data[i : i + n]).decode("utf-8"), i + n
    if b == 0xC0:
        return None, i
    if b == 0xC2:
        return False, i
    if b == 0xC3:
        return True, i
    fixed = {
        0xCA: (">f", 4), 0xCB: (">d", 8),
        0xCC: (">B", 1), 0xCD: (">H", 2), 0xCE: (">I", 4), 0xCF: (">Q", 8),
        0xD0: (">b", 1), 0xD1: (">h", 2), 0xD2: (">i", 4), 0xD3: (">q", 8),
    }  # fmt: skip
    if b in fixed:
        fmt, size = fixed[b]
        return struct.unpack(fmt, data[i : i + size])[0], i + size
    if b in (0xD9, 0xDA, 0xDB):
        size = {0xD9: 1, 0xDA: 2, 0xDB: 4}[b]
        n = int.from_bytes(data[i : i + size], "big")
        i += size
        return bytes(data[i : i + n]).decode("utf-8"), i + n
    if b in (0xC4, 0xC5, 0xC6):
        size = {0xC4: 1, 0xC5: 2, 0xC6: 4}[b]
        n = int.from_bytes(data[i : i + size], "big")
        i += size
        return bytes(data[i : i + n]), i + n
    if b in (0xDC, 0xDD):
        size = 2 if b == 0xDC else 4
        return _array(data, i + size, int.from_bytes(data[i : i + size], "big"))
    if b in (0xDE, 0xDF):
        size = 2 if b == 0xDE else 4
        return _map(data, i + size, int.from_bytes(data[i : i + size], "big"))
    raise ValueError(f"unsupported MessagePack type 0x{b:02x}")


def _array(data: memoryview, i: int, n: int):
    items = []
    for _ in range(n):
        item, i = _unpack(data, i)
        items.append(item)
    return items, i


def _map(data: memoryview, i: int, n: int):
    items = {}
    for _ in range(n):
        key, i = _unpack(data, i)
        items[key], i = _unpack(data, i)
    return items, i


__all__ = ["packb", "unpackb"]
