import numpy as np


def fp64_to_fp16(fp64_val: np.float64) -> np.float16:
    """
    Converts the given FP64 value to FP16.
    """
    return np.float16(fp64_val)


def hex_to_fp64(hex_string: str, endianness: str = "little") -> np.float64:
    """
    Converts the given hex string to np.float64. By default,
    it is assumed that the the given hex string represents a floating
    point number in little endian.
    """
    hex_value = int(hex_string, 16)
    hex_bytes = hex_value.to_bytes(8, endianness)
    fp64_value = np.frombuffer(hex_bytes, dtype=np.float64, count=1)[0]
    return fp64_value


def hex64_to_fp16(hex_string: str) -> np.float16:
    """
    Converts a 64bit hex string directly to np.float16 by
    first converting it to fp64 then to fp16.
    """
    return fp64_to_fp16(hex_to_fp64(hex_string))


def hex_to_fp16(hex_string: str, endianness: str = "little") -> np.float16:
    """
    Converts the given hex string to np.float16. By default,
    it is assumed that the the given hex string represents a floating
    point number in little endian.
    """
    hex_value = int(hex_string, 16).to_bytes(2, endianness)
    fp16_value = np.frombuffer(hex_value, dtype=np.float16, count=1)[0]
    return fp16_value


def fp16_to_hex(fp16_value: np.float16):
    """
    Converts the given fp16 value to its corresponding HEX form,
    assuming the number is in little endian.
    """
    hex_value = hex(fp16_value.view(np.uint16))
    hex_string = hex_value[2:].upper().zfill(4)
    return hex_string