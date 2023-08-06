from typing import Union
import numpy as np
from functools import lru_cache


def to_fp16(val: Union[np.float64, np.float32]) -> np.float16:
    """
    Converts the given value, either in FP64 or FP32 to FP16.
    """
    return np.float16(val)


@lru_cache(maxsize=10000)
def hex64_to_uint64(hex_string: str) -> np.uint64:
    """
    Converts the given hex string to np.uint64.
    """
    value = int(hex_string, 16)
    return np.uint64(value)


@lru_cache(maxsize=10000)
def hex64_to_fp64(hex_string: str, endianness: str = "little") -> np.float64:
    """
    Converts the given hex string to np.float64. By default,
    it is assumed that the the given hex string represents a floating
    point number in little endian.
    """
    hex_value = int(hex_string, 16)
    hex_bytes = hex_value.to_bytes(8, endianness)
    fp64_value = np.frombuffer(hex_bytes, dtype=np.float64, count=1)[0]
    return fp64_value


@lru_cache(maxsize=10000)
def hex64_to_fp32(hex_string: str, endianness: str = "little") -> np.float32:
    """
    Converts a 64-bit hex string to np.float32 by extracting the
    lower 32 bits from the string.
    """
    hex_value = int(hex_string[-8:], 16).to_bytes(4, endianness)
    fp32_value = np.frombuffer(hex_value, dtype=np.float32, count=1)[0]
    return fp32_value


@lru_cache(maxsize=1000)
def hex64_to_fp16(hex_string: str, is_double: bool = True) -> np.float16:
    """
    Converts a 64-bit hex string directly to np.float16.
    If `is_double` is True, it will convert the hex string to
    FP64 then to FP16. Otherwise, it will first convert it
    to FP32 then to FP16.
    """
    if is_double:
        intermediate_val = hex64_to_fp64(hex_string)
    else:
        intermediate_val = hex64_to_fp32(hex_string)
    return to_fp16(intermediate_val)


def hex16_to_fp16(hex_string: str, endianness: str = "little") -> np.float16:
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



def smooth(data: np.array, window_size: int = 10) -> np.array:
    """
    Applies a moving average filter to the given data according
    to the specified window size.
    Implementation from:
    https://stackoverflow.com/questions/11352047/finding-moving-average-from-data-points-in-python
    """
    vec = np.cumsum(np.insert(data, 0, 0)) 
    ma_vec = (vec[window_size:] - vec[:-window_size]) / window_size
    return ma_vec