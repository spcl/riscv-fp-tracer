import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Optional
from utils import hex64_to_fp64, hex64_to_fp32


class FpStatsCollector(object):
    """
    An object that collects and logs the statistics related
    to FP16 floating point numbers, e.g., the input number
    distribution, the occurrences of overflow/underflow, etc.
    """
    def __init__(self, out_file: str, enabled: bool = False) -> None:
        """
        @param enabled: Only collects data when `enabled`
        is set to True.
        @param out_file: The output file to which all the statistics
        will be recorded.
        """
        self.enabled = enabled
        self.out_file = out_file
        self.input_vals: List[np.float16] = []
        self.overflow_count = 0
        self.underflow_count = 0
        self.total_count = 0

    def add_input_val(self, fp_val: np.float16) -> None:
        """
        Appends a single input value to the list.
        """
        if self.enabled:
            assert isinstance(fp_val, np.float16)
            self.input_vals.append(fp_val)

    def count_overflow_underflow(self, fp16_val: np.float16,
                                 hex: str, is_double: bool = True) -> int:
        """
        Detects if an overflow or underflow has happened by comparing
        the np.float64 value converted from its corresponding hex string
        with the computed np.float32 or np.float64 value.
        """
        if not self.enabled:
            return
        
        # Converts the string to a higher precision value
        if is_double:
            hp_val = hex64_to_fp64(hex)
        else:
            hp_val = hex64_to_fp32(hex)
        self.total_count += 1
        # An overflow happens when the FP64 value is not 'inf' or 'NaN'
        # but the FP16 value is not finite
        if not np.isfinite(fp16_val) and np.isfinite(hp_val):
            self.overflow_count += 1
            print(f"[DEBUG] Overflow fp16: {fp16_val}, fp{'64' if is_double else '32'} {hex}")
            return
        
        # An underflow happens when the FP64 value is not 0
        # but its corresponding FP16 value is 0
        if fp16_val == 0 and hp_val != 0:
            self.underflow_count += 1
            print(f"[DEBUG] Underflow fp16: {fp16_val}, fp{'64' if is_double else '32'} {hex}")
            return

    def plot_input_dist(self) -> None:
        """
        Plots the input value distribution.
        """
        inputs = np.array(self.input_vals)
        # Plots only the valid values
        valid_vals = inputs[np.isfinite(inputs)]
        plt.hist(valid_vals)
        # Reports the percentage of NaN and inf
        nan_count = np.isnan(inputs).sum()
        inf_count = np.isinf(inputs).sum()
        input_size = len(inputs)
        print(f"Total number of input values: {input_size}")
        print(f"Percentage of 'NaN' in input: {nan_count / input_size * 100:.2f}%")
        print(f"Percentage of 'inf' in input: {inf_count / input_size * 100:.2f}%")
        plt.savefig("tmp.png", format="png")
        plt.close()

    
    def print_overflow_underflow_stats(self) -> None:
        """
        Displays the statistics about overflow and underflow statistics.
        """
        print("Overflow/Underflow statistics")
        print(f"Overflow : {self.overflow_count}/{self.total_count}"
              f" ({self.overflow_count / self.total_count * 100:.2f}%)")
        print(f"Underflow: {self.underflow_count}/{self.total_count}"
              f" ({self.underflow_count / self.total_count * 100:.2f}%)")
        