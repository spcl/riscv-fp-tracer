import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Optional, Dict
from instruction import Instruction
from utils import hex64_to_fp64, hex64_to_fp32


class FpStatsCollector(object):
    """
    An object that collects and logs the statistics related
    to FP16 floating point numbers, e.g., the input number
    distribution, the occurrences of overflow/underflow, etc.
    """
    def __init__(self, out_file: str, ignore_ex_prop: bool = False, 
                 enabled: bool = False, debug: bool = False) -> None:
        """
        @param out_file: The output file to which all the statistics
        will be recorded.
        @param enabled: Only collects data when `enabled`
        is set to True.
        @param ignore_ex_prop: If true, will only consider the
        instructions that trigger overflow/underflow and ignore
        exception propagation when computing metrics. As an example,
        if this argument is True and fadd fa5,fa5,fa4 causes an overflow
        in fa5, then all the instructions that depend on the value in fa5
        will not be counted as overflow instructions.
        """
        self.enabled = enabled
        self.out_file = out_file
        self.input_vals: List[np.float16] = []
        self.overflow_count = 0
        self.underflow_count = 0
        self.total_count = 0
        
        self.debug = debug

        self.ignore_ex_prop = ignore_ex_prop
        # A map that is used to keep track of which registers
        # and memory addresses sar storing overflown/underflown values
        self.ex_map: Dict[str, bool] = {}

    def add_input_val(self, fp_val: np.float16) -> None:
        """
        Appends a single input value to the list.
        """
        if self.enabled:
            assert isinstance(fp_val, np.float16)
            self.input_vals.append(fp_val)

    def count_overflow_underflow(self, fp16_val: np.float16,
                                 hex: str, is_double: bool = True,
                                 dst: Optional[str] = None,
                                 src: List[str] = [],
                                 is_insn: bool = True) -> bool:
        """
        Detects if an overflow or underflow has happened by comparing
        the np.float64 value converted from its corresponding hex string
        with the computed np.float32 or np.float64 value.

        @param fp16_val: The FP16 value to be checked.
        @param hex: The hex string representation of the 
        FP32/FP64 value against which the given FP16 value
        will be compared.
        @param is_double: A boolean indicating whether the given hex
        string is for FP64 or FP32.
        @param dst: The target register/memory address that will be
        storing the given FP16 value. Note that `dst` and `src`
        are only useful when `ignore_ex_prop` is set to True.
        @param src: The source registers/memory address on which
        the operation depends.
        @param is_insn: The overflow and underflow counters will
        only be affected if `is_insn` is set to True.

        @return: True if the given instruction causes an overflow/underflow
        """
        if not self.enabled:
            return False
        
        overflow = False
        underflow = False

        # Converts the string to a higher precision value
        if is_double:
            hp_val = hex64_to_fp64(hex)
        else:
            hp_val = hex64_to_fp32(hex)

        # An overflow happens when the FP64 value is not 'inf' or 'NaN'
        # but the FP16 value is not finite
        if not np.isfinite(fp16_val) and np.isfinite(hp_val):
            overflow = True
            # self.overflow_count += 1
            # return
        
        # An underflow happens when the FP64 value is not 0
        # but its corresponding FP16 value is 0
        if fp16_val == 0 and hp_val != 0:
            underflow = True
            # self.underflow_count += 1
            # return
        
        dep_orf_udf = False
        if self.ignore_ex_prop:
            # Checks whether the values stored in the dependent
            # registers/memory address are already overflown/underflown
            for dep in src:
                try:
                    dep_orf_udf = self.ex_map[dep] or dep_orf_udf
                except KeyError:
                    print(f"[ERROR] STATS COLLECTOR: value of {dep} has not been initialized")
                    exit(-1)
                
            # If one of the dependencies already have an invalid value
            # the status of the current instruction will not be influenced
            # by the values of `overflow` or `underflow`
            self.ex_map[dst] = overflow or underflow or dep_orf_udf

        if not is_insn:
            return False

        self.total_count += 1
        if overflow and not dep_orf_udf:
            self.overflow_count += 1
            if self.debug:
                print(f"[DEBUG] Overflow fp16: {fp16_val}, fp{'64' if is_double else '32'} {hex}")
            return True
        
        if underflow and not dep_orf_udf:
            self.underflow_count += 1
            if self.debug:
                print(f"[DEBUG] Underflow fp16: {fp16_val}, fp{'64' if is_double else '32'} {hex}")
            return True


    def plot_input_dist(self) -> None:
        """
        Plots the input value distribution.
        """
        if not self.enabled:
            print("Statistics collection not enabled...")
            return
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
        if not self.enabled:
            print("Statistics collection not enabled...")
            return
        print("Overflow/Underflow statistics")
        print(f"Overflow : {self.overflow_count}/{self.total_count}"
              f" ({self.overflow_count / self.total_count * 100:.2f}%)")
        print(f"Underflow: {self.underflow_count}/{self.total_count}"
              f" ({self.underflow_count / self.total_count * 100:.2f}%)")
        