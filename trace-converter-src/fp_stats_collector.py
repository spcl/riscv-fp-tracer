from __future__ import annotations
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Union, Tuple
from instruction import Instruction
from utils import hex64_to_fp64, hex64_to_fp32


class Float16(object):
    EXP_BIAS = 15
    EXP_SIZE = 5
    MANTISSA_SIZE = 10
    """
    A wrapper class around np.float16 that allows a more
    fine-grained access to different parts of the fp16 representation
    (i.e., sign, exp, mantissa).
    """
    def __init__(self, fp_num: Union[np.float16, np.float64]) -> None:
        """
        If the given number of is in np.float64, convert it to np.float16.
        """
        if not isinstance(fp_num, np.float16):
            fp_num = np.float16(fp_num)
        self.num = fp_num

    @property
    def is_denorm(self) -> bool:
        """
        Returns True if the float represents a denormalized value.
        i.e., the exponent bits are all 0s.
        """
        return self.exp_val == -Float16.EXP_BIAS

    @property
    def sign(self) -> int:
        """
        Returns the sign bit of the float.
        """
        return self.num < 0
    
    @property
    def exponent(self) -> int:
        """
        Returns the exponent bits of the float.
        """
        exp_bits = self.num.view(np.uint16) >> self.MANTISSA_SIZE & 0x1F
        return exp_bits

    @property
    def mantissa(self) -> int:
        """
        Returns the mantissa bits of the float.
        """
        mantissa_bits = self.num.view(np.uint16) & 0x3FF
        return mantissa_bits
    
    @property
    def exp_val(self) -> int:
        """
        Returns the actual value represented by the exponent,
        which can be calculated by first converting the exponent
        bits to an integer then subtracting the bias 15.
        """
        return self.exponent - Float16.EXP_BIAS
    

    @property
    def mantissa_with_implicit_bit(self) -> int:
        """
        Returns the mantissa bits along with the implicit bit.
        Note that for normalized values, the implicit bit is 1,
        while for the denormalized value, the implicit bit is 0.
        """
        if self.is_denorm:
            # If the value is denormalized
            return self.mantissa
        # If the value is normalized, appends a 1 to the front
        # of the mantissa bits
        return (1 << Float16.MANTISSA_SIZE) | self.mantissa

    def subtract(self, other: Float16) -> Tuple[int, int]:
        """
        Subtracts two Float16 numbers and returns the result
        mantissa as bits along with the number of places
        it needs to be shifted as a tuple.
        """
        # The mantissa of this float
        m1 = self.mantissa_with_implicit_bit
        # The other float's mantissa
        m2 = other.mantissa_with_implicit_bit
        # Aligns the radix points of hte two floats
        exp_diff = self.exp_val - other.exp_val
        if exp_diff > 0:
            # If this number's exp is larger
            m1 <<= exp_diff
        elif exp_diff < 0:
            # If the other number's exp is larger
            m2 <<= -exp_diff
        # Performs bitwise subtraction
        res = m1 - m2
        shift = 0
        # If both values are denormalized
        if (self.is_denorm and other.is_denorm) or res == 0:
            return (res, shift)
        # Computes the number of places the result needs
        # to be shifted in order to be normalized
        while (abs(res) >> 10) == 0:
            res <<= 1
            shift += 1
            # print(f"[DEBUG] res: {bin(res)}")
        return (res, shift)
    


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

        # A list that keeps track of the number of places shifted
        # by the most significant '1' when floating point subtractions happen.
        # It is used as a metric to measure catastrophic cancellation.
        self.pos_shifts = []

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
                if dep not in self.ex_map:
                    # If an entry for the dependent register/address
                    # has not been initialized, it will be set to False
                    # by default regardless of what its value actually is
                    if self.debug:
                        print(f"[WARNING] STATS COLLECTOR: value of {dep} "
                              "has not been initialized")
                    self.ex_map[dep] = False
                
                dep_orf_udf = self.ex_map[dep] or dep_orf_udf
                
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
                print(f"[DEBUG] Overflow fp16: {fp16_val}, fp{'64' if is_double else '32'}: {hp_val}")
            return True
        
        if underflow and not dep_orf_udf:
            self.underflow_count += 1
            if self.debug:
                print(f"[DEBUG] Underflow fp16: {fp16_val}, fp{'64' if is_double else '32'}: {hp_val}")
            return True

    def comp_catastrophic_cancellation(self, a: np.float16, b: np.float16) \
        -> None:
        """
        Computes statistics about catastrophic cancellation in floating point
        subtractions. Given two numbers, this function first wraps them
        in our own custom class, then computes the number of places
        shifted by the most significant '1' in the resulting value.
        For instance, if `a` as a fixed point is 1000001, and b
        as a fixed point is 1000000, the result would be 0000001,
        whose most significant '1' needs to shifted by 6 places to
        be normalized again.
        """
        if not self.enabled:
            return

        if not np.isfinite(a) or not np.isfinite(b):
            # If either a or b is not finite
            return

        f1 = Float16(a)
        f2 = Float16(b)

        _, shift = f1.subtract(f2)
        self.pos_shifts.append(shift)

    def plot_sub_shits_dist(self) -> None:
        """
        Plots the distribution of bits shift in floating point subtraction.
        """
        if not self.enabled:
            print("Statistics collection not enabled...")
            return
        
        plt.hist(self.pos_shifts)
        plt.savefig("fp_sub_shifts_dist.png", format="png")
        plt.close()

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
        plt.savefig("input_dist.png", format="png")
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