from __future__ import annotations
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Union, Tuple
from instruction import Instruction
from utils import hex64_to_fp64, hex64_to_fp32, smooth


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
        # Aligns the radix points of the two floats
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
                 cc_threshold: int = 5,
                 block_size: Optional[int] = None,
                 enabled: bool = False,
                 debug: bool = False) -> None:
        """
        TODO Need to log statistics collected to the output file.
        @param out_file: The output file to which all the statistics
        will be recorded.
        @param ignore_ex_prop: If true, will only consider the
        instructions that trigger overflow/underflow and ignore
        exception propagation when computing metrics. As an example,
        if this argument is True and fadd fa5,fa5,fa4 causes an overflow
        in fa5, then all the instructions that depend on the value in fa5
        will not be counted as overflow instructions.
        @param cc_threshold: A threshold that defines the number of bits that
        need to be shifted after a floating point subtraction in order
        to be considered to have caused catastrophic cancellation (CC).
        For example, if the threshold value is 3, and the number of bits
        shifted after a FP subtraction is 7 to normalize the result,
        this subtraction will raise the corresponding flag in the FCSR
        in the current hardware. This threshold is specified to mimic
        this hardware behavior.
        @param block_size: If not None, overflow, underflow and catastrophic
        cancellation analysis will be performed on a number of FP arithmetic
        instructions. This was added to produce a more accurate estimate
        of the hardware overhead of the current implementation of the
        snitch FPU and the checkpoint mechanism.
        @param enabled: Only collects data when `enabled`
        is set to True.
        """
        self.enabled = enabled
        self.out_file = out_file
        self.input_vals: List[np.float16] = []
        self.overflow_count = 0
        self.underflow_count = 0
        self.total_count = 0
        # Catastrophic cancellation related counters
        self.cc_count = 0
        self.total_sub_count = 0

        self.debug = debug

        self.ignore_ex_prop = ignore_ex_prop
        # A map that is used to keep track of which registers
        # and memory addresses are storing overflown/underflown values
        # or values that have potential high errors
        self.ex_map: Dict[str, bool] = {}


        self.cc_threshold = cc_threshold
        # A list that keeps track of the number of places shifted
        # by the most significant '1' when floating point subtractions happen.
        # It is used as a metric to measure catastrophic cancellation.
        self.pos_shifts: List[int] = []
        
        # Absolute errors and relative errors
        self.abs_errors: List[float] = []
        self.rel_errors: List[float] = []
        
        # Variables related to error analysis on a block
        self.block_size = -1 if block_size is None else block_size
        self.curr_block_insn_count = 0
        # Number of arithmetic instructions that caused overflow
        # in the current block
        self.curr_block_overflow_count = 0
        # Number of arithmetic instructions that caused underflow
        # in the current block
        self.curr_block_underflow_count = 0
        # Number of arithmetic instructions that caused catastrophic
        # cancellation in the current block
        self.curr_block_cc_count = 0
        # A list that stores a list of tuples, each containing three
        # items, which represent the number of exceptions that
        # have occurred in this block related to overflow, underflow,
        # and catastrophic cancellation respectively
        self.block_exs: List[Tuple[int, int, int]] = []


    def __check_ex_map(self, elem: str) -> bool:
        """
        A helper function that returns the value of `elem` in `ex_map`.
        If an entry has not been added in `ex_map`, it will be set to False
        by default regardless of its actual value.
        """
        if elem not in self.ex_map:
            if self.debug:
                print(f"[WARNING] STATS COLLECTOR: value of {elem} "
                        "has not been initialized in `ex_map`")
            self.ex_map[elem] = False
            return False

        return self.ex_map[elem]
        
    def add_input_val(self, fp_val: np.float16) -> None:
        """
        Appends a single input value to the list.
        """
        if self.enabled:
            assert isinstance(fp_val, np.float16)
            self.input_vals.append(fp_val)


    def update_ex_map(self, dst: str, src: List[str]) -> None:
        """
        Updates the value of the given `dst` in the exception
        tracking map with the values of the `src`. `dst` and
        `src` can be either names of the floating point registers
        or memory addresses.
        """
        if not self.enabled:
            return
        ex = False
        for dep in src:
            ex = ex or self.__check_ex_map(dep)
        self.ex_map[dst] = ex
        

    def reset_ex_map_entry(self, dst: str) -> None:
        """
        Resets the value of the given destination register or
        memory address in the exception map to False.
        """
        if not self.enabled:
            return
        
        self.ex_map[dst] = False


    def count_overflow_underflow(self, fp16_val: np.float16,
                                 hex: str, is_double: bool = True,
                                 dst: Optional[str] = None,
                                 src: List[str] = [],
                                 is_insn: bool = True) -> bool:
        """
        Detects if an overflow or underflow has happened by comparing
        the np.float16 value with the computed np.float32 or np.float64
        value converted from its corresponding hex string.

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

        @return: True if the destination register or address contains
        an overflown or underflown value.
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
        dep_vals = []
        if self.ignore_ex_prop:
            # Checks whether the values stored in the dependent
            # registers/memory address are already overflown/underflown
            for dep in src:
                dep_vals.append(self.__check_ex_map(dep))
                # dep_orf_udf = self.ex_map[dep] or dep_orf_udf
                dep_orf_udf = self.__check_ex_map(dep) or dep_orf_udf
                
            # If one of the dependencies already have an invalid value
            # the status of the current instruction will not be influenced
            # by the values of `overflow` or `underflow`
            self.ex_map[dst] = overflow or underflow or dep_orf_udf

        if not is_insn:
            return False
        
        self.total_count += 1
        print(f"[DEBUG] dep vals: {tuple(dep_vals)}")
        if overflow and not dep_orf_udf:
            self.overflow_count += 1
            self.curr_block_overflow_count += 1
            if self.debug:
                print(f"[DEBUG] Overflow fp16: {fp16_val}, fp{'64' if is_double else '32'}: {hp_val}")
            return True
        
        if underflow and not dep_orf_udf:
            self.underflow_count += 1
            self.curr_block_underflow_count += 1
            if self.debug:
                print(f"[DEBUG] Underflow fp16: {fp16_val}, fp{'64' if is_double else '32'}: {hp_val}")
            return True
        
        return dep_orf_udf


    def add_error(self, fp16_val: np.float16, hex: str,
                  is_double: bool = True) -> None:
        """
        Calculates the absolute and relative errors between the given
        np.float16 value and the higher precision np.float32 or np.float64
        value converted from its corresponding hex string.
        """
        if not self.enabled:
            return
        
        # Converts the string to a higher precision value
        if is_double:
            hp_val = hex64_to_fp64(hex)
        else:
            hp_val = hex64_to_fp32(hex)

        # Ignores cases where one of the floats is not finite
        if not np.isfinite(fp16_val) or not np.isfinite(hp_val):
            return
        
        # Computes the absolute and relative errors
        abs_error = abs(hp_val - fp16_val)
        # If higher precision value is 0, and
        # 1. The absolute error is non-zero, the relative error is 100%
        # 2. The absolute error is also zero, the relative error is 0
        if hp_val == 0:
            rel_error = 0 if abs_error == 0 else 1
        else:    
            rel_error = abs(abs_error / hp_val)

        if rel_error > 1 and self.debug:
            print(f"[DEBUG] absolute err: {abs_error}, hp val {hp_val}, fp16: {fp16_val}")
            print(f"[DEBUG] relative err: {rel_error}")
        
        self.abs_errors.append(abs_error)
        self.rel_errors.append(rel_error)

        
    def count_catastrophic_cancellation(self, a: np.float16, b: np.float16,
                                        dst: str, src: List[str])-> bool:
        """
        Computes statistics about catastrophic cancellation (CC) in 
        floating point subtractions. Given two numbers, this function
        first wraps them in our own custom class, then computes the number
        of places shifted by the most significant '1' in the resulting value.
        For instance, if `a` as a fixed point is 1000001, and b
        as a fixed point is 1000000, the result would be 0000001,
        whose most significant '1' needs to shifted by 6 places to
        be normalized again.
        
        If the number of bits shifted is above the predefined CC threshold,
        and both source registers contain valid values (i.e., values that 
        are not overflown, underflown, or caused by catastrophic cancellation),
        the CC counter will be incremented.

        @return: True if dst contains the result of catastrophic cancellation.
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
        self.total_sub_count += 1

        # Checks whether both source registers contain valid values
        # (i.e., their corresponding values in `ex_map` have to be False)
        dep_ex = False
        if self.ignore_ex_prop:
            for dep in src:
                dep_ex = dep_ex or self.__check_ex_map(dep)

        if shift > self.cc_threshold and not dep_ex:
            if self.debug:
                print(f"[DEBUG] Catastrophic cancellation: {a} - {b}, Shift: {shift}")
            self.cc_count += 1
            self.curr_block_cc_count += 1
            self.ex_map[dst] = True
            return True

        return dep_ex

    def count_block_stats(self, finalize: bool = False) -> None:
        """
        Increments an instruction counter. If the counter
        is equal to the block size, appends the exception counters
        of the current block to a list.

        If `finalize` is True, will add the exception counters
        of the current block to the result regardless of the
        instruction counter.
        """
        if not self.enabled:
            return
        
        # Increments the current block instruction counter
        self.curr_block_insn_count += 1
        if self.curr_block_insn_count == self.block_size or \
            (finalize and self.curr_block_insn_count > 1):
            # Summarizes the exceptions that have happened in this
            # code block
            block_ex_counts = (
                self.curr_block_overflow_count,
                self.curr_block_underflow_count,
                self.curr_block_cc_count
            )
            self.block_exs.append(block_ex_counts)
            
            self.curr_block_overflow_count = 0
            self.curr_block_underflow_count = 0
            self.curr_block_cc_count = 0
            self.curr_block_insn_count = 0

            if self.debug:
                print(f"[DEBUG] Exception count for block {len(self.block_exs)}: {block_ex_counts}")

    # ===========================================================
    # ===================== Result Plotting =====================
    # ===========================================================

    def output_results(self) -> None:
        """
        Outputs all the results either as text or plots. A
        wrapper function around
        - `plot_input_dist()`
        - `plot_sub_shifts_dist()`
        - `plot_error_dist()`
        - `plot_error_progression()`
        - `plot_sub_shifts_progression()`
        - `print_overflow_underflow_stats()`
        - `print_catastrophic_cancellation_stats()`
        - `output_block_stats()`
        """
        if not self.enabled:
            print("Statistics collection not enabled...")
            return
        
        self.plot_input_dist()
        self.plot_sub_shifts_dist()
        self.plot_error_dist()
        self.plot_error_progression()
        self.plot_sub_shifts_progression()
        self.print_overflow_underflow_stats()
        self.print_catastrophic_cancellation_stats()
        self.output_block_stats()

    def output_block_stats(self) -> None:
        """
        Outputs and plots all the statistics collected by analyzing
        the trace based on code blocks.
        """
        if self.block_size <= 0:
            return
        
        print("FP exception block analysis")
        num_blocks = len(self.block_exs)
        # Unzips the list of tuples
        overflows, underflows, cc = list(zip(*self.block_exs))
        # Checks the number of blocks that do not contain any exception
        ex_block_count = 0
        for i in range(num_blocks):
            total = overflows[i] + underflows[i]+ cc[i]
            ex_block_count += (total > 0)
        
        print(f"Number of blocks with FP exceptions: {ex_block_count} / {num_blocks} "
              f"({ex_block_count / num_blocks * 100:.2f}%)")


        # Visualizes the number of different types of exceptions in each block
        # and how they progress
        plt.plot(range(num_blocks), overflows, label="Overflow")
        plt.plot(range(num_blocks), underflows, label="Underflow")
        plt.plot(range(num_blocks), cc, label="Catastrophic cancellation")
        plt.xlabel("Block")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.savefig("block_ex_progression.png", format="png")
        plt.close()

    def plot_sub_shifts_dist(self) -> None:
        """
        Plots the distribution of bits shift in floating point subtraction.
        """        
        if self.total_sub_count > 0:
            plt.hist(self.pos_shifts)
            plt.xlabel("Number of bits shifted after subtraction")
            plt.ylabel("Count")

            plt.tight_layout()
            plt.savefig("fp_sub_shifts_dist.png", format="png")
            plt.close()

    def plot_sub_shifts_progression(self) -> None:
        """
        Visualizes how the number of bit-shifts changes for subtraction
        throughout the execution of the trace.
        """
        if self.total_sub_count > 0:
            pos_shifts = np.array(self.pos_shifts)
            pos_shifts = smooth(pos_shifts, 50)
            plt.plot(range(len(pos_shifts)), pos_shifts)
            plt.xlabel("Instruction")
            plt.ylabel("Number of bits shifted after subtraction")
            
            plt.tight_layout()
            plt.savefig("fp_sub_shifts_progression.png", format="png")
            plt.close()

    def plot_error_dist(self) -> None:
        """
        Plots the distributions of absolute and relative errors.
        """
        abs_errors = np.array(self.abs_errors)
        rel_errors = np.array(self.rel_errors) * 100

        assert len(abs_errors) == len(rel_errors)
        # TODO Refactor code
        # Plots absolute error distribution
        plt.hist(abs_errors, bins=100)
        plt.yscale("log")
        plt.xlabel("Absolute error")
        plt.ylabel("Count")

        plt.tight_layout()
        plt.savefig("abs_error_dist.png", format="png")
        plt.close()
        # Plots relative error distribution
        plt.hist(rel_errors, bins=100)
        plt.yscale("log")
        plt.xlabel("Relative error [%]")
        plt.ylabel("Count")

        plt.tight_layout()
        plt.savefig("rel_error_dist.png", format="png")
        plt.close()

    def plot_error_progression(self, twinx: bool = True) -> None:
        """
        Visualizes how the relative error and absolute error
        of the floating point operations change throughout
        the execution the trace.
        If `twinx` is set to True, will plot both the absolute error
        and the relative error progress in the same plot.
        """
        abs_errors = np.array(self.abs_errors)
        rel_errors = np.array(self.rel_errors) * 100
        abs_errors = smooth(abs_errors, 50)
        rel_errors = smooth(rel_errors, 50)
        assert len(abs_errors) == len(rel_errors)

        if twinx:
            # Absolute error
            fig, ax1 = plt.subplots()
            color = "tab:red"
            ax1.plot(range(len(abs_errors)), abs_errors, color=color)
            ax1.set_xlabel("Instruction")
            ax1.set_ylabel("Absolute error", color=color)
            ax1.tick_params(axis='y', labelcolor=color)

            # Relative error
            ax2 = ax1.twinx()
            color = "tab:blue"
            ax2.plot(range(len(rel_errors)), rel_errors)
            ax2.set_ylabel("Relative error [%]", color=color)
            ax2.tick_params(axis='y', labelcolor=color)

            fig.tight_layout()
            fig.savefig("error_progression.png", format="png")
            plt.close()

        else:
            plt.plot(range(len(abs_errors)), abs_errors)
            plt.xlabel("Instruction")
            plt.ylabel("Absolute error")

            plt.tight_layout()
            plt.savefig("abs_error_progression.png", format="png")
            plt.close()

            plt.plot(range(len(rel_errors)), rel_errors)
            plt.xlabel("Instruction")
            plt.ylabel("Relative error [%]")

            plt.tight_layout()
            plt.savefig("rel_error_progression.png", format="png")
            plt.close()

    def plot_input_dist(self) -> None:
        """
        Plots the input value distribution.
        """
        inputs = np.array(self.input_vals)
        # Plots only the valid values
        valid_vals = inputs[np.isfinite(inputs)]
        plt.hist(valid_vals)
        plt.xlabel("Input value")
        plt.ylabel("Count")
        # Reports the percentage of NaN and inf
        nan_count = np.isnan(inputs).sum()
        inf_count = np.isinf(inputs).sum()
        input_size = len(inputs)
        print(f"Total number of input values: {input_size}")
        print(f"Percentage of 'NaN' in input: {nan_count / input_size * 100:.2f}%")
        print(f"Percentage of 'inf' in input: {inf_count / input_size * 100:.2f}%")

        plt.tight_layout()
        plt.savefig("input_dist.png", format="png")
        plt.close()

    def print_catastrophic_cancellation_stats(self) -> None:
        """
        Displays statistics about catastrophic cancellation (i.e.,
        number of bits shifted after floating point subtraction.)
        """
        print("Catastrophic Cancellation statistics")
        if self.total_sub_count > 0:
            pos_shifts = np.array(self.pos_shifts)
            mean = np.mean(pos_shifts)
            std = np.std(pos_shifts)
            print(f"Catastrophic cancellation: {self.cc_count}/{self.total_sub_count}"
                f" ({self.cc_count / self.total_sub_count * 100:.2f}%)")
            print(f"Average number of bits shifted: {mean}")
            print(f"Standard deviation of number of bits shifted: {std}")
        else:
            print("No subtraction was performed...")


    def print_overflow_underflow_stats(self) -> None:
        """
        Displays statistics about overflow and underflow.
        """
        print("Overflow/Underflow statistics")
        print(f"Overflow : {self.overflow_count}/{self.total_count}"
              f" ({self.overflow_count / self.total_count * 100:.2f}%)")
        print(f"Underflow: {self.underflow_count}/{self.total_count}"
              f" ({self.underflow_count / self.total_count * 100:.2f}%)")