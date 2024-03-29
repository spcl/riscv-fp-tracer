import numpy as np
from collections import defaultdict
from typing import List, Optional, Dict, Callable, Tuple
from functools import partial

from utils import hex64_to_fp16
from fp_stats_collector import FpStatsCollector
from instruction import Instruction, OpCode, ArithOp


class HalfPrecisionEmulator(object):
    """
    A simplified emulator that executes floating point instructions
    as if all the floating point numbers are in FP16. Note that
    it only focuses on floating point registers and data access to
    memory addresses.
    """
    def __init__(self, collector: Optional[FpStatsCollector] = None) -> None:
        """
        Initializes the internal state of the emulator
        """
        self.fp_regs: Dict[str, np.float16] = defaultdict(np.float16)
        self.mem: Dict[str, np.float16] = defaultdict(np.float16)
        self.last_insn = None
        self.collector = collector

        # A dictionary that maps instruction opcodes to corresponding
        # actions that need to be performed
        self.insn_exec_fn: Dict[OpCode, Callable] = {
            OpCode.FCVT: self.__mv,
            OpCode.FCVTD: self.__mv_to_fp,
            OpCode.FCVTS: self.__mv_to_fp,
            OpCode.FMV: self.__mv,
            OpCode.FMVD: self.__mv_to_fp,
            OpCode.FMVS: self.__mv_to_fp,
            OpCode.FSW: self.__st,
            OpCode.FSD: self.__st,
            OpCode.FLW: self.__ld,
            OpCode.FLD: self.__ld,
            OpCode.FABS: partial(self.__arith_fd_fs1, op=ArithOp.ABS),
            OpCode.FSQRT: partial(self.__arith_fd_fs1, op=ArithOp.SQRT),
            OpCode.FNEG: partial(self.__arith_fd_fs1, op=ArithOp.NEG),
            OpCode.FADD: partial(self.__arith_fd_fs1_fs2, op=ArithOp.ADD),
            OpCode.FSUB: partial(self.__arith_fd_fs1_fs2, op=ArithOp.SUB),
            OpCode.FDIV: partial(self.__arith_fd_fs1_fs2, op=ArithOp.DIV),
            OpCode.FMUL: partial(self.__arith_fd_fs1_fs2, op=ArithOp.MUL),
            OpCode.FMAX: partial(self.__arith_fd_fs1_fs2, op=ArithOp.MAX),
            OpCode.FMIN: partial(self.__arith_fd_fs1_fs2, op=ArithOp.MIN),
            OpCode.FMADD: partial(self.__arith_fd_fs1_fs2_fs3,
                                  op1=ArithOp.MUL, op2=ArithOp.ADD),
            OpCode.FMSUB: partial(self.__arith_fd_fs1_fs2_fs3,
                                  op1=ArithOp.MUL, op2=ArithOp.SUB),
            OpCode.FNMADD: partial(self.__arith_fd_fs1_fs2_fs3,
                                   op1=ArithOp.MUL, op2=ArithOp.SUB, neg=True),
            OpCode.FNMSUB: partial(self.__arith_fd_fs1_fs2_fs3,
                                   op1=ArithOp.MUL, op2=ArithOp.SUB, neg=True),
            OpCode.FEQ: self.__comp_fs1_fs2,
            OpCode.FLE: self.__comp_fs1_fs2,
            OpCode.FLT: self.__comp_fs1_fs2,
            OpCode.FSGNJ: self.__fsgnj,
            OpCode.FSGNJN: partial(self.__fsgnj, neg=True),
            OpCode.SW: self.__i_st,
            OpCode.SD: self.__i_st,
        }

    # ===================== Arithmetic operations ======================
    def perform_unary_op(self, op: ArithOp, val: np.float16) -> np.float16:
        """
        Performs a unary operation as per the given operator
        and value.
        """
        if op == ArithOp.ABS:
            return abs(val)
        if op == ArithOp.SQRT:
            return np.sqrt(val)
        if op == ArithOp.NEG:
            return -val

        raise ValueError(f"[ERROR] '{op.name}' is not a unary operator.")

    def perform_op(self, op: ArithOp, val1: np.float16, val2: np.float16) \
        -> np.float16:
        """
        Performs the given arithmetic operation on the values and
        returns the result as np.float16.
        """
        if op == ArithOp.ADD:
            return val1 + val2
        if op == ArithOp.SUB:
            return val1 - val2
        if op == ArithOp.DIV:
            return val1 / val2
        if op == ArithOp.MUL:
            return val1 * val2
        if op == ArithOp.MAX:
            return max(val1, val2)
        if op == ArithOp.MIN:
            return min(val1, val2)
        raise ValueError(f"[ERROR] '{op.name}' not supported")

    def perform_ops(self, op1: ArithOp, op2: ArithOp,
                    val1: np.float16, val2: np.float16, val3: np.float16) \
        -> np.float16:
        """
        Performs two given arithmetic operations on three values.
        """
        tmp = self.perform_op(op1, val1, val2)
        res = self.perform_op(op2, tmp, val3)
        return res


    def print_register_state(self, row_len: int = 4) -> None:
        """
        Prints out the value of each FP16 register.
        """
        print("================ Registers ================")
        count = 0
        for reg, val in sorted(self.fp_regs.items()):
            print(f"{reg}: {val}", end='\t')
            count += 1
            if count % row_len == 0:
                print()
        if count % row_len != 0:
            print()


    def __get_reg_val(self, reg: str, trace_val: str, is_double: bool = True) \
        -> np.float16:
        """
        Returns the value in the given register as a np.float16.
        This function first checks if an value has already 
        been assigned to the register. If not, it converts the FP64
        value from the trace to FP16 and stores the converted value
        to `fp_regs`.
        
        If `is_double` is True, will treat the hex strings in the trace as
        64-bit double precision floating point values, otherwise, it will
        treat them as 32-bit single-precision floating point values.
        """
        if reg in self.fp_regs:
            return self.fp_regs[reg]

        # Converts HEX string to FP16
        val = hex64_to_fp16(trace_val, is_double)
        self.fp_regs[reg] = val
        self.collector.add_input_val(val)
        # self.collector.count_overflow_underflow(val, trace_val, is_double,
        #                                         dst=reg, is_insn=False)
        return val

    def __i_st(self, insn: Instruction) -> None:
        """
        Emulates an integer store operation. Simply parses the
        value in the trace and overwrites the value in the memory
        address specified instruction.
        """
        reg_vals = insn.reg_vals
        is_double = insn.is_double
        assert len(reg_vals) == 1
        reg_val = hex64_to_fp16(reg_vals[0], is_double)
        self.mem[insn.addr] = reg_val
        insn.reg_vals = [reg_val]
        self.collector.reset_ex_map_entry(insn.addr)

    def __comp_fs1_fs2(self, insn: Instruction) -> None:
        """
        Emulates a comparison instruction such as 'feq'. Since it
        modifies an integer register, this instruction does not change
        the state of the emulator at all.
        """
        reg_vals = insn.reg_vals
        operands = insn.operands
        is_double = insn.is_double
        assert len(reg_vals) == len(operands) == 2
        fs1, fs2 = operands
        fs1_val = self.__get_reg_val(fs1, reg_vals[0], is_double)
        fs2_val = self.__get_reg_val(fs2, reg_vals[1], is_double)

        insn.reg_vals = [fs1_val, fs2_val]

    def __mv(self, insn: Instruction) -> None:
        """
        Emulates a move instruction that has a non-floating-point
        register as the destination (e.g., fcvt.l.s, fmv.x.d etc.).
        """
        reg_vals = insn.reg_vals
        operands = insn.operands
        assert (len(reg_vals) == len(operands) == 1)
        reg = operands[0]
        reg_val = self.__get_reg_val(reg, reg_vals[0], insn.is_double)
        insn.reg_vals = [reg_val]
        self.collector.reset_ex_map_entry(reg)

    def __mv_to_fp(self, insn: Instruction) -> None:
        """
        Emulates a move instruction that has a floating point
        register as the destination (e.g., fmv.d, fcvt.s.l, etc.).
        """
        reg_vals = insn.reg_vals
        operands = insn.operands
        is_double = insn.is_double
        assert (len(reg_vals) == len(operands) == 1) or \
            (len(reg_vals) == len(operands) == 2)
        
        if len(reg_vals) == 1:
            reg = operands[0]
            reg_val = hex64_to_fp16(reg_vals[0], is_double)
            self.fp_regs[reg] = reg_val
            insn.reg_vals = [reg_val]
            # Collects statistics
            self.collector.add_input_val(reg_val)
            self.collector.reset_ex_map_entry(reg)
        else:
            # len(operands) == 2
            # Moves FP value from one FP register to the other
            fd, fs = operands
            fs_val = self.__get_reg_val(fs, reg_vals[1], is_double)
            self.fp_regs[fd] = fs_val
            insn.reg_vals = [fs_val, fs_val]
            # Collects statistics
            self.collector.update_ex_map(fd, [fs])

    def __ld(self, insn: Instruction) -> None:
        """
        Emulates a load instruction.
        """
        reg_vals = insn.reg_vals
        operands = insn.operands
        is_double = insn.is_double
        assert len(reg_vals) == len(operands) == 1
        assert insn.addr is not None
        addr = insn.addr
        # Checks if there is already data at the memory address
        if addr in self.mem:
            val = self.mem[addr]
        else:
            # If the value does not exist in memory
            val = hex64_to_fp16(reg_vals[0], is_double)
            self.collector.add_input_val(val)
            self.mem[addr] = val
        
        fd = operands[0]
        self.fp_regs[fd] = val
        insn.reg_vals = [val]
        self.collector.update_ex_map(fd, [addr])

    def __st(self, insn: Instruction) -> None:
        """
        Emulates a store instruction by setting the appropriate
        memory address to the value stored in the register.
        """
        reg_vals = insn.reg_vals
        operands = insn.operands
        is_double = insn.is_double
        assert len(reg_vals) == len(operands) == 1
        assert insn.addr is not None
        reg = operands[0]
        addr = insn.addr
        # Checks if the register is empty
        reg_val = self.__get_reg_val(reg, reg_vals[0], is_double)
        # Sets the value in the target memory address
        self.mem[addr] = reg_val
        insn.reg_vals = [reg_val]

        self.collector.update_ex_map(addr, [reg])

    def __fsgnj(self, insn: Instruction, neg: bool = False) -> None:
        """
        Emulates the floating point sign injection instruction,
        i.e., fsgnj.
        f[fd] = f[fs1][14:0] | (<neg>(f[fs2][15]) << 15)
        """
        reg_vals = insn.reg_vals
        operands = insn.operands
        is_double = insn.is_double
        assert len(reg_vals) == len(operands) == 3

        fd, fs1, fs2 = operands
        fs1_val = self.__get_reg_val(fs1, reg_vals[1], is_double)
        fs2_val = self.__get_reg_val(fs2, reg_vals[2], is_double)
        sign = np.sign(fs2_val)
        if neg:
            sign = -sign
        fd_val = sign * fs1_val

        self.fp_regs[fd] = fd_val
        insn.reg_vals = [fd_val, fs1_val, fs2_val]
        self.collector.update_ex_map(fd, [fs1, fs2])

    def __arith_fd_fs1(self, insn: Instruction, op: ArithOp) -> None:
        """
        Emulates an arithmetic operation that involves two registers
        (unary operations) given the instruction as well as the operation
        that needs to performed.
        f[fd] = <op> f[fs1]
        """
        reg_vals = insn.reg_vals
        operands = insn.operands
        is_double = insn.is_double
        assert len(reg_vals) == len(operands) == 2
        fd, fs1 = operands
        fs1_val = self.__get_reg_val(fs1, reg_vals[1], is_double)
        
        fd_val = self.perform_unary_op(op, fs1_val)
        self.fp_regs[fd] = fd_val
        insn.reg_vals = [fd_val, fs1_val]

        is_ex = self.collector.count_overflow_underflow(fd_val, reg_vals[0],
                                                        is_double,
                                                        dst=fd, src=[fs1])
        self.collector.ex_map[fd] = is_ex

        self.collector.add_error(fd_val, reg_vals[0], is_double)

    def __arith_fd_fs1_fs2(self, insn: Instruction, op: ArithOp) -> None:
        """
        Emulates an arithmetic operation involving three registers (binary
        operations) given the instruction as well as the operation that needs
        to be performed.
        f[fd] = f[fs1] <op> f[fs2]
        """
        reg_vals = insn.reg_vals
        operands = insn.operands
        is_double = insn.is_double
        assert len(reg_vals) == len(operands) == 3
        fd, fs1, fs2 = operands
        fs1_val = self.__get_reg_val(fs1, reg_vals[1], is_double)
        fs2_val = self.__get_reg_val(fs2, reg_vals[2], is_double)

        fd_val = self.perform_op(op, fs1_val, fs2_val)
        self.fp_regs[fd] = fd_val
        insn.reg_vals = [fd_val, fs1_val, fs2_val]

        is_ex = False
        if op == ArithOp.SUB:
            is_ex = is_ex or \
                self.collector.count_catastrophic_cancellation(fs1_val, fs2_val,
                                                               fd, [fs1, fs2])
        
        is_ex = self.collector.count_overflow_underflow(fd_val, reg_vals[0],
                                                        is_double,
                                                        dst=fd, src=[fs1, fs2]) \
                                                        or is_ex
        self.collector.ex_map[fd] = is_ex
        self.collector.add_error(fd_val, reg_vals[0], is_double)
    
    def __arith_fd_fs1_fs2_fs3(self, insn: Instruction,
                               op1: ArithOp, op2: ArithOp, neg: bool = False) \
                                -> None:
        """
        Emulates an arithmetic operation that involves four registers
        given the instruction as well as the 2 operations that
        need to performed. If `neg` is True, the result will be negated
        f[fd] = <neg>(f[fs1] <op1> f[fs2]) <op2> f[fs3]
        """
        reg_vals = insn.reg_vals
        operands = insn.operands
        is_double = insn.is_double
        assert len(reg_vals) == len(operands) == 4
        fd, fs1, fs2, fs3 = operands
        fs1_val = self.__get_reg_val(fs1, reg_vals[1], is_double)
        fs2_val = self.__get_reg_val(fs2, reg_vals[2], is_double)
        fs3_val = self.__get_reg_val(fs3, reg_vals[3], is_double)

        # Performs two arithmetic operations
        # fd_val = self.perform_ops(op1, op2, fs1_val, fs2_val, fs3_val)
        tmp = self.perform_op(op1, fs1_val, fs2_val)
        fd_val = self.perform_op(op2, tmp, fs3_val)

        is_ex = False
        if op2 ==  ArithOp.SUB:
            is_ex = is_ex or \
                self.collector.count_catastrophic_cancellation(tmp, fs3_val,
                                                               fd, src=[fs1, fs2, fs3])

        if neg:
            fd_val = -fd_val
        
        self.fp_regs[fd] = fd_val
        insn.reg_vals = [fd_val, fs1_val, fs2_val, fs3_val]
        is_ex = self.collector.count_overflow_underflow(fd_val, reg_vals[0],
                                                        is_double, dst=fd,
                                                        src=[fs1, fs2, fs3]) \
                                                        or is_ex
        self.collector.ex_map[fd] = is_ex
        self.collector.add_error(fd_val, reg_vals[0], is_double)

    def execute(self, insn: Instruction) -> None:
        """
        Executes the given instruction in the emulator by changing
        its internal state accordingly.
        """
        fn = self.insn_exec_fn.get(insn.opcode)
        if fn is None:
            raise ValueError(f"[ERROR] Opcode '{insn.opcode.value}' "
                             "not supported in the emulator")
        fn(insn)
        self.last_insn = insn
        if insn.is_arith_insn:
            self.collector.count_block_stats()

    def get_last_insn(self) -> Instruction:
        """
        Returns the last instruction that has been executed.
        """
        assert self.last_insn is not None
        return self.last_insn
    
    def finalize(self) -> None:
        """
        Performs a series of operations after the program has
        finished executing to finalize the output and statistics
        collection.
        """
        self.collector.count_block_stats(finalize=True)
