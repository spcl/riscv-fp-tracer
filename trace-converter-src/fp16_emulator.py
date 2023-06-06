import numpy as np
from collections import defaultdict
from typing import List, Optional, Dict, Callable, Tuple
from functools import partial

from utils import hex64_to_fp16
from instruction import Instruction, OpCode, ArithOp

class HalfPrecisionEmulator(object):
    """
    A simplified emulator that executes floating point instructions
    as if all the floating point numbers are in FP16. Note that
    it only focuses on floating point registers and data access to
    memory addresses.
    """
    def __init__(self) -> None:
        """
        Initializes the internal state of the emulator
        """
        self.fp_regs: Dict[str, np.float16] = defaultdict(np.float16)
        self.mem: Dict[str, np.float16] = defaultdict(np.float16)
        self.last_insn = None

        # A dictionary that maps instruction opcodes to corresponding
        # actions that need to be performed
        self.insn_exec_fn: Dict[OpCode, Callable] = {
            OpCode.FCVT: self.__mv,
            OpCode.FMV: self.__mv,
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
                                  op1=ArithOp.MUL,op2=ArithOp.SUB),
            OpCode.FNMADD: partial(self.__arith_fd_fs1_fs2_fs3,
                                   op1=ArithOp.MUL, op2=ArithOp.SUB, neg=True),
            OpCode.FNMSUB: partial(self.__arith_fd_fs1_fs2_fs3,
                                   op1=ArithOp.MUL, op2=ArithOp.SUB, neg=True),
            OpCode.FEQ: self.__comp_fs1_fs2,
            OpCode.FLE: self.__comp_fs1_fs2,
            OpCode.FLT: self.__comp_fs1_fs2
        }

    def __get_reg_val(self, reg: str, trace_val: str) -> np.float16:
        """
        Returns the value in the given register as a np.float16.
        This function first checks if an value has already 
        been assigned to the register. If not, it converts the FP64
        value from the trace to FP16 and store the converted value
        to `fp_regs`.
        """
        if reg in self.fp_regs:
            return self.fp_regs[reg]
        else:
            # Converts HEX string to FP16
            val = hex64_to_fp16(trace_val)
            self.fp_regs[reg] = val
            return val

    def __comp_fs1_fs2(self, insn: Instruction) -> None:
        """
        Emulates a comparison instruction such as 'feq'.
        """
        reg_vals = insn.reg_vals
        operands = insn.operands
        assert len(reg_vals) == len(operands) == 2
        fs1, fs2 = operands
        fs1_val = self.__get_reg_val(fs1, reg_vals[0])
        fs2_val = self.__get_reg_val(fs2, reg_vals[1])

        self.reg_vals = [fs1_val, fs2_val]

    def __mv(self, insn: Instruction) -> None:
        """
        Emulates a move instruction by setting the register value to
        that specified by `reg_vals`.
        """
        reg_vals = insn.reg_vals
        operands = insn.operands
        assert len(reg_vals) == len(operands) == 1
        # Converts hex string to np.float64
        reg = operands[0]
        reg_val = reg_vals[0]
        reg_val_fp16 = hex64_to_fp16(reg_val)
        self.fp_regs[reg] = reg_val_fp16
        # Sets the register values to be in FP16
        insn.reg_vals = [reg_val_fp16]

    def __ld(self, insn: Instruction) -> None:
        """
        Emulates a load instruction.
        """
        reg_vals = insn.reg_vals
        operands = insn.operands
        assert len(reg_vals) == len(operands) == 1
        assert insn.addr is not None
        addr = insn.addr
        # Checks if there is already data at the memory address
        if addr in self.mem:
            val = self.mem[addr]
        else:
            val = hex64_to_fp16(reg_vals[0])
        
        self.fp_regs[operands[0]] = val
        insn.reg_vals = [val]

    def __st(self, insn: Instruction) -> None:
        """
        Emulates a store instruction by setting the appropriate
        memory address to the value stored in the register.
        """
        reg_vals = insn.reg_vals
        operands = insn.operands
        assert len(reg_vals) == len(operands) == 1
        assert insn.addr is not None
        reg = operands[0]
        # Checks if the register is empty
        if reg in self.fp_regs:
            reg_val = self.fp_regs[reg]
        else:
            reg_val = hex64_to_fp16(reg_vals[0])
        # Sets the value in the target memory address
        self.mem[insn.addr] = reg_val
        insn.reg_vals = [reg_val]

    def __arith_fd_fs1(self, insn: Instruction, op: ArithOp) -> None:
        """
        Emulates an arithmetic operation that involves two registers
        (unary operations) given the instruction as well as the operation
        that needs to performed.
        f[fd] = <op> f[fs1]
        """
        reg_vals = insn.reg_vals
        operands = insn.operands
        assert len(reg_vals) == len(operands) == 1
        fd, fs1 = operands
        fs1_val = self.__get_reg_val(fs1, reg_vals[0])
        
        fd_val = ArithOp.perform_unary_op(op, fs1_val)
        self.fp_regs[fd] = fd_val
        insn.reg_vals = [fd_val, fs1_val]

    def __arith_fd_fs1_fs2(self, insn: Instruction, op: ArithOp) -> None:
        """
        Emulates an arithmetic operation involving three registers (binary
        operations) given the instruction as well as the operation that needs
        to be performed.
        f[fd] = f[fs1] <op> f[fs2]
        """
        reg_vals = insn.reg_vals
        operands = insn.operands
        assert len(reg_vals) == len(operands) == 3
        fd, fs1, fs2 = operands
        fs1_val = self.__get_reg_val(fs1, reg_vals[0])
        fs2_val = self.__get_reg_val(fs2, reg_vals[1])

        fd_val = ArithOp.perform_op(op, fs1_val, fs2_val)
        self.fp_regs[fd] = fd_val
        insn.reg_vals = [fd_val, fs1_val, fs2_val]
    
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
        assert len(reg_vals) == len(operands) == 4
        fd, fs1, fs2, fs3 = operands
        fs1_val = self.__get_reg_val(fs1, reg_vals[0])
        fs2_val = self.__get_reg_val(fs2, reg_vals[1])
        fs3_val = self.__get_reg_val(fs3, reg_vals[2])

        fd_val = ArithOp.perform_ops(op1, op2, fs1_val, fs2_val, fs3_val)
        if neg:
            fd_val = -fd_val
        
        self.fp_regs[fd] = fd_val
        insn.reg_vals = [fd_val, fs1_val, fs2_val, fs3_val]

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

    def get_last_insn(self) -> Instruction:
        """
        Returns the last instruction that has been executed.
        """
        assert self.last_insn is not None
        return self.last_insn