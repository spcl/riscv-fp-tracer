from __future__ import annotations
from typing import List, Optional, Union
from enum import Enum, auto
from utils import fp16_to_hex
import numpy as np


class ArithOp(Enum):
    ADD = auto()
    SUB = auto()
    DIV = auto()
    MUL = auto()
    MAX = auto()
    MIN = auto()

    # Unary operators
    ABS = auto()
    SQRT = auto()
    NEG = auto()

    @staticmethod
    def perform_unary_op(op: ArithOp, val: np.float16) -> np.float16:
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

    @staticmethod
    def perform_op(op: ArithOp, val1: np.float16, val2: np.float16) \
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

    @staticmethod
    def perform_ops(op1: ArithOp, op2: ArithOp,
                    val1: np.float16, val2: np.float16, val3: np.float16) \
        -> np.float16:
        """
        Performs two given arithmetic operations on three values.
        """
        tmp = ArithOp.perform_op(op1, val1, val2)
        res = ArithOp.perform_op(op2, tmp, val3)
        return res
    

class OpCode(Enum):
    FCVT = "fcvt"       # [x]
    FMV = "fmv"         # [x]
    FDIV = "fdiv"       # [x]
    FSUB = "fsub"       # [x]
    FADD = "fadd"       # [x]
    FABS = "fabs"       # [x]
    FMAX = "fmax"       # [x]
    FMIN = "fmin"       # [x]
    FMUL = "fmul"       # [x]
    FMADD = "fmadd"     # [x]
    FMSUB = "fmsub"     # [x]
    FNMADD = "fnmadd"   # [x]
    FNMSUB = "fnmsub"   # [x]
    FSQRT = "fsqrt"     # [x]
    FNEG = "fneg"       # [x]
    FEQ = "feq"         # [x]
    FLE = "fle"         # [x]
    FLT = "flt"         # [x]
    FLW = "flw"         # [x]
    FLD = "fld"         # [x]
    FSW = "fsw"         # [x]
    FSD = "fsd"         # [x]
    FSNGJ = "fsgnj"     # [x]


class Instruction(object):
    """
    An object that stores all the information related to a
    single instruction.
    """
    def __init__(self, opcode: OpCode, operands: List[str],
                 reg_vals: List[Union[np.float16, str]],
                 addr: Optional[str] = None) -> None:
        self.opcode = opcode
        self.operands = operands
        self.reg_vals = reg_vals
        self.addr = addr
    
    def __str__(self) -> str:
        regs = [f"{reg}({val})" for reg, val in zip(self.operands, self.reg_vals)]
        return f"{self.opcode.value} {','.join(regs)}"
    

    def output(self, end: str = "\n") -> str:
        """
        Returns the instruction as a string that conforms with the
        trace file format.
        <opcode> [ <fp_reg>]+ [ <reg_val_hex>]+
        """
        assert isinstance(self.reg_vals[0], np.float16)
        # reg_vals = list(map(fp16_to_hex, self.reg_vals))
        reg_vals = list(map(str, self.reg_vals))
        res = f"{self.opcode.value} {' '.join(self.operands)} "
        res += f"{' '.join(reg_vals)}{end}"
        return res