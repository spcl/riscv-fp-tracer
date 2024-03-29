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
    

class OpCode(Enum):
    # Move
    FCVT = "fcvt"       # [x]
    FCVTD = "fcvt.d"    # [x]
    FCVTS = "fcvt.s"    # [x]
    FMV = "fmv"         # [x]
    FMVD = "fmv.d"      # [x]
    FMVS = "fmv.s"      # [x]
    # Arithmetic
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
    # Comparison
    FEQ = "feq"         # [x]
    FLE = "fle"         # [x]
    FLT = "flt"         # [x]
    # Memory Access
    FLW = "flw"         # [x]
    FLD = "fld"         # [x]
    FSW = "fsw"         # [x]
    FSD = "fsd"         # [x]
    # Sign injection
    FSGNJ = "fsgnj"     # [x]
    FSGNJN = "fsgnjn"   # [x]
    # FIXME Sometimes the compiler makes the
    # dependent FP operations go through these integer stores
    # e.g. LULESH2.0
    SW = "sw"           # [x]
    SD = "sd"           # [x]


class Instruction(object):
    """
    An object that stores all the information related to a
    single instruction.
    """
    # Arithmetic instructions
    ARITH_INSN = frozenset([OpCode.FDIV, OpCode.FSUB, OpCode.FADD, 
                            OpCode.FABS, OpCode.FMAX, OpCode.FMIN,
                            OpCode.FMUL, OpCode.FMADD, OpCode.FMSUB,
                            OpCode.FNMADD, OpCode.FNMSUB,
                            OpCode.FSQRT, OpCode.FNEG])
    NON_FP_INSN = frozenset([OpCode.SW, OpCode.SD])
    def __init__(self, id: int, opcode: OpCode, operands: List[str],
                 reg_vals: List[Union[np.float16, str]],
                 addr: Optional[str] = None,
                 is_double: bool = True) -> None:
        """
        `is_double` indicates whether the instruction was
        performed on double-precision FP numbers.
        """
        self.id = id
        self.opcode = opcode
        self.operands = operands
        self.reg_vals = reg_vals
        self.addr = addr
        self.is_double = is_double

        # `is_fp_insn` specifies whether the instruction is a floating
        # point operation (i.e., 'sd' and 'sw' do not count).
        self.is_fp_insn = not opcode in Instruction.NON_FP_INSN
        # `is_arith_insn` specifies whether the instruction
        # is an arithmetic instruction
        self.is_arith_insn = opcode in Instruction.ARITH_INSN
    
    def __str__(self) -> str:
        regs = [f"{reg}({val})" for reg, val in zip(self.operands, self.reg_vals)]
        return f"{self.id}: {self.opcode.value} {','.join(regs)}"

    def output(self, end: str = "\n") -> str:
        """
        Returns the instruction as a string that conforms with the
        trace file format.
        <opcode>[ <fp_reg>]+ <addr>[ <reg_val_hex>]+
        """
        assert isinstance(self.reg_vals[0], np.float16)
        # reg_vals = list(map(fp16_to_hex, self.reg_vals))
        reg_vals = list(map(str, self.reg_vals))
        operands = f" {' '.join(self.operands)}" if len(self.operands) > 0 else ''
        addr = f" {self.addr}" if self.addr else ''

        res = f"{self.opcode.value}{operands}{addr} "
        res += f"{' '.join(reg_vals)}{end}"
        return res