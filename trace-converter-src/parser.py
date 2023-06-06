from typing import List, Optional, Dict
from instruction import Instruction, OpCode
from utils import hex_to_fp16

class Parser(object):
    """
    A class representation of the RISC-V instruction parser.
    """
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def parse(insn: str) -> Instruction:
        """
        Parses the given instruction by splitting into an
        Instruction object.
        """
        params = {}
        tokens = insn.split(";")
        if len(tokens) == 2:
            # If no memory address is present
            insn_str = tokens[0]
            reg_val_str = tokens[1]
        elif len(tokens) == 3:
            # If there is memory address
            insn_str = tokens[0]
            params["addr"] = tokens[1]
            reg_val_str = tokens[2]
        else:
            raise ValueError(f"[ERROR] Invalid format: {insn}")

        insn, *operands = insn_str.split()
        try:
            opcode = insn.split('.')[0]
            params["opcode"] = OpCode(opcode)
        except ValueError:
            raise ValueError(f"[ERROR] Invalid opcode: '{opcode}'")
        params["operands"] = operands
        reg_vals = reg_val_str.split()
        params["reg_vals"] = reg_vals
        insn = Instruction(**params)
        return insn
