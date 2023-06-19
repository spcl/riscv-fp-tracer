from typing import List, Optional, Dict
from instruction import Instruction, OpCode
from utils import hex64_to_fp16

class TraceParser(object):
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
            insn_tokens = insn.split('.')
            opcode = OpCode(insn_tokens[0])
            if opcode == OpCode.FCVT or opcode == OpCode.FMV:
                # Distinguishes the specific type of
                # conversion instruction
                target = insn_tokens[1]
                if target == 'd' or target == 's':
                    opcode = OpCode(opcode.value + '.' + target)
            params["opcode"] = opcode
        except ValueError:
            raise ValueError(f"[ERROR] Invalid opcode: '{opcode}'")
        params["operands"] = operands
        reg_vals = reg_val_str.split()
        params["reg_vals"] = reg_vals
        insn = Instruction(**params)
        return insn
