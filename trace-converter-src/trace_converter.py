import os
from tqdm import tqdm
from typing import List, Optional
from parser import Parser
from fp16_emulator import HalfPrecisionEmulator

class TraceConverter(object):
    """
    Converts all instructions in the given floating point trace in FP64
    to FP16. As an example, if we have instruction
    'fadd.d fa5 fa3 fa5;40091EB851EB851F 40091EB851EB851F 40191EB851EB851F'
    It will be converted to
    'fadd fa5 fa3 fa5;4248 4248 4648'
    """
    def __init__(self, trace_file: str, output_file: str) -> None:
        # Makes sure that the given file exists
        assert(os.path.exists(trace_file))
        self.trace_file = trace_file
        self.output_file = output_file

    def convert(self, flush_freq: int = 10000) -> None:
        """
        Converts the instructions in the given trace file to FP16.
        @param flush_freq: An integer that indicates how often the converted
        instructions are flushed to the output file. For instance,
        `flush` = 100 indicates that the output buffer will be flushed
        after converting 100 instructions.
        """
        out_buf = ""
        trace = open(self.trace_file, "r")
        output = open(self.output_file, "w")
        count = 0
        # Initializes the emulator
        emulator = HalfPrecisionEmulator()

        # Iterates through every line in the trace file
        for line in trace:
            if count % flush_freq == 0 and count > 0:
                # Flushes the output buffer according to `flush`
                output.write(out_buf)
                count = 0
            
            insn = Parser.parse(line)
            emulator.execute(insn)
            out_buf += emulator.get_last_insn().output()
            print(emulator.get_last_insn())

        output.write(out_buf)
        trace.close()
        output.close()