import os
from time import time
from tqdm import tqdm
from typing import List, Optional
from trace_parser import TraceParser
from fp16_emulator import HalfPrecisionEmulator
from fp_stats_collector import FpStatsCollector
from mem_trace_analyzer import MemoryTraceAnalyzer

class TraceConverter(object):
    """
    Converts all instructions in the given floating point trace in FP64
    to FP16.
    
    As an example, if we have instruction
    'fadd.d fa5 fa3 fa5;40091EB851EB851F 40091EB851EB851F 40191EB851EB851F'.

    It will be converted to
    'fadd fa5 fa3 fa5;4248 4248 4648'
    """
    def __init__(self, trace_file: str, output_file: str,
                 collector: Optional[FpStatsCollector] = None,
                 mem_analyzer: Optional[MemoryTraceAnalyzer] = None,
                 num_arith_insn: Optional[int] = None,
                 debug: bool = False) -> None:
        """
        @param trace_file: The FP64 trace file to be parsed and converted.
        @param output_file: The path to the file that stores the FP16
        trace after the conversion.
        @param collector: An floating point statistics collector that will
        keep track of all the relevant statistics.
        @param mem_analyzer: Memory trace analyzer. This is an experimental
        feature that is unrelated to floating point analysis.
        @param num_arith_insn: If given, the program will exit after
        executing `num_arith_insn` number of FLOATING POINT
        ARITHMETIC instructions.
        @param debug: If True, will print more information to stdout.
        """
        # Makes sure that the given file exists
        assert(os.path.exists(trace_file))
        self.trace_file = trace_file
        self.output_file = output_file
        self.collector = collector
        self.mem_analyzer = mem_analyzer
        # Specifies the number of FP arithmetic instructions to execute
        self.num_arith_insn = \
            num_arith_insn if num_arith_insn is not None else -1
        self.debug = debug

    def convert(self, flush_freq: int = 1000000) -> None:
        """
        Converts the instructions in the given trace file to FP16.
        @param flush_freq: An integer that indicates how often the converted
        instructions are flushed to the output file. For instance,
        `flush` = 100 indicates that the output buffer will be flushed
        after converting 100 instructions.
        """
        out_buf = ""
        trace = open(self.trace_file, "r")
        output = open(self.output_file, "w+")
        count = 0
        # Initializes the emulator
        emulator = HalfPrecisionEmulator(self.collector)

        prev_time = time()
        arith_insn_count = 0
        # Iterates through every line in the trace file
        for line in trace:
            if count % flush_freq == 0 and count > 0:
                curr_time = time()
                print(f"[INFO] Progress [{count}]: {int(flush_freq / (curr_time - prev_time))} iter/s")
                # Flushes the output buffer according to `flush`
                output.write(out_buf)
                out_buf = ""
                prev_time = curr_time
            insn = TraceParser.parse(line, count)
            
            # Experimental feature: memory trace analysis
            # if insn.is_fp_insn:
            #     self.mem_analyzer.add_mem_addr(insn.addr)

            # if insn.is_arith_insn:
            #     arith_insn_count += 1
            #     if self.debug:
            #         print(f"[DEBUG] Insn {count + 1}: {last_insn[:-1]}")
            
            emulator.execute(insn)
            last_insn = emulator.get_last_insn().output()
            # print(emulator.get_last_insn())
            out_buf += last_insn
            count += 1

            if insn.is_fp_insn:
                if insn.is_arith_insn:
                    arith_insn_count += 1
                    if self.debug:
                        print(f"[DEBUG] Insn {arith_insn_count} ({count}): {last_insn[:-1]}")
                elif self.debug:
                    print(f"[DEBUG] Insn ({count}): {last_insn[:-1]}")

            if arith_insn_count == self.num_arith_insn:
                break
        
        emulator.finalize()
        output.write(out_buf)
        trace.close()
        output.close()