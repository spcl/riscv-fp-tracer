import argparse
from pathlib import Path
from trace_converter import TraceConverter
from fp_stats_collector import FpStatsCollector
from mem_trace_analyzer import MemoryTraceAnalyzer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Trace converter")
    parser.add_argument("-i", "--input-file", dest="input_file", required=True,
                        help="Path to the trace file to be converted.")
    parser.add_argument("-o", "--output-file", dest="output_file",
                        default="fp16_trace.out",
                        help="Path to the destination file after conversion.")
    parser.add_argument("-s", "--collect-stats", dest="collect_stats",
                        default=False, action="store_true",
                        help="If set, will collect statistics about FP operations. "
                        "The statistics will be saved automatically to <input-file-name>.stats")
    parser.add_argument("--ignore-ex-prop", dest="ignore_ex_prop", default=False,
                        action="store_true",
                        help="If set, will ignore the propagation of exceptions when "
                        "computing the overflow/underflow and catastrophic cancellation metrics.")
    parser.add_argument("-n", "--num-arith-insn", dest="num_arith_insn",
                        type=int, default=None,
                        help="Specifies the maximum number of FP ARITHMETIC instructions to execute "
                        " from the trace. Note that instructions such as 'fmv', 'sd', and 'sw' "
                        " do not count as FP instructions.")
    parser.add_argument("--cc-threshold", dest="cc_threshold",
                        type=int, default=5,
                        help="A threshold that defines the number of bits "
                        "that need to shifted after a floating point subtraction "
                        "in order for that operation to be considered to have "
                        "caused catastrophic cancellation (CC)")
    parser.add_argument("--debug", dest="debug", default=False,
                        action="store_true", help="Output debug messages")
    parser.add_argument("-m", "--analyze-mem-trace", dest="analyze_mem_trace",
                        default=False, action="store_true",
                        help="If set will turn on memory trace analysis. Note that "
                        "this is an experimental feature that is unrelated to the FP project.")
    parser.add_argument("-b", "--block", dest="block_size",
                        default=None, type=int,
                        help="If set, will perform overflow/underflow and "
                        "catastrophic cancellation analysis on a block by "
                        "block basis, where the block size (i.e., number of "
                        "instructions in a block) is determined by this number. "
                        "Note that the analysis results will only be accurate "
                        "when `ignore-ex-prop` is set.")
    args = parser.parse_args()

    if args.collect_stats:
        print("[INFO] FP16 statistics will be collected")
        print(f"[INFO] Ignore FP exception propagation: {args.ignore_ex_prop}")
        print(f"[INFO] Catastrophic cancellation bit shift threshold: {args.cc_threshold}")
        if args.block_size is not None:
            assert args.block_size > 0
            print("[INFO] Exception analysis will be performed on blocks of FP "
                  f"arithmetic instructions of size {args.block_size}")
        
    if args.num_arith_insn is not None:
        print(f"[INFO] Executing {args.num_arith_insn} FP arithmetic instructions")

    if args.analyze_mem_trace:
        print(f"[INFO] Memory trace analysis enabled [experimental feature]")

    stats_file_name = f"{Path(args.input_file).stem}.stats"
    collector = FpStatsCollector(stats_file_name, args.ignore_ex_prop,
                                 args.cc_threshold, args.block_size,
                                 args.collect_stats, args.debug)
    mem_analyzer = MemoryTraceAnalyzer(args.analyze_mem_trace)
    converter = TraceConverter(args.input_file, args.output_file, collector,
                               mem_analyzer, args.num_arith_insn, args.debug)
    converter.convert()
    
    if args.collect_stats:
        # Output statistics
        collector.output_results()

    if args.analyze_mem_trace:
        # mem_analyzer.visualize_access_pattern()
        mem_analyzer.save_mem_trace()
        print("[INFO] Saved memory trace file")