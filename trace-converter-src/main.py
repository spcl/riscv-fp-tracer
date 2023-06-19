import argparse
from pathlib import Path
from trace_converter import TraceConverter
from fp_stats_collector import FpStatsCollector

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
    args = parser.parse_args()

    if args.collect_stats:
        print("[INFO] FP16 statistics will be collected")
    stats_file_name = f"{Path(args.input_file).stem}.stats"
    collector = FpStatsCollector(stats_file_name, args.collect_stats)
    converter = TraceConverter(args.input_file, args.output_file, collector)
    converter.convert()
    
    if args.collect_stats:
        # Prints statistics
        collector.plot_input_dist()
        collector.print_overflow_underflow_stats()