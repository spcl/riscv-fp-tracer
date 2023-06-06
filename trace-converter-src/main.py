import argparse
from trace_converter import TraceConverter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Trace converter")
    parser.add_argument("-i", "--input-file", dest="input_file", required=True,
                        help="Path to the trace file to be converted.")
    parser.add_argument("-o", "--output-file", dest="output_file",
                        default="fp16_trace.out",
                        help="Path to the destination file after conversion.")
    args = parser.parse_args()

    converter = TraceConverter(args.input_file, args.output_file)
    converter.convert()
