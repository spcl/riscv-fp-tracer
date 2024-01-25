import re
import time


class TraceX86(gdb.Command):
    """
    Implementation from:
    https://stackoverflow.com/questions/8841373/displaying-each-assembly-instruction-executed-in-gdb
    """
    reg_offset_pattern = re.compile(r"(-?\d+)\((\w+\d*)\)")
    
    def __init__(self):
        super().__init__(
            'trace-x86',
            gdb.COMMAND_BREAKPOINTS,
            gdb.COMPLETE_NONE,
            False
        )
    def invoke(self, argument, from_tty):
        argv = gdb.string_to_argv(argument)
        if argv:
            gdb.write('Does not take any arguments.\n')
        else:
            # gdb.execute("set logging on")
            # FIXME: Seems like the script only works for single-threaded programs
            # Needs to verify what multi-threaded programs look like
            thread = gdb.inferiors()[0].threads()[0]
            # Stores each line of assembly in the list first, prints
            # everything at the end
            asm_trace = []
            prev_sym = None

            while thread.is_valid():
                frame = gdb.selected_frame()
                pc = frame.pc()
                # Retrieves the current instruction
                # asm = frame.architecture().disassemble(pc)[0]['asm']
                # out = asm
                # asm_trace.append(out)
                            
                sal = frame.find_sal()
                symtab = sal.symtab
                if symtab:
                    tab_name = symtab.fullname()
                    if tab_name != prev_sym:
                        # gdb.write(f"[DEBUG] symbol: {tab_name}\n", gdb.STDERR)
                        asm_trace.append(tab_name)
                    prev_sym = symtab.fullname()
                # TODO: This is probably not the best way to identify end of a program
                # if symtab and "libc" in symtab.fullname():
                #     break
                gdb.execute('si', to_string=False)
                if len(asm_trace) == 100:
                    
            # gdb.execute('continue', to_string=True)
                    with open("trace.out", "a") as trace_file:
                        trace_file.write("\n".join(asm_trace))
                        asm_trace.clear()

            with open("trace.out", "a") as trace_file:
                trace_file.write("\n".join(asm_trace))


TraceX86()
