# ===================================================
# ===================================================
# Warning:
# This is an experimental feature used for another potential project
# ===================================================
# ===================================================
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Optional, List
from utils import hex64_to_uint64


class MemoryTraceAnalyzer(object):
    """
    An object that is in charge of collecting all the information
    related to the memory accesses in FP traces.
    """
    def __init__(self, is_enabled: bool = True) -> None:
        self.is_enabled = is_enabled
        self.mem_addrs = []

    def add_mem_addr(self, addr: Optional[str] = None) -> None:
        """
        First converts the given memory address from a string to
        an np.uint64 value, then stores it in the array for
        the analysis later.
        """
        if not self.is_enabled:
            return

        if addr is not None:
            self.mem_addrs.append(hex64_to_uint64(addr))

    def plot_mem_addr_distribution(self) -> None:
        """
        Plots the distribution of the collected memory addresses.
        """
        if not self.is_enabled:
            print("Memory trace analysis not enabled...")
            return
        plt.hist(self.mem_addrs, bins=1000)

        plt.xlabel("Memory address")
        plt.ylabel("Count")

        plt.yscale("log")
        plt.xscale("log")

        # Change x tick labels to hex
        ax = plt.gca()
        xlabels = map(lambda t: '0x%08X' % int(t), ax.get_xticks())
        ax.set_xticklabels(xlabels)
        # Tilt the x tick labels
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig("mem_addr_distribution.png", format="png")
        plt.close()
    

    def save_mem_trace(self, out_file: Optional[str] = None) -> None:
        """
        Saves the collected memory trace to the given file as a
        binary file.
        If `out_file` is None, the default will be "mem_trace.npy"
        """
        if out_file is None:
            out_file = "mem_trace.npy"
        
        np.save(out_file, self.mem_addrs)
        

    def visualize_access_pattern(self, block_size: int = 4,
                                 row_size: int = 512,
                                 min_addr: Optional[int] = None,
                                 max_addr: Optional[int] = 0x10000000) \
                                    -> None:
        """
        Visualizes the memory access pattern as a heat map.
        @param block_size: ...
        @param row_size:...
        @param min_addr:...
        """
        if not self.is_enabled:
            print("Memory trace analysis not enabled...")
            return
        assert row_size % block_size == 0
        # Converts the collected memory addresses to an np array
        addrs = np.array(self.mem_addrs)
        if min_addr is not None:
            addrs = addrs[np.where(addrs > min_addr)]
        if max_addr is not None:
            if min_addr is not None:
                assert max_addr > min_addr
            addrs = addrs[np.where(addrs < max_addr)]

        min_val = np.min(addrs)
        max_val = np.max(addrs)

        min_col = min_val // block_size
        min_row = min_val // row_size

        unique_addrs, addr_counts = \
            np.unique(addrs, return_counts=True)
        
        # Create a matrix to represent the heatmap
        # The number of rows in the matrix should equal
        # (max_val - min_val) divided by the row_size
        # The number of columns should be equal to 
        # the row size divided by the block size
        rows = math.ceil((max_val - min_val) / row_size)
        columns = row_size // block_size
        print(rows, columns)
        heatmap = np.zeros((rows, columns))

        # Fill the heatmap matrix with the frequency of each address
        for i, addr in enumerate(unique_addrs):
            row = int((addr - min_val) // row_size)
            col = int(((addr // block_size) - min_col) % columns)
            # print(hex(addr), row, col, addr_counts[i])
            heatmap[row, col] = addr_counts[i]

        plt.figure(figsize=(10, 16))
        min_count = np.min(addr_counts)
        max_count = np.max(addr_counts)
        # Plot the heatmap
        plt.imshow(heatmap,
                   norm=matplotlib.colors.LogNorm(vmin=min_count, vmax=max_count))

        # Add colorbar
        plt.colorbar()

        plt.savefig("mem_access_pattern.png", format="png")
        plt.close()


