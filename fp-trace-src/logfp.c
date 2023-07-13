/*
 * Copyright (C) 2021, Alexandre Iooss <erdnaxe@crans.org>
 *
 * Log instruction execution with memory access.
 *
 * License: GNU GPL, version 2 or later.
 *   See the COPYING file in the top-level directory.
 */
#include <glib.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <qemu-plugin.h>

#define START_SYMBOL "main"
#define END_SYMBOL "_dl_fini"
#define HEX_DIGITS 16
#define FLUSH_FREQ 3000000
#define ENTRY_SIZE 64

static int counter = 0;


QEMU_PLUGIN_EXPORT int qemu_plugin_version = QEMU_PLUGIN_VERSION;

/* Store last executed instruction on each vCPU as a GString */
static GPtrArray *last_exec;
static GMutex expand_array_lock;

static GPtrArray *imatches;
static GArray *amatches;

// Stores all functions whose instructions need to be traced
static GPtrArray *fmatches;

// Stores all functions whose instructions need to be excluded from tracing
static GPtrArray *fexcludes;

// If true, will trace all instructions that are executed between
// `main()` and `_dl_init()`. This is compatible with `ffilter` and `fexclude`.
static bool trace_main = false;
// Is only used when `trace_main` is set
static bool start = false;
// A mutex for multi-threaded trace logging
static GMutex log_lock;

/* static GPtrArray *log_entries; */
static GString *log_entries;
static int log_entry_count = 0;
static GPtrArray *fnames;

static const char freg_name[32][5] = {
    "ft0",  "ft1",  "ft2",  "ft3",  "ft4",  "ft5",  "ft6",  "ft7",
    "fs0",  "fs1",  "fa0",  "fa1",  "fa2",  "fa3",  "fa4",  "fa5",
    "fa6",  "fa7",  "fs2",  "fs3",  "fs4",  "fs5",  "fs6",  "fs7",
    "fs8",  "fs9",  "fs10", "fs11", "ft8",  "ft9",  "ft10", "ft11",
};




struct fp_data
{
    bool is_fp_insn;
    int num_regs;
    int fp_regs[4];
    uint64_t vaddr;
    char insn[64];
};


/* format instruction */
static void append(char *s1, const char *s2, size_t n)
{
    size_t l1 = strlen(s1);
    if (n - l1 - 1 > 0) {
        strncat(s1, s2, n - l1);
    }
}

/**
 * An efficient way to convert double to a hex string
 */
void double_to_hex_str(double value, char *hex_string) {
    unsigned char *ptr = (unsigned char *)&value;
    size_t i, j;

    for (i = 0, j = sizeof(double) - 1; i < sizeof(double); i++, j--) {
        hex_string[i * 2] = "0123456789ABCDEF"[ptr[j] / 16];
        hex_string[i * 2 + 1] = "0123456789ABCDEF"[ptr[j] % 16];
    }

    hex_string[sizeof(double) * 2] = '\0';
}


/**
 * Converts 
 */
static void fp_data_to_str(struct fp_data *data, char *buf, uint32_t n)
{
    /* char vaddr_buf[32]; */
    /* sprintf(vaddr_buf, "0x%x ", data->vaddr); */
    /* append(buf, vaddr_buf, n); */
    append(buf, data->insn, n);
    append(buf, ";", n);
    char tmp[sizeof(double) + 1];
    assert(data->num_regs <= 4);
    /* if (data->vaddr == 0x1b34a) */
    /* { */
    /*     printf("[DEBUG] Current pc: 0x%lx\n", qemu_plugin_read_pc()); */
    /*     printf("[DEBUG] reg: %d, reg val: %f\n", data->fp_regs[0], qemu_plugin_read_fp_reg(data->fp_regs[0])); */
    /* } */
    /* printf("[DEBUG] opcode: %s %d %d\n", data->insn, data->num_regs, data->fp_regs[0]); */
    for (int reg = 0; reg < data->num_regs; ++reg)
    {
        double reg_val;
        if (data->is_fp_insn)
        {
            reg_val = qemu_plugin_read_fp_reg(data->fp_regs[reg]);
        }
        else
        {
            reg_val = qemu_plugin_read_reg(data->fp_regs[reg]);
        }
        /* printf(" reg %d\n", data->fp_regs[reg]); */
        double_to_hex_str(reg_val, tmp);
        /* printf(" %s %d", tmp, reg); */
        append(buf, tmp, n);
        if (reg < data->num_regs - 1)
            append(buf, " ", n);
    }
    append(buf, "\n", n);
}


/**
 * Fast string to integer conversion
 */
static int fast_str_to_int(const char* str) {
    int result = 0;

    while (*str >= '0' && *str <= '9') {
        result = 10 * result + (*str - '0');
        str++;
    }

    return result;
}

/**
 * Fast integer to string function, assuming that the given
 * unsigned 32-bit integer only has two digits.
 */
static int fast_int_to_str(char *buf, uint32_t n)
{
    if (n < 10)
    {
        buf[0] = n % 10 + '0';
        return 1;
    }
    
    int i = 1;
    while (n > 0)
    {
        buf[i] = n % 10 + '0';
        n /= 10;
        i--;
    }
    return 2;
}


/**
 * Fast conversion from an unsigned 64-bit integer to
 * a string without any zero padding.
 */
static int fast_hex_to_str(char *buf, uint64_t addr)
{
    int offset = 3;
    int i;

    for (i = HEX_DIGITS - 1 + offset; i>= offset && addr > 0; --i)
    {
        int digit = addr & 0xf;
        buf[i] = (digit < 10) ? (digit + '0') : (digit - 10 + 'a');
        addr >>= 4;
    }
    buf[HEX_DIGITS + offset] = '\0';
    return i;
}



/**
 * Flushes all the trace entries to the log file.
 */
static void flush_log_entries()
{
    /* g_mutex_lock(&log_lock); */
    /* qemu_plugin_outs(buf->str); */
    // Reinitializes log_entries
    /* g_ptr_array_free(log_entries, true); */
    /* log_entries = g_ptr_array_new(); */
    qemu_plugin_outs(log_entries->str);
    g_string_truncate(log_entries, 0);
    /* g_mutex_unlock(&log_lock); */
}


/*
 * Expand last_exec array.
 *
 * As we could have multiple threads trying to do this we need to
 * serialise the expansion under a lock. Threads accessing already
 * created entries can continue without issue even if the ptr array
 * gets reallocated during resize.
 */
static void expand_last_exec(int cpu_index)
{
    g_mutex_lock(&expand_array_lock);
    while (cpu_index >= last_exec->len) {
        /* GString *s = g_string_new(NULL); */
        /* g_ptr_array_add(last_exec, s); */
        struct fp_data *data = g_new0(struct fp_data, 1);
        g_ptr_array_add(last_exec, data);
    }
    g_mutex_unlock(&expand_array_lock);
}

/**
 * Add memory read or write information to current instruction log
 */
static void vcpu_mem(unsigned int cpu_index, qemu_plugin_meminfo_t info,
                     uint64_t vaddr, void *udata)
{
    /* GString *s; */
    struct fp_data* data;
    /* Find vCPU in array */
    g_assert(cpu_index < last_exec->len);
    data = g_ptr_array_index(last_exec, cpu_index);
    char buf[20];
    int i = fast_hex_to_str(buf, vaddr);
    buf[i--] = 'x';
    buf[i--] = '0';
    buf[i] = ';';
    append(data->insn, &buf[i], 64);
    /* g_string_insert_len(s, -1, (const char *) &buf[i], 19 - i); */
    /* g_string_append(s, &buf[i]); */
    /* printf("[DEBUG] %d, len: %ld\n", i, strlen(&buf[i])); */
    /* g_string_append(s, &buf[i]); */
    /* g_string_append_printf(s, ";0x%08"PRIx64, vaddr); */
    /* Indicate type of memory access */
    /*
    if (qemu_plugin_mem_is_store(info)) {
        g_string_append(s, ", store");
    } else {
        g_string_append(s, ", load");
    }
    */
    /* If full system emulation log physical address and device name */
    /* struct qemu_plugin_hwaddr *hwaddr = qemu_plugin_get_hwaddr(info, vaddr); */
    /* if (hwaddr) { */
    /*     uint64_t addr = qemu_plugin_hwaddr_phys_addr(hwaddr); */
    /*     const char *name = qemu_plugin_hwaddr_device_name(hwaddr); */
    /*     g_string_append_printf(s, " 0x%08"PRIx64" %s", addr, name); */
    /* } else { */
    /*     g_string_append_printf(s, " 0x%08"PRIx64, vaddr); */
    /* } */
}




/**
 * Log instruction execution
 */
static void vcpu_insn_exec(unsigned int cpu_index, void *udata)
{
    /* Find or create vCPU in array */
    if (cpu_index >= last_exec->len)
    {
        expand_last_exec(cpu_index);
    }
    struct fp_data *data = g_ptr_array_index(last_exec, cpu_index);
    /* struct fp_data *data = (struct fp_data *) udata; */
    if (data->num_regs > 0)
    {
        char buf[256];
        buf[0] = '\0';
        /* printf("[DEBUG] Current pc: 0x%lx\n", qemu_plugin_read_pc()); */
        /* if (data->vaddr == 0x1b34a) */
        /* { */
        /*     printf("[DEBUG] Current pc: 0x%lx\n", qemu_plugin_read_pc()); */
        /*     printf("[DEBUG] reg: %d, reg val: %f\n", data->fp_regs[0], qemu_plugin_read_fp_reg(data->fp_regs[0])); */
        /* } */
        fp_data_to_str(data, buf, sizeof(buf));
        g_string_append(log_entries, buf);
        log_entry_count++;
    }
    /* g_free(data); */
    /* printf("[DEBUG] insn: %s, reg[0]: %d\n", data->insn, data->fp_regs[0]); */

    /* Store new instruction in cache */
    struct fp_data *new_data = (struct fp_data *) udata;
    memcpy(data, udata, sizeof(struct fp_data));
    /* strcpy(data->insn, new_data->insn); */
    /* memcpy(data->fp_regs, new_data->fp_regs, sizeof(new_data->fp_regs)); */
    /* data->num_regs = new_data->num_regs; */
    if (log_entry_count % FLUSH_FREQ == 0)
        flush_log_entries();
}

/**
 * On translation block new translation
 *
 * QEMU convert code by translation block (TB). By hooking here we can then hook
 * a callback on each instruction and memory access.
 */
static void vcpu_tb_trans(qemu_plugin_id_t id, struct qemu_plugin_tb *tb)
{
    struct qemu_plugin_insn *insn;
    bool use_filter = (fmatches != NULL) || (fexcludes != NULL);
    bool skip = (fmatches != NULL) || (fexcludes != NULL) || trace_main;
    struct fp_data *data;
    size_t n = qemu_plugin_tb_n_insns(tb);
    for (size_t i = 0; i < n; i++)
    {
        // skip = fmatches != NULL;
        
        char *insn_disas;
        const char *symbol;
        /* uint64_t insn_vaddr; */
        /*
         * `insn` is shared between translations in QEMU, copy needed data here.
         * `output` is never freed as it might be used multiple times during
         * the emulation lifetime.
         * We only consider the first 32 bits of the instruction, this may be
         * a limitation for CISC architectures.
         */
        insn = qemu_plugin_tb_get_insn(tb, i);
        symbol = qemu_plugin_insn_symbol(insn);
        /* if (trace_main) */
        /* { */
        /*     if (g_strcmp0(symbol, START_SYMBOL) == 0) */
        /*         start = true; */
        /*     else if (g_strcmp0(symbol, END_SYMBOL) == 0) */
        /*         start = false; */
        /* } */
        insn_disas = qemu_plugin_insn_disas(insn);
        if (strncmp(insn_disas, "nop", 3) == 0)
        {
            start = !start;
            printf("[DEBUG] nop, start: %d\n", start);
        }
        // If the disassembled code matches TRACE_MARK
        /* if (strncmp(insn_disas, TRACE_MARK, 3) == 0) */
        /*     // Sets `start` to false if it is true, false otherwise */
        /*     start = start ? false : true; */
        
        /* if (!start || symbol == NULL || g_str_has_prefix(symbol, "kernel")) */
        /*     continue; */
        /*
         * If we are filtering we better check out if we have any
         * hits. The skip "latches" so we can track memory accesses
         * after the instruction we care about.
         */
        if (trace_main && start && !use_filter)
            skip = false;
        
        if ((trace_main && start) || !trace_main)
        {
            /* if (symbol == NULL) */
            /*     skip = false; */
            
            if (use_filter && symbol != NULL)
            {
                // Filter based on function names (i.e. symbols)
                if (fexcludes)
                {
                    skip = false;
                    // Checks if the symbol belongs to a list of
                    // functions that need to be ignored
                    int j;
                    for (j = 0; j < fexcludes->len && !skip; ++j)
                    {
                        char *fname = g_ptr_array_index(fexcludes, j);
                        if (g_str_has_prefix(symbol, fname))
                            skip = true;
                    }
                } else { skip = false; }
                
                if (!skip && fmatches)
                {
                    // Checks if the symbol belongs to a list of
                    // functions that need to be traced. Note
                    // that the excluded functions take priority.
                    // In other words if a symbol prefix appears both
                    // in `fmathces` and `fexcludes`, it will not
                    // be traced.
                    skip = true;
                    int j;
                    for (j = 0; j < fmatches->len && skip; ++j)
                    {
                        char *fname = g_ptr_array_index(fmatches, j);
                        if (g_str_has_prefix(symbol, fname))
                        {
                            skip = false;
                        }
                    }
                }
            }
        }
        
        if (skip) {
            g_free(insn_disas);
        } else {
            /* insn_disas = cleanup_disas_insn(insn_disas); */
            /* insn_vaddr = qemu_plugin_insn_vaddr(insn); */
            /* uint32_t insn_opcode; */
            /* insn_opcode = *((uint32_t *)qemu_plugin_insn_data(insn)); */
            /* char *output = g_strdup_printf("0x%"PRIx64", 0x%"PRIx32", \"%s\"", */
            /*                                insn_vaddr, insn_opcode, insn_disas); */
            
            // Splits the string into tokens
            /* struct fp_data *data = malloc(sizeof(struct fp_data)); */
            data = g_new0(struct fp_data, 1);
            if (insn_disas[0] != '\0')
            {
                /* data = &tmp_data; */
                char insn_buf[64] = { 0 };
                int n = sizeof(insn_buf);
                char *token;
                // Checks if the instruction is actually
                // a floating point instruction
                bool is_fp_insn = insn_disas[0] == 'f';
                
                token = strtok(insn_disas, " ");
                append(insn_buf, token, n);
                int reg_count = 0;
                while (token != NULL)
                {
                    token = strtok(NULL, " ");
                    if (token != NULL)
                    {
                        int reg = fast_str_to_int(token);
                        if (is_fp_insn)
                        {
                            append(insn_buf, " ", n);
                            append(insn_buf, freg_name[reg], n);
                        }
                        data->fp_regs[reg_count++] = reg;
                    }
                }
                strncpy(data->insn, insn_buf, n);
                data->is_fp_insn = is_fp_insn;
                data->num_regs = reg_count;
                data->vaddr = qemu_plugin_insn_vaddr(insn);
            }
            /* bool exists = false; */
            /* for (int x = 0; x < fnames->len && !exists; ++x) */
            /* { */
            /*     char *fname = g_ptr_array_index(fnames, x); */
            /*     if (strcmp(fname, symbol) == 0) { */
            /*         exists = true; */
            /*     } */
            /* } */
            /* if (!exists) */
            /* { */
            /*     printf("[DEBUG] new symbol: %s\n", symbol); */
            /*     g_ptr_array_add(fnames, symbol); */
            /* } */
            /* Register callback on memory read or write */
            qemu_plugin_register_vcpu_mem_cb(insn, vcpu_mem,
                                             QEMU_PLUGIN_CB_NO_REGS,
                                             QEMU_PLUGIN_MEM_RW, NULL);

            /* Register callback on instruction */
            qemu_plugin_register_vcpu_insn_exec_cb(insn, vcpu_insn_exec,
                                                   QEMU_PLUGIN_CB_NO_REGS, data);

            /* reset skip */
            // skip = (imatches || amatches);
            skip = (fmatches != NULL) || (fexcludes != NULL) || trace_main;
        }
    }
}

/**
 * On plugin exit, print last instruction in cache
 */
static void plugin_exit(qemu_plugin_id_t id, void *p)
{
    flush_log_entries();
    /* guint i; */
    /* GString *s; */
    /* for (i = 0; i < last_exec->len; i++) { */
        /* s = g_ptr_array_index(last_exec, i); */
        /* if (s->str) { */
        /*     flush_log_entries(); */
            /* qemu_plugin_outs(s->str); */
            /* qemu_plugin_outs("\n"); */
    /*     } */
    /* } */
}

/* Add a match to the array of matches */
static void parse_insn_match(char *match)
{
    if (!imatches) {
        imatches = g_ptr_array_new();
    }
    g_ptr_array_add(imatches, match);
}

static void parse_vaddr_match(char *match)
{
    uint64_t v = g_ascii_strtoull(match, NULL, 16);

    if (!amatches) {
        amatches = g_array_new(false, true, sizeof(uint64_t));
    }
    g_array_append_val(amatches, v);
}


static void parse_function_match(char *match)
{
    if (!fmatches)
        // Initializes the array
        fmatches = g_ptr_array_new();
    // Stores the name of the function
    g_ptr_array_add(fmatches, match);
}


/**
 * Excludes a single function whose name is
 * specified by `exclude` from being traced.
 * If the argument is "default" however, a predefined
 * list of functions such as `_dl_runtime_resolve()` and
 * `_dl_start()` will be added to the list of
 * functions to be excluded.
 */
static void parse_exclude_function(char *exclude)
{

    if (!fexcludes)
        // Initializes the array
        fexcludes = g_ptr_array_new();

    if (g_strcmp0(exclude, "default") == 0)
    {
        const char *excluded_prefixes[] = {
            "dl", "index", "search_cache", "strlen", "check_match",
            "do_lookup_x", "strcmp", "call_init", "lookup_malloc_symbol",
            "memset", "sbrk", "print"
        };
        // Adds a list of functions to be excluded from tracing
        int len = sizeof(excluded_prefixes) / sizeof(char *);
        for (int i = 0; i < len; ++i)
            g_ptr_array_add(fexcludes, (char *) excluded_prefixes[i]);

    }
    else
    {
        g_ptr_array_add(fexcludes, exclude);
    }
}


/**
 * Install the plugin
 */
QEMU_PLUGIN_EXPORT int qemu_plugin_install(qemu_plugin_id_t id,
                                           const qemu_info_t *info, int argc,
                                           char **argv)
{
    /*
     * Initialize dynamic array to cache vCPU instruction. In user mode
     * we don't know the size before emulation.
     */
    if (info->system_emulation) {
        last_exec = g_ptr_array_sized_new(info->system.max_vcpus);
    } else {
        last_exec = g_ptr_array_new();
    }

    for (int i = 0; i < argc; i++) {
        char *opt = argv[i];
        g_autofree char **tokens = g_strsplit(opt, "=", 2);
        if (g_strcmp0(tokens[0], "ifilter") == 0)
        {
            parse_insn_match(tokens[1]);
        }
        else if (g_strcmp0(tokens[0], "afilter") == 0)
        {
            parse_vaddr_match(tokens[1]);
        }
        else if (g_strcmp0(tokens[0], "ffilter") == 0)
        {
            parse_function_match(tokens[1]);
        }
        else if (g_strcmp0(tokens[0], "fexclude") == 0)
        {
            parse_exclude_function(tokens[1]);
        }
        else if (g_strcmp0(tokens[0], "trace_main") == 0)
        {
            if (g_strcmp0(tokens[1], "1") == 0)
                trace_main = true;
        }
        else
        {
            fprintf(stderr, "option parsing failed: %s\n", opt);
            return -1;
        }
    }
    // Initializes the log array
    log_entries = g_string_new(NULL);
    fnames = g_ptr_array_new();
    /* Register translation block and exit callbacks */
    qemu_plugin_register_vcpu_tb_trans_cb(id, vcpu_tb_trans);
    qemu_plugin_register_atexit_cb(id, plugin_exit, NULL);

    return 0;
}
