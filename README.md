## Quick Start
- Clone QEMU from https://github.com/qemu/qemu/
- Enter the following commands to build QEMU
  ```console
  > cd fp-trace-src
  > cp api.c <qemu-dir>/plugins/
  > cp qemu-plugins.symbols <qemu-dir>/plugins
  > cp qemu-plugin.h <qemu-dir>/include/qemu/
  > cp riscv.c <qemu-dir>/disas/
  > cd <qemu-dir>
  > ./configure --enable-plugins --target-list=riscv64-linux-user
  > make -j8
  ```
- To collect the floating point traces of applications simply 
use the `trace.sh` script followed by the binary executable of
the application that you want to trace. For instance, to trace LULESH, the following command can be entered, and the collected
trace can be found in `trace.out`.
```console
> ./trace.sh "lulesh2.0 -i 100 -s 8"
```

### Applications
- [LULESH2.0](https://github.com/LLNL/LULESH/tree/master)
  - Command: `lulesh2.0 -i 100 -s 4`
- [OpenCV](https://github.com/opencv/opencv) (4.x)
  - Application: Square detection
  - Command: `square`
  - Images tested:
    1. `home.jpg`
    2. `apple.jpg`
    3. `pic1.png`
    4. `pic2.png`
    5. `pic3.png`
    6. `pic4.png`
    7. `pic5.png`
    8. `pic6.png`
    9. `orange.jpg`
    10. `lena.jpg`
    11. `mask.png`
    12. `stuff.jpg`
    13. `HappyFish.jpg`
    14. `Blender_Suzanne1.jpg`
    15. `notes.png`
- [HPCG](https://github.com/hpcg-benchmark/hpcg) (V3.1)
  - Command: `xhpcg 16 16 16`
- [Kripke](https://github.com/LLNL/Kripke) (V1.2.5-dev)
  - Command: `kripke.exe --zones 8,8,8 --niter 3 --groups 2`
- [Nyx](https://github.com/AMReX-Astro/Nyx?tab=readme-ov-file) (V 21.02.1-184-gfb5216c7cb87)
  - Change the following fields in `Exec/LyA/inputs.rt` in the build directory:
    - `max_step = 3`
    - `amr.n_cell = 8 8 8`
    - `amr.max_grid_size = 8`
  - Make sure to place `trace.sh` in the `Exec/LyA` repository
  - Command: `nyx_LyA inputs.rt`
- [LAMMPS](https://www.lammps.org/download.html) (V 2Aug2023 Stable)
  - After compilation move `lmp_serial` to the `bench` directory. Before tracing, make sure to have `in.eam` and `Cu_u3.eam` in the same directory as `trace.sh`.
  - Command: `lmp_serial -in in.eam`
  - The configuration file `in.eam` is as follows:
```
# bulk Cu lattice

variable        x index 1
variable        y index 1
variable        z index 1

variable        xx equal 10*$x
variable        yy equal 10*$y
variable        zz equal 10*$z

units           metal
atom_style      atomic

lattice         fcc 3.615
region          box block 0 ${xx} 0 ${yy} 0 ${zz}
create_box      1 box
create_atoms    1 box

pair_style      eam
pair_coeff      1 1 Cu_u3.eam

velocity        all create 1600.0 376847 loop geom

neighbor        1.0 bin
neigh_modify    every 1 delay 5 check yes

fix             1 all nve

timestep        0.005
thermo          50

run             5
```
