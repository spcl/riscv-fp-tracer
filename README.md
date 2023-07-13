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