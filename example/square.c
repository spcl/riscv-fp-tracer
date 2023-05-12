#include "stdio.h"
#define X 1.2345

// Example code for performing floating point operations
double __attribute__ ((noinline)) kernel(double num) {
  num += 1.11;
  num *= 1.2;
  for (int i = 0; i < 100; ++i)
    num += 2.31;
  num /= 1.5;
  return num * num;
}

int main(int argc, char **argv)
{
  double x = X;
  return kernel(x);
}
