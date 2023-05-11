#include "stdio.h"
#define X 1.2345

// Type your code here, or load an example.
double __attribute__ ((noinline)) kernel(double num) {
    return num * num;
}

int main(int argc, char **argv)
{
  double x = X;
  printf("[DEBUG] x: %f\n", x);
  return kernel(x);
}
