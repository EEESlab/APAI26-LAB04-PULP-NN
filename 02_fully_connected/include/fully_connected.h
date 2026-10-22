#ifndef __FULLY_CONNECTED_H__
#define __FULLY_CONNECTED_H__


#include <stdint.h>


typedef struct {
  uint8_t *input;
  int8_t  *weights;
  int32_t *output;
  int32_t *partial;  // NUM_CORES * channels_out partial sums (Task 2.4)
  int      channels_in;
  int      channels_out;
} fc_args_t;


void fully_connected(const fc_args_t args);
void fully_connected_split_in(const fc_args_t args);


#endif  // __FULLY_CONNECTED_H__
