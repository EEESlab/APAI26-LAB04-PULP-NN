#include "fully_connected.h"
#include "pmsis.h"
#include "pulp_nn_utils.h"


/* dotp_u8_i8_i32
 *
 * Dot-product operation that does:
 *
 *        sum(a[i] * b[i])
 *
 * Function name: dotp_<type of first vector>_<type of second vector>_<type of the result>
 *
 * Arguments:
 *   - a: pointer to the first vector
 *   - b: pointer to the second vector
 *   - length: vector length
 *   - result: dot-product result
 */
int32_t dotp_u8_i8_i32(uint8_t *a, int8_t *b, size_t length) {
  int32_t sum = 0;
  for (int i = 0; i < length; i++)  {
    sum += a[i] * b[i];
  }
  return sum;
}


/* TASK 2.1: SIMD dot-product
 *
 * Implement the dot-product above using the SIMD extension of the
 * PULP cores. All the arguments are the same as in the function above.
 *
 * A 32-bit register can hold 4 8-bit operands. The vector types
 * v4u (4 x uint8_t) and v4s (4 x int8_t) let you see the a and b
 * vectors as arrays of 32-bit words, so that one load brings in
 * 4 elements at once.
 *
 * The SumDotp4(x, y, acc) macro (see include/pulp_nn_utils.h) maps to
 * a single instruction that multiplies the 4 elements of x with the
 * 4 elements of y and adds the 4 products to acc:
 *
 *        acc + x[0]*y[0] + x[1]*y[1] + x[2]*y[2] + x[3]*y[3]
 *
 * Remember that length is not always a multiple of 4: handle the
 * leftover elements too!
 */
int32_t dotp_u8_i8_i32_simd(uint8_t *a, int8_t *b, size_t length) {
  int32_t sum = 0;

  /* YOUR CODE HERE */

  return sum;
}


/* calculate_chunk_size
 *
 * Calculate the chunk size from the a priori known number of cores (NUM_CORES).
 */
int calculate_chunk_size(const int total_size) {
  // Divide the workload by the number of cores
  int chunk_size = total_size / NUM_CORES;

  // If there is a remainder from the division, increment the chunk_size
  if (total_size % NUM_CORES > 0) chunk_size++;

  return chunk_size;
}


/* fully_connected
 *
 * Execute a FullyConnected layer with given inputs and weights.
 * The `fc_args_t` structure defined in the `include/fully_connected.h`.
 */
void fully_connected(const fc_args_t args) {
  // Parallelize over the output channel dimension
  const int chunk_size = calculate_chunk_size(args.channels_out);

  // Begin and end of the core's chunk
  const int core_id = pi_core_id();
  const int begin = min(chunk_size * core_id, args.channels_out);
  const int end = min(begin + chunk_size, args.channels_out);

  /* Weights * Input
   *
   * The FullyConnected layer executes a Matrix-Vector multiplication
   * between the Weights matrix and the Input vector.
   * Matrix-Vector multiplication implemented as a series of dot-products.
   */
  for (int i = begin; i < end; i++) {
    // Pointer to the weight row corresponding to the processed output channel
    int8_t *weights_row = args.weights + (i * args.channels_in);

    // Task 2.1 - Change here between regular and simd dot-product implementation
    // Calculate the channel output at index i by doing a dot product between the input and the weight row
    args.output[i] = dotp_u8_i8_i32(args.input, weights_row, args.channels_in);
  }
}


/* TASK 2.4 (OPTIONAL): Parallelize over the input channels
 *
 * fully_connected() splits the output channels among the cores.
 * With CH_OUT=12 and 8 cores, how many output channels does each
 * core get? How many cores are idle? In general, when the layer has
 * fewer output channels than cores, some cores have nothing to do.
 *
 * Here, instead, every core computes ALL the output channels, but only
 * over its own chunk of the input channels. Every core produces
 * a partial sum for each output channel, and these partial sums must
 * then be added together (reduction) to get the final output.
 *
 * args.partial is a buffer of NUM_CORES * channels_out partial sums.
 *
 * To use this function, change the call in main.c.
 */
void fully_connected_split_in(const fc_args_t args) {
  const int core_id = pi_core_id();

  // Step 1: compute the begin and end of this core's chunk of input channels.
  //         Hint: keep begin a multiple of 4, so the SIMD loads stay aligned.
  /* YOUR CODE HERE */

  // Step 2: for each output channel i, compute the partial dot-product over
  //         this core's chunk and store it in args.partial[core_id * args.channels_out + i]
  /* YOUR CODE HERE */

  // Step 3: before the reduction, make sure all cores have written their partial sums
  /* YOUR CODE HERE */

  // Step 4: add the partial sums of all the cores into args.output.
  //         Which cores should do it? Can the reduction be parallelized too?
  /* YOUR CODE HERE */
}
