/*
 * data_allocation_8_32_8.h
 * Nazareno Bruschi <nazareno.bruschi@unibo.it>
 *
 * Copyright (C) 2019-2020 University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __PULPNN_TEST_DATA_ALLOCATION__
#define __PULPNN_TEST_DATA_ALLOCATION__

// Layer dimensions, can be overridden from the command line:
// make clean all run CH_IN=<value> CH_OUT=<value>
#ifndef CH_IM_IN
#define CH_IM_IN 1024
#endif
#ifndef CH_IM_OUT
#define CH_IM_OUT 16
#endif


// Inputs and weights are generated at runtime, the reference output is
// computed on the cluster with a plain (non-SIMD) dot-product
PI_L1 uint8_t IN_INT8_L1[CH_IM_IN] __attribute__((aligned(4)));
PI_L1 int8_t WEIGHT_INT8_L1[(CH_IM_IN * CH_IM_OUT)] __attribute__((aligned(4)));
PI_L1 int32_t OUT_L1[CH_IM_OUT];
PI_L1 int32_t PARTIAL_L1[NUM_CORES * CH_IM_OUT];
PI_L2 int32_t OUT_REF_L2[CH_IM_OUT];


#endif
