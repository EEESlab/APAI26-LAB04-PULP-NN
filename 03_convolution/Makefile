# APPLICATION NAME
APP = convolution

# SOURCE FILES
APP_SRCS = main.c pulp_nn_conv_u8_u8_i8.c pulp_nn_matmul_u8_i8.c

# NUMBER OF USED CLUSTER CORES
CORES ?= 1

# SPLIT THE CYCLES OF EACH CORE BETWEEN IM2COL AND MATMUL
# Uncomment the line below to measure the im2col and matmul cycles on all cores.
#APP_CFLAGS += -DPROFILE_SPLIT

# COMPILER FLAGS
APP_CFLAGS += -O3 -Iinclude -DNUM_CORES=$(CORES)
APP_LDFLAGS +=

include $(RULES_DIR)/pmsis_rules.mk
