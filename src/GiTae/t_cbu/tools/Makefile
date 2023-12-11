CC ?= gcc

all: reg_rw reg_rw_16MB reg_rw_32MB reg_rw_256MB  dma_to_device dma_from_device performance pcx_test pcie_bypass_user_test

dma_to_device: dma_to_device.o
	$(CC) -lrt -o $@ $< -D_FILE_OFFSET_BITS=64 -D_GNU_SOURCE -D_LARGE_FILE_SOURCE

dma_from_device: dma_from_device.o
	$(CC) -lrt -o $@ $< -D_FILE_OFFSET_BITS=64 -D_GNU_SOURCE -D_LARGE_FILE_SOURCE

performance: performance.o
	$(CC) -o $@ $< -D_FILE_OFFSET_BITS=64 -D_GNU_SOURCE -D_LARGE_FILE_SOURCE

reg_rw: reg_rw.o
	$(CC) -o $@ $<

reg_rw_16MB: reg_rw_16MB.o
	$(CC) -o $@ $<

reg_rw_32MB: reg_rw_32MB.o
	$(CC) -o $@ $<

reg_rw_256MB: reg_rw_256MB.o
	$(CC) -o $@ $<



pcx_test: pcx_test.o
	$(CC) -o $@ $<

pcie_bypass_user_test: pcie_bypass_user_test.o
	$(CC) -o $@ $<

%.o: %.c
	$(CC) -c -std=c99 -o $@ $< -D_FILE_OFFSET_BITS=64 -D_GNU_SOURCE -D_LARGE_FILE_SOURCE

clean:
	rm -rf pcx_test reg_rw reg_rw_16MB reg_rw_32MB reg_rw_256MB *.o *.bin dma_to_device dma_from_device performance pcie_bypass_user_test

