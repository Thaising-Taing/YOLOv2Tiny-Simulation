/*
 * This file is part of the Xilinx DMA IP Core driver tools for Linux
 *
 * Copyright (c) 2016-present,  Xilinx, Inc.
 * All rights reserved.
 *
 * This source code is licensed under BSD-style license (found in the
 * LICENSE file in the root directory of this source tree)
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <byteswap.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <fcntl.h>
#include <ctype.h>
#include <termios.h>

#include <sys/types.h>
#include <sys/mman.h>

/* ltoh: little to host */
/* htol: little to host */
#if __BYTE_ORDER == __LITTLE_ENDIAN
#define ltohl(x)       (x)
#define ltohs(x)       (x)
#define htoll(x)       (x)
#define htols(x)       (x)
#elif __BYTE_ORDER == __BIG_ENDIAN
#define ltohl(x)     __bswap_32(x)
#define ltohs(x)     __bswap_16(x)
#define htoll(x)     __bswap_32(x)
#define htols(x)     __bswap_16(x)
#endif

#define FATAL do { fprintf(stderr, "Error at line %d, file %s (%d) [%s]\n", __LINE__, __FILE__, errno, strerror(errno)); exit(1); } while(0)

#define MAP_SIZE_ram (16*1024*1024UL)
#define MAP_SIZE_reg (8*1024*1024UL)
#define MAP_MASK_ram (MAP_SIZE_ram - 1)
#define MAP_MASK_reg (MAP_SIZE_reg - 1)

#define PC_LENGTH (0x820)
#define PC_SRC_0 (PC_LENGTH*0)
#define PC_SRC_1 (PC_LENGTH*1)
#define PC_SRC_2 (PC_LENGTH*2)
#define PC_SRC_3 (PC_LENGTH*3)

#define PC_DST_0 (PC_LENGTH*4)
#define PC_DST_1 (PC_LENGTH*5)
#define PC_DST_2 (PC_LENGTH*6)
#define PC_DST_3 (PC_LENGTH*7)

#define OP_SEQ_0           0x108
#define OP_SEQ_1           0x10C
#define PC_SRC_0_INFO_1    0x114
#define PC_SRC_0_INFO_2    0x118
#define PC_SRC_0_INFO_3    0x11C
#define PC_SRC_0_INFO_4    0x120
#define PC_SRC_0_INFO_5    0x124
#define PC_SRC_1_INFO_1    0x128
#define PC_SRC_1_INFO_2    0x12C
#define PC_SRC_1_INFO_3    0x130
#define PC_SRC_1_INFO_4    0x134
#define PC_SRC_1_INFO_5    0x138
#define PC_SRC_2_INFO_1    0x13C
#define PC_SRC_2_INFO_2    0x140
#define PC_SRC_2_INFO_3    0x144
#define PC_SRC_2_INFO_4    0x148
#define PC_SRC_2_INFO_5    0x14C
#define PC_SRC_3_INFO_1    0x150
#define PC_SRC_3_INFO_2    0x154
#define PC_SRC_3_INFO_3    0x158
#define PC_SRC_3_INFO_4    0x15C
#define PC_SRC_3_INFO_5    0x160
#define PC_DST_0_INFO_1    0x1B4
#define PC_DST_0_INFO_2    0x1B8
#define PC_DST_0_INFO_3    0x1BC
#define PC_DST_0_INFO_4    0x1C0
#define PC_DST_0_INFO_5    0x1C4
#define PC_DST_1_INFO_1    0x1C8
#define PC_DST_1_INFO_2    0x1CC
#define PC_DST_1_INFO_3    0x1D0
#define PC_DST_1_INFO_4    0x1D4
#define PC_DST_1_INFO_5    0x1D8
#define PC_DST_2_INFO_1    0x1DC
#define PC_DST_2_INFO_2    0x1E0
#define PC_DST_2_INFO_3    0x1E4
#define PC_DST_2_INFO_4    0x1E8
#define PC_DST_2_INFO_5    0x1EC
#define PC_DST_3_INFO_1    0x1F0
#define PC_DST_3_INFO_2    0x1F4
#define PC_DST_3_INFO_3    0x1F8
#define PC_DST_3_INFO_4    0x1FC
#define PC_DST_3_INFO_5    0x200
#define PC_CONCAT_INFO_1   0x254
#define PC_CONCAT_INFO_2   0x258
#define PC_CONCAT_INFO_3   0x25C
#define PC_CONCAT_INFO_4   0x260
#define PC_CONCAT_INFO_5   0x264
#define PC_SOR_DATA_INFO_1 0x268
#define PC_SOR_DATA_INFO_2 0x26C
#define PC_SOR_DATA_INFO_3 0x270
#define PC_SOR_DATA_INFO_4 0x274
#define PC_SOR_DATA_INFO_5 0x278
#define DMA_PARAMS_1       0x280
#define OP_START           0x100

int main(int argc, char **argv)
{
	int fd_reg,fd_ram;
	void *map_base_reg, *virt_addr_reg;
	void *map_base_ram, *virt_addr_ram;
	uint32_t read_result,read_result2, writeval;
	off_t target;
	/* access width */
	int access_width = 'w';
	char *device;

        int i;


	if ((fd_reg = open("/dev/xdma0_user", O_RDWR | O_SYNC)) == -1)
		FATAL;
	printf("reg opened.\n");
	fflush(stdout);

	/* map one page */
//	map_base_reg = mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd_reg, 0);
	map_base_reg = mmap(0, MAP_SIZE_reg, PROT_READ | PROT_WRITE, MAP_SHARED, fd_reg, 0);
	if (map_base_reg == (void *)-1)
		FATAL;
	printf("reg Memory mapped at address %p.\n", map_base_reg);
	fflush(stdout);

	if ((fd_ram = open("/dev/xdma0_bypass", O_RDWR | O_SYNC)) == -1)
		FATAL;
	printf("bypass(ram) opened.\n");
	fflush(stdout);

	map_base_ram = mmap(0, MAP_SIZE_ram, PROT_READ | PROT_WRITE, MAP_SHARED, fd_ram, 0);
	if (map_base_ram == (void *)-1)
		FATAL;
	printf("ram Memory mapped at address %p.\n", map_base_ram);
	fflush(stdout);


	/* calculate the virtual address to be accessed */
	virt_addr_reg = map_base_reg;
	virt_addr_ram = map_base_ram;

	printf("ram Memory set.\n");
	fflush(stdout);

	for(i=0;i<2080;i+=4){
	     writeval = htoll(rand());
   	     *((uint32_t *) (virt_addr_ram + PC_SRC_0 + i)) = writeval;
	     writeval = htoll(rand());
   	     *((uint32_t *) (virt_addr_ram + PC_SRC_1 + i)) = writeval;
	     writeval = htoll(rand());
   	     *((uint32_t *) (virt_addr_ram + PC_SRC_2 + i)) = writeval;
	     writeval = htoll(rand());
   	     *((uint32_t *) (virt_addr_ram + PC_SRC_3 + i)) = writeval;
	}
	printf("regs set.\n");

	*((uint32_t *) (virt_addr_reg + OP_SEQ_0             )) = htoll( 0x1         );
	*((uint32_t *) (virt_addr_reg + OP_SEQ_1             )) = htoll( 0x0         );
	*((uint32_t *) (virt_addr_reg + PC_SRC_0_INFO_1      )) = htoll( 0xC         );
	*((uint32_t *) (virt_addr_reg + PC_SRC_0_INFO_2      )) = htoll( 0x3300000A         );
	*((uint32_t *) (virt_addr_reg + PC_SRC_0_INFO_3      )) = htoll( 0x80000000         );
	*((uint32_t *) (virt_addr_reg + PC_SRC_0_INFO_4      )) = htoll( 0x0         );
	*((uint32_t *) (virt_addr_reg + PC_SRC_0_INFO_5      )) = htoll( 0x820         );
	*((uint32_t *) (virt_addr_reg + PC_SRC_1_INFO_1      )) = htoll( 0xC         );
	*((uint32_t *) (virt_addr_reg + PC_SRC_1_INFO_2      )) = htoll( 0x3300000A         );
	*((uint32_t *) (virt_addr_reg + PC_SRC_1_INFO_3      )) = htoll( 0x80000820         );
	*((uint32_t *) (virt_addr_reg + PC_SRC_1_INFO_4      )) = htoll( 0x0         );
	*((uint32_t *) (virt_addr_reg + PC_SRC_1_INFO_5      )) = htoll( 0x820         );
	*((uint32_t *) (virt_addr_reg + PC_SRC_2_INFO_1      )) = htoll( 0xC         );
	*((uint32_t *) (virt_addr_reg + PC_SRC_2_INFO_2      )) = htoll( 0x3300000A         );
	*((uint32_t *) (virt_addr_reg + PC_SRC_2_INFO_3      )) = htoll( 0x80001040         );
	*((uint32_t *) (virt_addr_reg + PC_SRC_2_INFO_4      )) = htoll( 0x0         );
	*((uint32_t *) (virt_addr_reg + PC_SRC_2_INFO_5      )) = htoll( 0x820         );
	*((uint32_t *) (virt_addr_reg + PC_SRC_3_INFO_1      )) = htoll( 0xC         );
	*((uint32_t *) (virt_addr_reg + PC_SRC_3_INFO_2      )) = htoll( 0x3300000A         );
	*((uint32_t *) (virt_addr_reg + PC_SRC_3_INFO_3      )) = htoll( 0x80001860         );
	*((uint32_t *) (virt_addr_reg + PC_SRC_3_INFO_4      )) = htoll( 0x0         );
	*((uint32_t *) (virt_addr_reg + PC_SRC_3_INFO_5      )) = htoll( 0x820         );
//	*((uint32_t *) (virt_addr_reg + PC_DST_0_INFO_1      )) = htoll( -         );
	*((uint32_t *) (virt_addr_reg + PC_DST_0_INFO_2      )) = htoll( 0x33000000         );
	*((uint32_t *) (virt_addr_reg + PC_DST_0_INFO_3      )) = htoll( 0x80002080         );
	*((uint32_t *) (virt_addr_reg + PC_DST_0_INFO_4      )) = htoll( 0x0         );
	*((uint32_t *) (virt_addr_reg + PC_DST_0_INFO_5      )) = htoll( 0x820         );
//	*((uint32_t *) (virt_addr_reg + PC_DST_1_INFO_1      )) = htoll( -         );
	*((uint32_t *) (virt_addr_reg + PC_DST_1_INFO_2      )) = htoll( 0x33000000         );
	*((uint32_t *) (virt_addr_reg + PC_DST_1_INFO_3      )) = htoll( 0x800028A0         );
	*((uint32_t *) (virt_addr_reg + PC_DST_1_INFO_4      )) = htoll( 0x0         );
	*((uint32_t *) (virt_addr_reg + PC_DST_1_INFO_5      )) = htoll( 0x820         );
//	*((uint32_t *) (virt_addr_reg + PC_DST_1_INFO_1      )) = htoll( -         );
	*((uint32_t *) (virt_addr_reg + PC_DST_2_INFO_2      )) = htoll( 0x33000000         );
	*((uint32_t *) (virt_addr_reg + PC_DST_2_INFO_3      )) = htoll( 0x800030C0         );
	*((uint32_t *) (virt_addr_reg + PC_DST_2_INFO_4      )) = htoll( 0x0         );
	*((uint32_t *) (virt_addr_reg + PC_DST_2_INFO_5      )) = htoll( 0x820         );
//	*((uint32_t *) (virt_addr_reg + PC_DST_3_INFO_1      )) = htoll( -         );
	*((uint32_t *) (virt_addr_reg + PC_DST_3_INFO_2      )) = htoll( 0x33000000         );
	*((uint32_t *) (virt_addr_reg + PC_DST_3_INFO_3      )) = htoll( 0x800038E0         );
	*((uint32_t *) (virt_addr_reg + PC_DST_3_INFO_4      )) = htoll( 0x0         );
	*((uint32_t *) (virt_addr_reg + PC_DST_3_INFO_5      )) = htoll( 0x820         );
//	*((uint32_t *) (virt_addr_reg + PC_CONCAT_INFO_1     )) = htoll( -         );
	*((uint32_t *) (virt_addr_reg + PC_CONCAT_INFO_2     )) = htoll( 0x33000000         );
	*((uint32_t *) (virt_addr_reg + PC_CONCAT_INFO_3     )) = htoll( 0x80004100         );
	*((uint32_t *) (virt_addr_reg + PC_CONCAT_INFO_4     )) = htoll( 0x0         );
	*((uint32_t *) (virt_addr_reg + PC_CONCAT_INFO_5     )) = htoll( 0x820         );
//	*((uint32_t *) (virt_addr_reg + PC_SOR_DATA_INFO_1   )) = htoll( -         );
	*((uint32_t *) (virt_addr_reg + PC_SOR_DATA_INFO_2   )) = htoll( 0x33000000         );
	*((uint32_t *) (virt_addr_reg + PC_SOR_DATA_INFO_3   )) = htoll( 0x80006180         );
	*((uint32_t *) (virt_addr_reg + PC_SOR_DATA_INFO_4   )) = htoll( 0x0         );
	*((uint32_t *) (virt_addr_reg + PC_SOR_DATA_INFO_5   )) = htoll( 0x820         );
	*((uint32_t *) (virt_addr_reg + DMA_PARAMS_1         )) = htoll( 0x00010004         );
	*((uint32_t *) (virt_addr_reg + OP_START             )) = htoll( 0x1         );


	printf("wait int.\n");
	fflush(stdout);
        int wait_time=0;
	while(1){
	read_result = *((uint32_t *) (virt_addr_reg+0x100008));
        read_result = ltohl(read_result);
        if(read_result !=0 ) break;
        printf("\r%d",wait_time++);
	}

	printf("\nint.\n");

	read_result = *((uint32_t *) (virt_addr_reg+0x10));
        read_result = ltohl(read_result);
	printf("Interrupt status register %8x\n",read_result);


	printf("ram Memory compare.\n");
	fflush(stdout);

        int comp_result=1;

	for(i=0;i<1920;i+=4){
		read_result = *((uint32_t *) (virt_addr_ram + PC_DST_0 + i));
        	read_result = ltohl(read_result);
		read_result2 = *((uint32_t *) (virt_addr_ram + PC_SRC_0 + i));
        	read_result2 = ltohl(read_result2);
                if( read_result != read_result2 ) {
					printf("ram Memory mismatch address %8x expected %x, read %x.\n", PC_DST_0 + i, i, read_result );
					comp_result = 0;
					}

		read_result = *((uint32_t *) (virt_addr_ram + PC_DST_1 + i));
        	read_result = ltohl(read_result);
		read_result2 = *((uint32_t *) (virt_addr_ram + PC_SRC_1 + i));
        	read_result2 = ltohl(read_result2);
                if( read_result != read_result2 ) {
//					printf("ram Memory mismatch address %8x expected %x, read %x.\n", PC_DST_1 + i, i, read_result );
					comp_result = 0;
					}

		read_result = *((uint32_t *) (virt_addr_ram + PC_DST_2 + i));
        	read_result = ltohl(read_result);
		read_result2 = *((uint32_t *) (virt_addr_ram + PC_SRC_2 + i));
        	read_result2 = ltohl(read_result2);
                if( read_result != read_result2 ) {
//					printf("ram Memory mismatch address %8x expected %x, read %x.\n", PC_DST_2 + i, i, read_result );
					comp_result = 0;
					}

		read_result = *((uint32_t *) (virt_addr_ram + PC_DST_3 + i));
        	read_result = ltohl(read_result);
		read_result2 = *((uint32_t *) (virt_addr_ram + PC_SRC_3 + i));
        	read_result2 = ltohl(read_result2);
                if( read_result != read_result2 ) {
//					printf("ram Memory mismatch address %8x expected %x, read %x.\n", PC_DST_3 + i, i, read_result );
					comp_result = 0;
					}
	}

        if(comp_result ==0) printf("compare error\n");
                  else printf("compare done\n");
	fflush(stdout);

	if (munmap(map_base_reg, MAP_SIZE_reg) == -1)
		FATAL;
	close(fd_reg);
	if (munmap(map_base_ram, MAP_SIZE_ram) == -1)
		FATAL;
	close(fd_ram);

	return 0;
}
