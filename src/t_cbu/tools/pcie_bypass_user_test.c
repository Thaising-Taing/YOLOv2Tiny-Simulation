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
#include <unistd.h>

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

#define MAP_SIZE_ram (256*1024*1024UL)
#define MAP_SIZE_reg (256*1024*1024UL)
#define MAP_MASK_ram (MAP_SIZE_ram - 1)
#define MAP_MASK_reg (MAP_SIZE_reg - 1)



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

	for(i=0;i<4096;i+=4){
//	     writeval = htoll(rand());
	     writeval = i>>2;
   	     *((uint32_t *) (virt_addr_ram + i)) = writeval;
	}



	printf("ram Memory compare.\n");
	fflush(stdout);

        int comp_result=1;

	for(i=0;i<4096;i+=4){
		read_result = *((uint32_t *) (virt_addr_ram + i));
        	read_result = ltohl(read_result);
                if( read_result != i>>2 ) {
					printf("ram Memory mismatch address %8x expected %x, read %x.\n",  i, i>>2, read_result );
					comp_result = 0;
					}

	}

        if(comp_result ==0) printf("compare error\n");
                  else printf("compare done\n");
        for(i=0 ; i< 100 ; i++){
        printf("led output test %x\r",i);
        *((uint32_t *) (virt_addr_reg)) = i;
         usleep(50000);

         }
        printf("\n");

	fflush(stdout);

	if (munmap(map_base_reg, MAP_SIZE_reg) == -1)
		FATAL;
	close(fd_reg);
	if (munmap(map_base_ram, MAP_SIZE_ram) == -1)
		FATAL;
	close(fd_ram);

	return 0;
}
