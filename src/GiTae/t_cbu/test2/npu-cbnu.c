/*

Making a c-test for CBNU NPU

(IP Slave Address is assigned 0x8a01_0000 ~ 0x8a01_ffff)

*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <poll.h>
#include <stdint.h>
#include <inttypes.h>


#define IS_VECTOR_PRELOADED
#define CLEAR(x) memset(&(x), 0, sizeof(x))
/*
#define CBNUNPU_YOLO_START   0x8a010000  // This is base address of NPU AXI SLAVE
#define CBNUNPU_WR_MC_DATA   0x8a010004  // This is for writing MICROCODE to instruction Memory
#define CBNUNPU_WR_MC_ADDR   0x8a010008  // First 10-bits for write address of Microcode
*/

#define	CBNUNPU_YOLO_START   0x0
#define CBNUNPU_WR_MC_DATA   0x4  // This is for writing MICROCODE to instruction Memory
#define CBNUNPU_WR_MC_ADDR   0x8  // First 10-bits for write address of Microcode
#define CBNUNPU_RD_MC_DATA   0x12  // First 10-bits for write address of Microcode
#define CBNUNPU_CSR	     0x16  // First 10-bits for write address of Microcod

#define CBNU_DRAM_BASE_ADDR  0x887000000 // This is DRAM base address for CBNU NPU
#define CBNU_DRAM_IMAGE_ADDR 0x887000000 // We will store our image from this address

void main(void)
{

	uint32_t fd;
	uint32_t fd2;
	uint32_t * npu_base;
	uint64_t * image_base;
	uint64_t * weights_base;


	printf("--------------------------------------------------\n");
	printf("    CBNU NPU Test\n");
	printf("--------------------------------------------------\n");
	if((fd = open("/dev/mem", O_RDWR | O_SYNC)) == -1){
		printf("/dev/mem open Fail \n");
		exit(1);
	}

 npu_base = mmap(0, 0x100, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0x8a010000);
        if(npu_base == MAP_FAILED){
		printf("npu_base error\n");
		exit(1);
	}	

	image_base = mmap(0, 0x10000, PROT_READ | PROT_WRITE, MAP_SHARED, fd, CBNU_DRAM_BASE_ADDR);
	if(image_base == MAP_FAILED){
	        printf("weights_base error\n");
	        exit(1);
	}

	 uint64_t data[4];
	 data[0] = image_base[0]; printf("%" PRIu64 " \n", image_base[0]);
	 data[1] = image_base[1]; printf("%" PRIu64 " \n", image_base[1]); 
	 data[2] = image_base[2]; printf("%" PRIu64 " \n", image_base[2]);
	 data[3] = image_base[3]; printf("%" PRIu64 " \n", image_base[3]); 

    char *filename = "test.txt";

    // open the file for writing
    FILE *fp = fopen(filename, "w");
    if (fp == NULL)
    {
        printf("Error opening the file %s", filename);
        exit(1);
    }
    // write to the text file
    for (int i = 0; i < 10; i++)
        fprintf(fp, "This is the line #%ld\n", data[i]);

    // close the file
    fclose(fp);


	FILE *myfile;
	myfile = fopen("microcode.txt","r");
	uint32_t numberArray[15];
	uint32_t MC_READ[15];
	int i;

	for (i = 0; i < 15; i++)
	{
	 	fscanf(myfile, "%d", &numberArray[i]);
	}

	for (i = 0; i < 2; i++)
	{
		printf("First 2 lines of Microcode is: %d\t, Number in HEX is: %x\n",numberArray[i], numberArray[i]);
	}

	printf("\n\n------------------------------------------------------------------------------------\n\n");

	for (i = 13; i < 15; i++)
	{
	   printf("Last 5 lines of Microcode is: %d\t, Number in HEX is: %x\n",numberArray[i], numberArray[i]);
	}


	npu_base[CBNUNPU_YOLO_START] = (unsigned int)0x00000002;

	for( uint32_t i=0; i<(sizeof(numberArray)/sizeof(numberArray[0])); i++ ) 
	{
	  npu_base[CBNUNPU_WR_MC_DATA>>2] = numberArray[i];
	  npu_base[CBNUNPU_WR_MC_ADDR>>2] = i;
	}

	npu_base[CBNUNPU_YOLO_START] = (unsigned int)0x00000001;
	npu_base[CBNUNPU_YOLO_START] = (unsigned int)0x00000000;

	// MICROCODE READ OPERATION STARTS
	npu_base[CBNUNPU_YOLO_START] = (unsigned int)0x00000004;
	for( uint32_t i=0; i<15; i++ ) 
	{
	  npu_base[CBNUNPU_WR_MC_ADDR>>2] = i;
	  MC_READ[i] = npu_base[CBNUNPU_RD_MC_DATA>>2];
      printf("Microcode Line Number: %d\t, Value is: %x\n",i, MC_READ[i]);
	}

    // Wait for the Finish Signal
    printf("\n\n\n Waiting for the Finish Signal:");
	uint32_t read_finish;
	uint64_t FMAP_READ[4];
	int k=0;
	while(k<=1)
	{
  read_finish= npu_base[CBNUNPU_CSR>>2];
  if (read_finish==0x10080402)
  {
	printf("\n\n\n FMAP Read back Values \n\n\n");
	 for( uint64_t i=0;i<3;i++)
	    {
		  FMAP_READ[i] = image_base[i];
		  printf("%" PRIu64 " \n", image_base[i]);
  	    }
        k=2;
  }
    char *filename1 = "test1.txt";

    // open the file for writing
    FILE *fp = fopen(filename1, "w");
    if (fp == NULL)
    {
        printf("Error opening the file %s", filename1);
        exit(1);
    }
    // write to the text file
    for (int i = 0; i < 10; i++)
	{
		fprintf(fp, "This is the line #%ld\n", FMAP_READ[i]);
	}
        
    // close the file
    fclose(fp);
	} 
}

		                     

