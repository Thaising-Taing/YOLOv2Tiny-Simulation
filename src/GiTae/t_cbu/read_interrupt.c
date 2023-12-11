#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <linux/kernel.h>
#include "libxdma.h"
#include "libxdma_api.h"
#include "cdev_sgdma.h"
#include "xdma_thread.h"

static irqreturn_t sample_irq(int irq, void *dev_id)
{
printf("irq %d\n", irq);

return IRQ_RETVAL(1);
}


int main(int argc, char **argv)
{
int ret;
    ret =  request_irq(99, sample_irq, 0, "xdma", NULL);
 //   rv = request_irq(vector, xdma_channel_irq, 0, xdev->mod_name,
				// engine);

if (ret==0)
 {
printf(" Result is not good");
}
else
{ printf(" Result is very good");
}

/*
        int source, n;
        unsigned char buffer[4*8192];

        source = open("/proc/interrupts", O_RDONLY);
        for(int i = 0; i < 2; i++)
        {
                n=read(source, buffer, 4*8192);
                buffer[n] = 0;
                printf("%d chars in /proc/interrupts:\n", n);
                printf("%s", buffer);
        }

        close(source);
*/
        return 0;
}
