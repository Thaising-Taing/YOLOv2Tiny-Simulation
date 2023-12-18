#! /usr/bin/env python3

import mmap
import os
import struct
import subprocess
import numpy as np

INTERRUPT_CHANNELS=4
AXI_LITE_MEM_SIZE=0x1000000 # in bytes

class XdmaAccess():
    def __init__(self, device_index):
        user_filename     = "/dev/xdma" + str(device_index) + "_user"
        c2h0_filename     = "/dev/xdma" + str(device_index) + "_c2h_0"
        h2c0_filename     = "/dev/xdma" + str(device_index) + "_h2c_0"
        events_filenames = []
        self.fd_irq = []
        for i in range(16):
            events_filenames.append("/dev/xdma" + str(device_index) + "_events_" + str(i))
            self.fd_irq.append(os.open(events_filenames[i], os.O_RDWR))            
        self.fd_user = os.open(user_filename, os.O_RDWR)
        self.fd_c2h0 = os.open(c2h0_filename, os.O_RDWR)
        self.fd_h2c0 = os.open(h2c0_filename, os.O_RDWR)

        self.mem = mmap.mmap(self.fd_user, int(AXI_LITE_MEM_SIZE))

    def __close__(self):
        self.mem.close()
        os.close(self.fd_user)
        os.close(self.fd_c2h0)
        os.close(self.fd_h2c0)
        for i in range(16):
            os.close(self.fd_irq[i])

    def read(self, addr):
        addr_w_o = addr
        bs = self.mem[addr_w_o : addr_w_o + 4]
        val = struct.unpack("I", bs)[0]
        return np.array([val])

    def write(self, addr, data):
        bs = struct.pack("I", int(data))
        addr_w_o = addr
        self.mem[addr_w_o : addr_w_o + 4] = bs

    def read_bytes(self, addr, length):
        bs = b""
        for i in range(0, length // 4):
            addr_w_o = addr + 4 * i
            b = self.mem[addr_w_o : addr_w_o + 4]
            bs += b
        return np.array([bs[0:length]])

    def read_dma0(self, addr, len_bytes):
        os.lseek(self.fd_c2h0, addr, os.SEEK_SET)
        dma_raw = os.read(self.fd_c2h0, len_bytes)
        data = np.frombuffer(dma_raw, np.uint32)
        # data = np.frombuffer(dma_raw, np.uint16)
        return data
    
    def read_dma(self, addr, len_bytes):
        os.lseek(self.fd_c2h0, addr, os.SEEK_SET)
        dma_raw = os.read(self.fd_c2h0, len_bytes)
        data = np.frombuffer(dma_raw, np.int16 )
        data1 = data * 65536 
        data2 = np.frombuffer(data1, np.float32 )
        
        return data2

    def write_dma(self, addr, data):
        os.lseek(self.fd_h2c0, addr, os.SEEK_SET)
        os.write(self.fd_h2c0, data.tobytes())

    def wait_irq(self, channel):
        # Blocks the code until interrupt is asserted from the FPGA
        os.read(self.fd_irq[channel],4)
 
