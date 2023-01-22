from pynvml.smi import nvidia_smi
nvsmi = nvidia_smi.getInstance()
print(nvsmi.DeviceQuery('memory.free, memory.total'))