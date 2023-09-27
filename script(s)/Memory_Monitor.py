import torch
import psutil
import tracemalloc

class _memory_monitor:
    unit = {
        #  "B": 1024 ** 0,
        # "kB": 1024 ** 1,
        "MB": 1024 ** 2,
        # "GB": 1024 ** 3,
        # "TB": 1024 ** 4,
        }
    
    def __enter__(self):
        # return # skip
        tracemalloc.start(100)
        torch.cuda.synchronize()
        self.trc_mem, self.trc_mem_peak = tracemalloc.get_traced_memory()
        self.ram_mem = psutil.virtual_memory()
        self.gpu_mem = torch.cuda.memory_allocated()
        
    def __exit__(self, exc_type, exc_value, exc_traceback):
        # return # skip
        exc_type, exc_value, exc_traceback # dummy line just to skip the unused warning
        trc_mem, trc_mem_peak = tracemalloc.get_traced_memory()
        ram_mem = psutil.virtual_memory()
        gpu_mem = torch.cuda.max_memory_allocated()
        
        for unit, divisor in _memory_monitor.unit.items():
            msg  = f""
            msg += f"\nRAM Memory"
            msg += f"\n\tTotal     {ram_mem.total / divisor:>10.2f} {unit:>2}"
            msg += f"\n\tAvailable {ram_mem.available / divisor:>10.2f} {unit:>2}"
            msg += f"\n\tUsed      {ram_mem.used / divisor:>10.2f} {unit:>2} ({ram_mem.percent} %)"
            msg += f"\nTrace Memory"
            msg += f"\n\tNow  {trc_mem / divisor:>10.2f} {unit:>2}"
            msg += f"\n\tPeak {trc_mem_peak / divisor:>10.2f} {unit:>2}"
            msg += f"\nGPU"
            msg += f"\n\tVRAM {gpu_mem / divisor:>10.2f} {unit:>2}"
            
            print(f"{msg}")
        print(f"*"*80)
        tracemalloc.stop()

if __name__ == "__main__":
    print(f"Memory Monitor Demo")
    with _memory_monitor():
        print(f"\n\nAllocate a tensor in CPU")
        device = "cpu" # "cuda" or "cpu"
        tensor = torch.zeros(1, 3, 1080, 1920).to(device)
    with _memory_monitor():
        print(f"\n\nAllocate a tensor in GPU")
        device = "cuda" # "cuda" or "cpu"
        tensor = torch.zeros(1, 3, 1080, 1920).to(device)