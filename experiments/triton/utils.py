import triton

DEVICE = triton.runtime.driver.active.get_active_torch_device()
