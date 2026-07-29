import torch
print(f"Torch Version: {torch.__version__}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"Capability: {cap}")
    archs = torch.cuda.get_arch_list()
    print(f"Arch List: {archs}")
    target_arch = f"sm_{cap[0]}{cap[1]}"
    print(f"Target {target_arch} in Arch List: {target_arch in archs}")
    try:
        x = torch.randn(1).to("cuda")
        print(f"Tensor on GPU: SUCCESS, {x}")
    except Exception as e:
        print(f"Tensor on GPU: FAILED, {e}")
else:
    print("CUDA NOT AVAILABLE")
