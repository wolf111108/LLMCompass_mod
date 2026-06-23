class IOModule:
    def __init__(self, bandwidth, latency):
        self.bandwidth = bandwidth
        self.latency = latency


IO_module_dict = {
    # "A100": IOModule(6.4e9, 1e-6),
    "A100": IOModule(2039e9, 1e-6),
    "TPUv3": IOModule(float("inf"), 1e-6),
    "MI210": IOModule(1.6e12, 1e-6),
    "CIM_IO": IOModule(1.49e10, 1e-6),
    "CIM_WU_IO": IOModule(1.49e10, 1e-6)
}
