from time import monotonic

class Profiler:
    registry = dict()
    
    def __init__(self):
        self.start = None
        self.duration = 0.0
        self.avg_duration = 0.0
        self.calls = 0
    
    @classmethod
    def __getitem__(cls, key):
        return cls.registry.get(key, None)

    @classmethod
    def set_property_gpu(cls, gpu):
        cls.gpu = gpu

    @classmethod
    def set_property_detector(cls, detector):
        cls.detector = detector

    @classmethod
    def set_property_dataset(cls, dataset):
        cls.dataset = dataset

    @classmethod
    def start(cls, name):
        # If the name does not exist already in the registry
        # add it as a new class instance
        if name not in cls.registry.keys():
            cls.registry[name] = cls()

        # Set the start time
        cls.registry[name].start = monotonic()

    @classmethod
    def end(cls, name):
        assert name in cls.registry.keys(), "Profiler name not in registry"

        # Set the duration
        cls.registry[name].duration = (monotonic() - cls.registry[name].start)*1000
        
        # Set the number of calls
        cls.registry[name].calls += 1

        # Set the new average duration
        cls.registry[name].avg_duration += cls.registry[name].duration / cls.registry[name].calls

    @classmethod
    def report(cls):
        print("===== Average duration report =====")
        for r in cls.registry.keys():
            print("{}:\t\t{} ms".format(r, round(cls.registry[r].avg_duration, 2)))
        print("===================================")
        
        # Write to file
        with open("profiler/{}_{}_{}.txt".format(cls.dataset, cls.detector, cls.gpu), "w") as f:
            f.write("computation, duration\n")
            for r in cls.registry.keys():
                f.write("{}, {}\n".format(r, round(cls.registry[r].avg_duration, 2)))
