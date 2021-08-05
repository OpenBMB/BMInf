import cupy
class Context:
    def __init__(self, device_list):
        self.__load_stream = {}
        self.__calc_stream = {}
        for device in device_list:
            with device:
                self.__load_stream[device.id] = cupy.cuda.Stream()
                self.__calc_stream[device.id] = cupy.cuda.Stream()


    @property
    def load_stream(self):
        device_id = cupy.cuda.get_device_id()
        return self.__load_stream[device_id]
    
    @property
    def calc_stream(self):
        device_id = cupy.cuda.get_device_id()
        return self.__calc_stream[device_id]
    
    def __sync(self, items, stream):
        if stream is None:
            stream = cupy.cuda.get_current_stream()
        
        events = []
        for idx, stream in items:
            with cupy.cuda.Device(idx):
                events.append( stream.record() )
        
        for event in events:
            stream.wait_event(event)

    def sync_calc(self, stream = None):
        self.__sync(self.__calc_stream.items(), stream)
    
    def sync_load(self, stream = None):
        self.__sync(self.__load_stream.items(), stream)
