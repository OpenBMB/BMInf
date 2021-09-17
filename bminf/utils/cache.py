import logging
logger = logging.getLogger(__name__)
class HandleCache:
    def __init__(self, cache_size = 8) -> None:
        self.cache_size = cache_size
        self.t = 0
        self.lru_counter = {}
        self.cache = {}
    
    def __call__(self, *args):
        self.t += 1
        if args in self.cache:
            logger.debug("Get %s HIT" % str(args))
            self.lru_counter[args] = self.t
            return self.cache[args]
        logger.debug("Get %s Missing" % str(args))
        v = self.create(*args)
        if len(self.cache) >= self.cache_size:
            mn_kw = None
            mn_vl = None
            for kw in self.cache.keys():
                if mn_vl is None or mn_vl > self.lru_counter[kw]:
                    mn_vl = self.lru_counter[kw]
                    mn_kw = kw
            del self.lru_counter[mn_kw]
            logger.debug("Release %s" % str(mn_kw))
            self.release( self.cache[mn_kw] )
            del self.cache[mn_kw]
        self.cache[args] = v
        self.lru_counter[args] = self.t
        return v
    
    def create(self, *args):
        raise NotImplementedError()
    
    def release(self, x):
        raise NotImplementedError()