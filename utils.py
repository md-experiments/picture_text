import itertools
import copy
import datetime

def flatten_list(res):
    res=copy.deepcopy(res)
    return list(itertools.chain.from_iterable(res))

class TimeClass():
    def __init__(self):
        self.t0=datetime.datetime.now()
        self.times=[self.t0]
    def take(self):
        """
        Takes time, but adding another point to the timeline and 
        returning the time since last measure in seconds & minutes
        >>> t = TimeClass()
        >>> time.sleep(1)
        >>> t.take()
        (1, 0)
        """
        self.t1=datetime.datetime.now()
        delta_secs=(self.t1-self.t0).seconds
        delta_mins=delta_secs//60
        self.times.append(self.t1)
        self.t0=datetime.datetime.now()
        return delta_secs, delta_mins