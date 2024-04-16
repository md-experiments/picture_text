import itertools
import copy
import datetime
import time
    
from hashlib import md5
import traceback
import itertools

def flatten_list(res):
    """
    Flattens a list of lists

    Args:
        res (list): list of lists
    Returns:
        flattened deepcopy of the list
    """
    res=copy.deepcopy(res)
    return list(itertools.chain.from_iterable(res))

class TimeClass():
    """
    Time class to take note of starting time of an experiment and then take additional times e.g. when steps finish
    """
    def __init__(self):
        """
        Initializes class. Takes current time and starts the list of timepoints of time-taking
        """
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


def log_traceback():
    return traceback.format_exc()

def hash_text(t):
    return md5(str(t).encode()).hexdigest()