import time


def make_timer(func):
    def wrapper(*args, **kwargs):
        t1 = time.perf_counter()
        ret_val = func(*args, **kwargs)
        t2 = time.perf_counter()
        print('Time elapsed was', t2 - t1)
        return ret_val
    return wrapper