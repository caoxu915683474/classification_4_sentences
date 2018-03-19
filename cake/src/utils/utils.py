import time

# Decoration wrapper for time profiling
def help_display(information):
    def _decoration(fun):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            print("\nStart %s..." % information)
            res = fun(*args, **kwargs)
            end_time = time.time()
            print("%s execute duration %fs\n" % (information, end_time-start_time))
            return res
        return wrapper
    return _decoration

# Wrapper for catching kwarg not found error.
def get_kwargs(default, *keys, **kwargs):
    try:
        r = kwargs
        for k in keys:
            r = r[k]
        return r
    except (KeyError, TypeError):
        return default
