
# Init config to allow verbose print for tracing the algorithm
def init(verbose):
    global vprint
    vprint = print if verbose else lambda *a, **k: None