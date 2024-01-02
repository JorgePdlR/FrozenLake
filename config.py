
# Init config to allow verbose print
def init(verbose):
    global vprint
    vprint = print if verbose else lambda *a, **k: None