import traceback

def round_up(x, d):
    return (x + d - 1) // d * d

def get_traceback(pos=None):
    if pos is None:
        return "\n\n".join(traceback.format_stack())
    return traceback.format_stack(limit=10)[pos]