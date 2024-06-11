import numpy as np


def traces_random_start(traces, base:int, l:int, max_offset:int):
    '''Sample traces with l points but random starting points

    input: 
    - traces: input raw traces.
    - base: base starting points.
    - l: lenth of sampled traces.
    - max_offset: the max offset to the base.

    output: 
    - offset: offset list indicating offset points for each tarce.
    - output_traces: desynced traces, shape (nb_trs, l).
    '''
    # print(traces.shape, base, l, max_offset)
    n_trs = len(traces)
    offset = np.random.randint(-max_offset, max_offset+1, size=n_trs)
    start = base+offset
    output_traces = np.zeros([n_trs, l])
    for idx in range(n_trs):
        output_traces[idx] = traces[idx, start[idx]:start[idx]+l]

    return offset, output_traces

def traces_raw(traces, base:int, l:int):
    '''Sample traces with l points but random starting points

    input: 
    - traces: input raw traces.
    - base: base starting points.
    - l: lenth of sampled traces.

    output: 
    - output_traces: desynced traces, shape (nb_trs, l).
    '''
    # print(traces.shape, base, l, max_offset)
    output_traces = traces[:, base:base+l]

    return output_traces
