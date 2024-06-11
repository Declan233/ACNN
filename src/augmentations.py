import numpy as np
import random

def addClockJitter(traces, pspe, trace_length, clock_range=1):
    '''add Clock Jitter to traces.

    input: 
    - traces: input traces to be desynced, shape (nb_trs, _).
    - pspe: Start and end index of original area of interest(AOI).
    - trace_length
    - clock_range: default set as 1.

    output: 
    - new_pspe_list: Start and end index of new AOI.
    - output_traces: desynced traces, shape (nb_trs, trace_length).
    '''

    ps, pe = pspe
    nb_trs, _ = traces.shape
    new_pspe_list = []
    output_traces = []
    for trace_idx in range(nb_trs):
        trace = traces[trace_idx]
        point = 0 # pointer to the original trace
        offset = 0 # use to track the shift of ps, pe
        xset, yset = False, False
        new_trace = []
        new_pspe = []
        while point < len(trace)-1:
            new_trace.append(trace[point])
            r = random.randint(-clock_range, clock_range)# generate a random number
            if r <= 0:  # if r < 0, delete r point afterward
                point += abs(r)
            else:  # if r > 0, add r point afterward
                avg_point = (trace[point] + trace[point+1])/2
                for _ in range(r):
                    new_trace.append(avg_point)
            if (not xset) and point >= ps: # record the new ps
                xset = True
                if ps+offset > trace_length:
                    break
                new_pspe.append(ps+offset)
            if (not yset) and point >= pe: # record the new pe
                yset = True
                if offset >= trace_length-pe:
                    new_pspe.append(trace_length-1)
                else:
                    new_pspe.append(pe+offset)
            offset += r
            point += 1
        output_traces.append(new_trace)
        new_pspe_list.append(new_pspe)
    return np.array(new_pspe_list).reshape(-1,2), regulateMatrix(output_traces, trace_length)


### A function to make sure the traces has same length (padding zeros)
def regulateMatrix(M, size):
    #maxlen = max(len(r) for r in M)
    maxlen = size
    Z = np.zeros((len(M), maxlen))
    for enu, row in enumerate(M):
        if len(row) <= maxlen:
            Z[enu, :len(row)] += row
        else:
            Z[enu, :] += row[:maxlen]
    return Z


def addDesync(traces, max_desync, left=True):
    '''Desync traces, padding with 0.

    input: 
    - traces: input traces to be desynced, shape (nb_trs, len_trs).
    - max_desync: maximun desync points.
    - left: Direction of desynchronization, left is True else right. Default: True.

    output: 
    - offset: offset list indicating desync points for each tarce.
    - output_traces: desynced traces, shape (nb_trs, len_trs).
    '''
    output_traces = np.zeros_like(traces)
    nb_trs, len_trs = traces.shape
    offset = np.random.randint(max_desync, size=nb_trs)
    if left:
        for i in range(nb_trs):
            output_traces[i, :len_trs-offset[i]] = traces[i, offset[i]:]
    else:
        for i in range(nb_trs):
            output_traces[i, offset[i]:] = traces[i, :len_trs-offset[i]]

    return offset, output_traces


def addGaussianNoise(traces, max_noise_level):
    '''Add GaussianNoise
    input: 
    - traces: input traces to be desynced, shape (nb_trs, len_trs).
    - max_noise_level: maximun noise level.
    output: 
    - nl: moise level list indicating the nosie added to each tarce.
    - output_traces: desynced traces, shape (nb_trs, len_trs).
    '''
    if max_noise_level == 0:
        return traces
    else:
        nb_trs, len_trs = traces.shape
        output_traces = np.zeros_like(traces)
        nl = np.random.randint(max_noise_level, size=nb_trs)
        for ti in range(len(traces)):
            profile_trace = traces[ti]
            noise = np.random.normal(0, nl[ti], size=len_trs)
            output_traces[ti] = profile_trace + noise
        return nl, output_traces


