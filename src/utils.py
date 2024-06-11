from __future__ import division

import platform
import torch
import numpy as np
import subprocess
import random
import imgaug as ia
import matplotlib.pyplot as plt 


Sbox = np.array([99, 124, 119, 123, 242, 107, 111, 197, 48, 1, 103, 43, 254, 215, 171, 118, 202, 130, 201, 125, 250, 89, 71,
        240, 173, 212, 162, 175, 156, 164, 114, 192, 183, 253, 147, 38, 54, 63, 247, 204, 52, 165, 229, 241, 113, 216,
        49, 21, 4, 199, 35, 195, 24, 150, 5, 154, 7, 18, 128, 226, 235, 39, 178, 117, 9, 131, 44, 26, 27, 110, 90, 160,
        82, 59, 214, 179, 41, 227, 47, 132, 83, 209, 0, 237, 32, 252, 177, 91, 106, 203, 190, 57, 74, 76, 88, 207, 208,
        239, 170, 251, 67, 77, 51, 133, 69, 249, 2, 127, 80, 60, 159, 168, 81, 163, 64, 143, 146, 157, 56, 245, 188,
        182, 218, 33, 16, 255, 243, 210, 205, 12, 19, 236, 95, 151, 68, 23, 196, 167, 126, 61, 100, 93, 25, 115, 96,
        129, 79, 220, 34, 42, 144, 136, 70, 238, 184, 20, 222, 94, 11, 219, 224, 50, 58, 10, 73, 6, 36, 92, 194, 211,
        172, 98, 145, 149, 228, 121, 231, 200, 55, 109, 141, 213, 78, 169, 108, 86, 244, 234, 101, 122, 174, 8, 186,
        120, 37, 46, 28, 166, 180, 198, 232, 221, 116, 31, 75, 189, 139, 138, 112, 62, 181, 102, 72, 3, 246, 14, 97,
        53, 87, 185, 134, 193, 29, 158, 225, 248, 152, 17, 105, 217, 142, 148, 155, 30, 135, 233, 206, 85, 40, 223, 140,
        161, 137, 13, 191, 230, 66, 104, 65, 153, 45, 15, 176, 84, 187, 22])


HW_byte = np.array([0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2,
            3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3,
            3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3,
            4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4,
            3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5,
            6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4,
            4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5,
            6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8])


def coeff(traces, labels):
    nb_features = traces.shape[0]
    # print(nb_features)
    co = np.zeros(nb_features)
    for i in range(0, nb_features):
        co[i] = np.corrcoef(traces[i], labels)[0][1]
    return co


def provide_determinism(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ia.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def worker_seed_set(worker_id):
    # See for details of numpy:
    # https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562
    # See for details of random:
    # https://pytorch.org/docs/stable/notes/randomness.html#dataloader

    # NumPy
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    np.random.seed(ss.generate_state(4))

    # random
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)


def print_environment_info():
    """
    Prints infos about the environment and the system.
    This should help when people make issues containg the printout.
    """

    print("Environment information:")

    # Print OS information
    print(f"System: {platform.system()} {platform.release()}")

    # Print poetry package version
    try:
        print(f"Current Version: {subprocess.check_output(['poetry', 'version'], stderr=subprocess.DEVNULL).decode('ascii').strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Not using the poetry package")

    # Print commit hash if possible
    try:
        print(f"Current Commit Hash: {subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("No git or repo found")


def get_parameter_number(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f'the number of netwrok parameters: {total_params}, trainable parameters: {trainable_params}'


def to_cpu(tensor):
    return tensor.detach().cpu()


def hamming_weight(n):
    count = 0
    while n > 0:
        count += n & 1
        n >>= 1
    return count


def sr2se(sr):
    '''(ps,pr) to (ps,pe)
    '''
    se = sr.new(sr.shape)
    se[..., 0] = sr[..., 0] - sr[..., 1]/2
    se[..., 1] = sr[..., 0] + sr[..., 1]/2
    return se


def sr2se_np(sr):
    se = np.zeros_like(sr)
    se[..., 0] = sr[..., 0] - sr[..., 1]/2
    se[..., 1] = sr[..., 0] + sr[..., 1]/2
    return se


def nms(AOIs:torch.Tensor, conf:torch.Tensor, mask:torch.Tensor, thres_nms:float):
    """ Non-Maximum Suppression
    Input:
    - AOIs: All predicted AOIs of a target traces (nl, 2)
    - conf: confidences for each AOI (nl,)
    - mask: condition of solutions (nl,)
    - thres_nms: nms filtering threshold

    Output:
    - mask: mask of solutions that indicating the remaining solutions 

    """
    # print(AOIs.dtype, conf.dtype, mask.dtype) # torch.float32 torch.float32 torch.bool
    ditch = torch.tensor([]).bool().to(AOIs.device)
    conf[~mask] = 0 # exclude those that already being rule out 
    idxs = conf.argsort()[mask.sum().item():]
    # print(idxs)
    while idxs.numel() > 0:  
        max_conf_index = idxs[-1]
        max_conf_AOI = AOIs[max_conf_index][None, :]
        if idxs.size(0) == 1:
            break
        idxs = idxs[:-1]  
        other_AOIs = AOIs[idxs]
        ious = compute_iou(max_conf_AOI.repeat(other_AOIs.size(0),1).t(), other_AOIs.t())
        ditch = torch.cat([ditch, idxs[ious > thres_nms]], dim=0)
        idxs = idxs[ious <= thres_nms] # Redundent AOIs are removed.
    # print(ditch)
    mask[ditch] = False # mark the exclued solutions
    conf[~mask] = 0
    return mask


def solution_filtering(prediction:torch.Tensor, 
                       thres_conf:float=0.75,
                       thres_nms:float=0.45, 
                       max_det:int=16
                       ):
    """Performs Non-Maximum Suppression (NMS) on inference results  

    Input:
    - prediction: bs*nl*(3+nbclass)  (sigmoid(ps_t), pr_t, conf, cls_pred)
    - thres_conf: first filtering 
    - thres_nms: second filtering: nms 
    - max_det: last filtering

    Returns:
         solution_mask with shape: (bs, nl)
    """
    bs, nl ,_ = prediction.shape
    device = prediction.device
    # to abosulution coordination
    prediction[..., 0] += torch.arange(nl).repeat((bs, 1)).to(device)
    prediction[..., 1] = torch.exp(prediction[..., 1])
    prediction[..., 3:] = torch.nn.functional.softmax(prediction[..., 3:], dim=1)

    solution_mask = torch.ones([bs, nl]).bool().to(prediction.device)

    # Apply the first filter
    solution_mask[prediction[..., 2] < thres_conf] = False

    for xi, x in enumerate(prediction):  # trace index, solutions for this traces
        n = solution_mask[xi].sum().item()  # number of solutions after first filtering

        if not n:  # no predicted AOI
            continue
        AOIs, conf = x[:, :2] , x[:, 2]
        # Second filtering: NMS
        solution_mask[xi] = nms(AOIs, conf, solution_mask[xi], thres_nms)

        # Last filtering: limit detections
        if solution_mask[xi].sum().item() > max_det:
            idxs = conf.argsort(descending=True)[max_det:] # the ditched solutions' index
            solution_mask[xi, idxs] = False # mark those that already being rule out 

    return solution_mask


def select_cls_pred(AOI_target, pred, pred_mask, thres_overlap):
    '''Select the predtion of class on top of detected AOI. 
    IoU(AOI_target, pred[AOI_target_mask, :2]) > thres_overlap
    -------------------
    Input:
    - AOI_target: 
    - pred: the prediction of network
    - pred_mask: the mask of prediction solutions
    - scale:
    - thres_overlap: 
    
    Output:
    - pred_solution: the solutions that corresponds to target AOIs.
    - idx_solution: the idex of solutions that corresponds to target AOIs.
    '''
    # print(AOI_target.shape, pred.shape, pred_mask.shape)
    pred_solution = []
    idx_solution = []
    for idx, aoi in enumerate(AOI_target):
        pred_trs = pred[idx, pred_mask[idx]] #(nbsolution, 3+nbclass)
        if len(pred_trs)==0:
            continue
        iou = compute_iou_np(aoi.reshape(1,2).repeat(len(pred_trs), axis=0).T, pred_trs[:,:2].T)
        # print(iou, iou.max())
        if iou.max() > thres_overlap:
            maxi = np.argmax(iou)
            # print(maxi)
            idx_solution.append(idx)
            pred_solution.append(pred_trs[maxi])
    return np.array(pred_solution), np.array(idx_solution)
    

def compute_iou(AOI1:torch.Tensor, AOI2:torch.Tensor, pspe=False, eps=1e-16):
    '''Compute the Intersection over Unoin (IoU) for AOI1 and AOI2. 
    
    Input:
    - AOI1 and AOI2: Two AOIs. shape: (2, bs)
    - pspe: indicatior of the representation of AOI coordination system.
    - eps:

    Output: the IoU value between AOI1 and AOI2. (1, bs)
    '''

    assert AOI1.shape==AOI2.shape, f'Illegal shape for AOI1{AOI1.shape} and AOI2{AOI2.shape}'
    # Get the coordinates of AOI
    if pspe:  # ps, pe = AOI
        b1_ps, b1_pe = AOI1[0], AOI1[1]
        b2_ps, b2_pe = AOI2[0], AOI2[1]
    else:  # transform from pspr to pspe
        b1_ps, b1_pe = AOI1[0] - AOI1[1]/2, AOI1[0] + AOI1[1]/2
        b2_ps, b2_pe = AOI2[0] - AOI2[1]/2, AOI2[0] + AOI2[1]/2

    # Intersection Area
    inter = (torch.min(b1_pe, b2_pe) - torch.max(b1_ps, b2_ps)).clamp(0) 
    # Union Area
    union = (b1_pe - b1_ps) + (b2_pe - b2_ps) - inter + eps

    return inter / union


def compute_iou_np(AOI1:np.ndarray, AOI2:np.ndarray, pspe=False, eps=1e-16):
    '''Compute the Intersection over Unoin (IoU) for AOI1 and AOI2. 
    
    Input:
    - AOI1 and AOI2: Two AOIs. shape: (2, bs)
    - pspe: indicatior of the representation of AOI coordination system.
    - eps:

    Output: the IoU value between AOI1 and AOI2. (1, bs)
    '''

    assert AOI1.shape == AOI2.shape, f'Illegal shape for AOI1{AOI1.shape} and AOI2{AOI2.shape}'
    
    if pspe:
        b1_ps, b1_pe = AOI1[0], AOI1[1]
        b2_ps, b2_pe = AOI2[0], AOI2[1]
    else:
        b1_ps, b1_pe = AOI1[0] - AOI1[1]/2, AOI1[0] + AOI1[1]/2
        b2_ps, b2_pe = AOI2[0] - AOI2[1]/2, AOI2[0] + AOI2[1]/2

    inter = (np.minimum(b1_pe, b2_pe) - np.maximum(b1_ps, b2_ps)).clip(0)
    union = (b1_pe - b1_ps) + (b2_pe - b2_ps) - inter + eps

    return inter / union


def compute_intersection(AOI1:torch.Tensor, AOI2:torch.Tensor, pspe=False, base=1):
    '''Compute the percentage of base AOI that overlap with the other. 
    
    Input:
    - AOI1 and AOI2: Two AOIs. shape: (2, bs)
    - pspe: indicatior of the representation of AOI coordination system.
    - base: default as 1, that means AOI1 is the base.

   Output: the IoU value between AOI1 and AOI2. (1, bs)
    '''

    assert AOI1.shape==AOI2.shape, print('Illegal shape for AOI1 and AOI2')
    # Get the coordinates of AOI
    if pspe:  # ps, pe = AOI
        b1_ps, b1_pe = AOI1[0], AOI1[1]
        b2_ps, b2_pe = AOI2[0], AOI2[1]
    else:  # transform from pspr to pspe
        b1_ps, b1_pe = AOI1[0] - AOI1[1]/2, AOI1[0] + AOI1[1]/2
        b2_ps, b2_pe = AOI2[0] - AOI2[1]/2, AOI2[0] + AOI2[1]/2

    # Intersection Area
    inter = (torch.min(b1_pe, b2_pe) - torch.max(b1_ps, b2_ps)).clamp(0) 
    one = b1_pe-b1_ps if base==1 else b2_pe-b2_ps
    return inter/one


## Evaluation demonstration
def plot_guessing_entropy(preds, plaintext, mask, real_key, label_method='ID', savefile=None):
    """
    - preds : the probability for each class (n*256 for a byte, n*9 for Hamming weight)
    - real_key : the secret key byte
    - model_flag : a string for naming GE result
    """
    num_averaged = 100
    # print(preds.shape, plaintext.shape)
    nb_trs_max = preds.shape[0]
    step = 1
    if nb_trs_max > 500 and nb_trs_max < 1000:
        step = 2
    if nb_trs_max >= 1000 and nb_trs_max < 5000:
        step = 4
    if nb_trs_max >= 5000 and nb_trs_max < 10000:
        step = 5
    
    guessing_entropy = np.zeros((num_averaged, int(nb_trs_max/step)))
    
    # attack multiples times for average
    for time in range(num_averaged):
        # select the attack traces randomly
        random_index = list(range(plaintext.shape[0]))
        random.shuffle(random_index)
        random_index = random_index[0:nb_trs_max]

        # initialize score matrix
        score_mat = np.zeros((nb_trs_max, 256))
        for key_guess in range(0, 256):
            for i in range(0, nb_trs_max):
                sout = Sbox[plaintext[random_index[i]] ^ key_guess]
                sout = sout ^ mask[random_index[i]] if mask is not None else sout
                label = sout if label_method == 'ID' else HW_byte[sout]
                score_mat[i, key_guess] = preds[random_index[i], label]
        score_mat = np.log(score_mat + 1e-40)
        for i in range(0, int(nb_trs_max/step)):
            log_likelihood = np.sum(score_mat[0:i*step+1,:], axis=0)
            ranked = np.argsort(log_likelihood)[::-1]
            guessing_entropy[time,i] = list(ranked).index(real_key)

    guessing_entropy = np.mean(guessing_entropy, axis=0)
    plt.figure()
    x = np.arange(0, nb_trs_max, step)+1
    plt.plot(x, guessing_entropy[0:int(nb_trs_max/step)], color='r', label='GE')
    idx = np.where(guessing_entropy[0:int(nb_trs_max/step)]< 1)[0]
    idx = idx[0] if len(idx)!=0 else -1
    if idx != -1:
        plt.axvline(x=idx+1, linestyle='--', color='b', label=f'#traces for GE<1: {idx+1}')
    plt.xlabel('Number of trace')
    plt.ylabel('Guessing entropy')
    plt.tight_layout()
    plt.grid(True)
    plt.legend()
    if savefile:
        plt.savefig(f'./output/GE_{savefile}.eps', format='eps')
    else:
        plt.show()

    return {'Nt':x,'GE':guessing_entropy[0:int(nb_trs_max/step)]}



def plot_instance_result(traces, AOI_pred, AOI_mask, AOI_target, startat, savefile:str=None):
    '''Plot predicted AOIs for trace instances along with the target AOIs.
    ----------------
    Input:
    - trace: trace instances.
    - AOI_pred: Network preditions.
    - AOI_mask: filtering mask of prediction solutions
    - AOI_target: target AOI for each trace instance.
    - startat: offset of trace.
    - savefile:
    '''
    m = len(traces) # number of trace instances
    fig, axes = plt.subplots(m, 1, sharex=True, sharey=False, figsize=(10, 3))
    for i in range(m):
        trace = traces[i].reshape(-1)
        axes[i].plot(np.arange(len(trace))+startat, trace)
        if AOI_target is not None:
            target = AOI_target[i]
        pred = AOI_pred[i, AOI_mask[i]]

        mmax, mmin = np.max(trace), np.min(trace)
        mmax, mmin = mmax+10, mmin-10
        if target is not None:
            ps, pr, _ = target
            ps += startat
            axes[i].plot([ps, ps+pr],[mmax,mmax], color='g')
            axes[i].plot([ps, ps+pr],[mmin,mmin], color='g')
            axes[i].plot([ps, ps],[mmax,mmin], color='g')
            axes[i].plot([ps+pr, ps+pr],[mmax,mmin], color='g')
        # print(pred.shape)
        for _, aoi in enumerate(pred):
            ps_t, pr_t = aoi[:2]
            cls_pred = aoi[3:]
            ps_t += startat
            pe_t = ps_t + pr_t
            
            axes[i].plot([ps_t, pe_t],[mmax,mmax], linestyle='-.', color='r')
            axes[i].plot([ps_t, pe_t],[mmin,mmin], linestyle='-.', color='r')
            axes[i].plot([ps_t, ps_t],[mmax,mmin], linestyle='-.', color='r')
            axes[i].plot([pe_t, pe_t],[mmax,mmin], linestyle='-.', color='r')
            if target is not None:
                ps, pr, cls_target = target
                pe = ps + pr
                inter = min(pe, pe_t)-max(ps, ps_t)
                inter = 0 if inter<0 else inter
                union = (pe-ps+pe_t-ps_t) - inter
                IoU = 0 if union==0 else inter/union
                if IoU>0.8:
                    axes[i].text(pe_t+5, mmin+5, 'IoU:{:.2f} rank:{:3d}'.format(IoU, (cls_pred>cls_pred[cls_target]).sum()), color='r')

    fig.supxlabel('Time')
    fig.supylabel('Power/EM')
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    if savefile:
        plt.savefig(f'./output/instance_{savefile}.eps', format='eps')
    else:
        plt.show()


def plot_pred_AOI_freq(AOIs, mask, trace_length, startat, savefile:str=None):
    '''Plot a figure of detected AOIs frequency
    --------------
    Input:
    - AOIs: ndarray with shape (bs, nl, 2)
    - mask: mask of AOIs (bs, nl)
    - trace_length: 
    - startat: offset of traces relative to raw traces. 
    '''
    freq = np.zeros(trace_length)
    for xi, x in enumerate(AOIs):
        pred_AOI = x[mask[xi]]
        for aoi_sr in pred_AOI:
            aoi_se = sr2se_np(aoi_sr)
            px1, px2 = int(aoi_se[0]), int(aoi_se[1])
            freq[px1: px2] += 1.0
    freq = freq/len(AOIs)

    plt.figure(figsize=(12,4))
    plt.plot(np.arange(startat, startat+trace_length), freq, label='frequency', color='r')
    # plt.ylim([0, 1.1])
    plt.axhline(y=0.9, label='freq=0.9', linestyle='--', color='b')
    plt.xlabel('Samples')
    plt.ylabel('Average Frequency of predicted AOI.')
    plt.tight_layout()
    if savefile:
        plt.savefig(f'./output/frequency_{savefile}.eps', format='eps')
    else:
        plt.show()
    return freq

