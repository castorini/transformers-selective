import os
import numpy as np


def rcc_auc(conf, risk, handle=None, args=None):
    # risk-coverage curve's area under curve
    n = len(conf)
    cr_pair = list(zip(conf, risk))
    cr_pair.sort(key=lambda x: x[0], reverse=True)

    cumulative_risk = [cr_pair[0][1]]
    for i in range(1, n):
        cumulative_risk.append(cr_pair[i][1] + cumulative_risk[-1])

    points_x = []
    points_y = []
    if handle is not None:
        save_fname = os.path.join(
            args.plot_data_dir,
            'saved_data',
            os.path.basename(args.data_dir),
            args.model_name_or_path + f'-{handle}-rcc.npy'
        )
        if not os.path.exists(save_fname):
            os.makedirs(os.path.dirname(save_fname), exist_ok=True)

    auc = 0
    for k in range(n):
        auc += cumulative_risk[k] / (1+k)
        points_x.append((1+k) / n)  # coverage
        points_y.append(cumulative_risk[k] / (1+k))  # current avg. risk

    if handle is not None:
        np.save(save_fname, [points_x, points_y])

    return auc


def rpp(conf, risk):
    # reverse pair proportion
    # for now only works when risk is binary
    n = len(conf)
    cr_pair = list(zip(conf, risk))
    cr_pair.sort(key=lambda x: x[0], reverse=False)

    pos_count, rp_count = 0, 0
    for i in range(n):
        # print(f'{i}\t{cr_pair[i][1]==0}\t{pos_count}\t{rp_count}')
        if cr_pair[i][1] == 0:  # risk==0
            pos_count += 1
        else:
            rp_count += pos_count

    return rp_count / (n**2)
