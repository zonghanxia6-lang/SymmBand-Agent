import math, copy
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.data import Batch  # ←←←← 就是这行！必须加！

from collections import defaultdict
from typing import Any, Dict, List
import hydra
import pytorch_lightning as pl
from torch_geometric.utils import to_dense_batch
from tqdm import tqdm

from pyxtal.symmetry import search_cloest_wp, Group

from symmcd.common.data_utils import (
    EPSILON, cart_to_frac_coords, mard, lengths_angles_to_volume, lattice_params_to_matrix_torch,
    frac_to_cart_coords, min_distance_sqr_pbc, lattice_ks_to_matrix_torch,
    sg_to_ks_mask, mask_ks, N_SPACEGROUPS)

from symmcd.pl_modules.diff_utils import d_log_p_wrapped_normal
from symmcd.pl_modules.model import build_mlp
MAX_ATOMIC_NUM = 94
NUM_WYCKOFF = 186
NUM_SPACEGROUPS = 231
SITE_SYMM_AXES = 15
SITE_SYMM_PGS = 13
SITE_SYMM_DIM = SITE_SYMM_AXES * SITE_SYMM_PGS
SG_CONDITION_DIM = 397
SG_SYM = {spacegroup: Group(spacegroup) for spacegroup in range(1, 231)}
SG_TO_WP_TO_SITE_SYMM = dict()
for spacegroup in range(1, 231):
    SG_TO_WP_TO_SITE_SYMM[spacegroup] = dict()
    for wp in SG_SYM[spacegroup].Wyckoff_positions:
        wp.get_site_symmetry()
        SG_TO_WP_TO_SITE_SYMM[spacegroup][wp] = wp.get_site_symmetry_object().to_one_hot()


class DiscreteNoise(nn.Module):
    def __init__(self, atom_type_prior, site_symm_prior_per_sg, beta_scheduler, P_ss, P_a):
        super().__init__()
        self.beta_scheduler = beta_scheduler
        self.site_symm_prior_per_sg = site_symm_prior_per_sg
        self.atom_type_prior = atom_type_prior
        self.P_ss = P_ss
        self.P_a = P_a
        self.site_symm_pgs = SITE_SYMM_PGS
        self.site_symm_axes = SITE_SYMM_AXES
        self.max_atomic_num = MAX_ATOMIC_NUM

    def ss_to_sections(self, ss):
        return [ss[..., i * self.site_symm_pgs:(i + 1) * self.site_symm_pgs] for i in range(self.site_symm_axes)]

    def reshape_ss(self, ss):
        return ss.reshape(-1, SITE_SYMM_AXES, self.site_symm_pgs)

    def multiply_block_diagonal(self, Qs, d):
        '''
        Multiply each matrix Qi in Qs with the corresponding block of d
        Qs: list of D matrices Qi, each Qi is of shape (ni, ni)
        d:  vector of length sum(ni)
        returns: vector of length sum(ni)
        '''
        outs = []
        idx = 0
        for Qi in Qs:
            ni = Qi.shape[-1]
            outs.append(d[..., idx:idx + ni] @ Qi)
            idx += ni
        return torch.cat(outs, -1)

    def q_t(self, P, t):
        alpha = self.beta_scheduler.alphas[t]
        num_classes = P.shape[-1]
        return alpha.view(-1, 1, 1) * torch.eye(num_classes, device=P.device) + (1 - alpha.view(-1, 1, 1)) * P

    def q_t_atom(self, t):
        return self.q_t(self.P_a, t)

    def q_t_ss(self, t, sgs):
        return [self.q_t(self.P_ss[i][sgs], t) for i in range(len(self.P_ss))]

    def q_t_bar(self, P, t):
        alpha_bar = self.beta_scheduler.alphas_cumprod[t]
        num_classes = P.shape[-1]
        return alpha_bar.view(-1, 1, 1) * torch.eye(num_classes, device=P.device) + (1 - alpha_bar.view(-1, 1, 1)) * P

    def q_t_bar_atom(self, t):
        return self.q_t_bar(self.P_a, t)

    def q_t_bar_ss(self, t, sgs):
        return [self.q_t_bar(self.P_ss[i][sgs], t) for i in range(len(self.P_ss))]

    def sigma_sqr_ratio(self, s_int, t_int):
        return self.beta_scheduler.alphas_cumprod[t_int] / self.beta_scheduler.alphas_cumprod[s_int]

    def apply_atom_noise(self, atom_type, t):
        Q_t_bar = self.q_t_bar_atom(t)
        prob_atom_types = atom_type @ Q_t_bar
        return prob_atom_types

    def apply_site_symm_noise(self, site_symm, t, sgs):
        Q_t_bars = self.q_t_bar_ss(t, sgs)
        prob_site_symms = self.multiply_block_diagonal(Q_t_bars, site_symm)
        return prob_site_symms

    def sample_atom_types(self, atom_probs):
        return F.one_hot(torch.multinomial(atom_probs, 1).reshape(-1), self.max_atomic_num).float()

    def sample_site_symms(self, site_symms):
        outs = []
        idx = 0
        for ni in self.ss_lengths:
            outs.append(F.one_hot(torch.multinomial(site_symms[..., idx:idx + ni], 1).reshape(-1), ni).float())
            idx += ni
        return torch.cat(outs, 1)

    def sample_limit_dist(self, node_mask, sgs):
        """ Sample from the limit distribution of the diffusion process"""
        bs, n_max = node_mask.shape
        a_limit = self.atom_type_prior.expand(bs, n_max, -1)
        U_a = a_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max).to(node_mask.device)
        U_a = F.one_hot(U_a, num_classes=a_limit.shape[-1]).float()
        U_a = U_a * node_mask.unsqueeze(-1)

        U_ss_list = []
        for ss_priors_i_per_sg in self.site_symm_prior_per_sg:
            ss_priors_i_per_sg = ss_priors_i_per_sg.to(node_mask.device)
            ss_priors_i = ss_priors_i_per_sg[sgs]
            ss_limit_i = ss_priors_i.unsqueeze(-2).expand(bs, n_max, -1)
            U_ss_i = ss_limit_i.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max).to(node_mask.device)
            U_ss_i = F.one_hot(U_ss_i, num_classes=ss_limit_i.shape[-1]).float()
            U_ss_list.append(U_ss_i)
        U_ss = torch.cat(U_ss_list, dim=-1)

        return U_a, U_ss

    def sample_discrete_features(self, prob_a, prob_ss, node_mask):
        ''' Sample features from multinomial distribution with given probabilities (probX, probE, proby)
            :param prob_a: bs, n, dx_out        node features
            :param prob_ss: bs, n, dx_out        node features
        '''
        bs, n = node_mask.shape
        # The masked rows should define probability distributions as well
        prob_a[~node_mask] = 1 / prob_a.shape[-1]
        prob_ss_list = self.ss_to_sections(prob_ss)
        # prob_ss_norm_list = []
        for i in range(SITE_SYMM_AXES):
            prob_ss_list[i][~node_mask] = 1 / prob_ss_list[i].shape[-1]
        # Flatten the probability tensor to sample with multinomial
        prob_a = prob_a.reshape(bs * n, -1)  # (bs * n, dx_out)
        # Sample a
        atom_t = prob_a.multinomial(1)  # (bs * n, 1)
        atom_t = atom_t.reshape(bs, n)  # (bs, n)
        atom_t = F.one_hot(atom_t, num_classes=prob_a.shape[-1]).float()
        # Sample ss
        site_symm_t_list = []
        for i in range(SITE_SYMM_AXES):
            prob_ss_i = prob_ss_list[i].reshape(bs * n, -1)  # (bs * n, dx_out)
            site_symm_t_i_cat = prob_ss_i.multinomial(1).reshape(bs, n)
            site_symm_t_i = F.one_hot(site_symm_t_i_cat, num_classes=self.site_symm_pgs).float()
            site_symm_t_list.append(site_symm_t_i)
        site_symm_t = torch.cat(site_symm_t_list, -1)

        return atom_t, site_symm_t

    def p_s_and_t_given_0(self, z_t, Qt, Qsb, Qtb):
        """ M: X, E or charges
            Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T for each possible value of x0
            X_t: bs, n, dt
            Qt: bs, d_t-1, dt
            Qsb: bs, d0, d_t-1
            Qtb: bs, d0, dt.
        """
        Qt_T = Qt.transpose(-1, -2)  # bs, dt, d_t-1
        left_term = z_t @ Qt_T  # bs, N, d_t-1
        left_term = left_term.unsqueeze(dim=2)  # bs, N, 1, d_t-1

        right_term = Qsb.unsqueeze(1)  # bs, 1, d0, d_t-1
        numerator = left_term * right_term  # bs, N, d0, d_t-1

        X_t_transposed = z_t.transpose(-1, -2)  # bs, dt, N

        prod = Qtb @ X_t_transposed  # bs, d0, N
        prod = prod.transpose(-1, -2)  # bs, N, d0
        denominator = prod.unsqueeze(-1)  # bs, N, d0,
        denominator[denominator == 0] = 1e-6

        out = numerator / denominator
        return out

    def p_s_and_t_given_0_a(self, z_t_a, t, s):
        Qtb_a = self.q_t_bar_atom(t)
        Qsb_a = self.q_t_bar_atom(s)
        Qt_a = self.q_t_atom(t)
        return self.p_s_and_t_given_0(z_t_a, Qt_a, Qsb_a, Qtb_a)

    def p_s_and_t_given_0_ss(self, z_t_ss, t, s, sgs):
        Qtb_ss = self.q_t_bar_ss(t, sgs)
        Qsb_ss = self.q_t_bar_ss(s, sgs)
        Qt_ss = self.q_t_ss(t, sgs)
        p_s_and_t_given_0_site_symms = []  # torch.zeros((list(z_t_ss.shape[:-1]) +  [27, 0]), device=z_t_ss.device)
        for i in range(len(self.P_ss)):
            p_s_and_t_given_0_site_symms.append(self.p_s_and_t_given_0(z_t_ss[i], Qt_ss[i], Qsb_ss[i], Qtb_ss[i]))
        return p_s_and_t_given_0_site_symms

    def sample_zs_from_zt_and_pred(self, z_t_a, z_t_ss, pred_a, pred_ss, t, s, node_mask, sgs):
        """Samples from zs ~ p(zs | zt). Only used during sampling. """

        # Retrieve transitions matrix
        # Qtb_a = self.q_t_bar_atom(t)
        # Qtb_ss = self.q_t_bar_ss(t)
        # Qsb_a = self.q_t_bar_atom(s)
        # Qsb_ss = self.q_t_bar_ss(s)
        # Qt_a = self.q_t_atom(t)
        # Qt_ss = self.q_t_ss(t)

        # Normalize predictions for the categorical features
        z_t_ss_split = self.ss_to_sections(z_t_ss)
        p_s_and_t_given_0_atom_types = self.p_s_and_t_given_0_a(z_t_a, t, s)
        p_s_and_t_given_0_site_symms = self.p_s_and_t_given_0_ss(z_t_ss_split, t, s, sgs)

        # Dim of these two tensors: bs, N, d0, d_t-1
        # pred_a = F.softmax(pred_a, dim=-1)               # bs, n, d0
        weighted_a = pred_a.unsqueeze(-1) * p_s_and_t_given_0_atom_types  # bs, n, d0, d_t-1
        unnormalized_prob_a = weighted_a.sum(dim=2)  # bs, n, d_t-1
        unnormalized_prob_a[torch.sum(unnormalized_prob_a, dim=-1) == 0] = 1e-5
        prob_a = unnormalized_prob_a / torch.sum(unnormalized_prob_a, dim=-1, keepdim=True)  # bs, n, d_t-1

        pred_ss_split = self.ss_to_sections(pred_ss)
        prob_ss_list = []
        for pred_ss_i, p_s_and_t_given_0_site_symms_i in zip(pred_ss_split, p_s_and_t_given_0_site_symms):
            # pred_ss_i = F.softmax(pred_ss_i, dim=-1)              # bs, n, d0
            weighted_ss = pred_ss_i.unsqueeze(-1) * p_s_and_t_given_0_site_symms_i  # bs, n, d0, d_t-1
            unnormalized_prob_ss = weighted_ss.sum(dim=2)  # bs, n, d_t-1
            unnormalized_prob_ss[torch.sum(unnormalized_prob_ss, dim=-1) == 0] = 1e-5
            prob_ss = unnormalized_prob_ss / torch.sum(unnormalized_prob_ss, dim=-1, keepdim=True)  # bs, n, d_t-1
            prob_ss_list.append(prob_ss)
        prob_ss = torch.cat(prob_ss_list, -1)

        assert ((prob_a.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_ss.sum(dim=-1) - len(prob_ss_list)).abs() < 1e-4).all()

        sampled_a_s, sampled_ss_s = self.sample_discrete_features(prob_a, prob_ss, node_mask)
        return sampled_a_s, sampled_ss_s

    def discrete_loss(self, sample_a, sample_ss, pred_a, pred_ss):
        '''
        Cross entropy loss for atom_types as well as each site_symm component
        '''
        loss_a = F.nll_loss(torch.log(pred_a + 1e-20), sample_a)
        pred_ss_split = self.ss_to_sections(pred_ss)
        losses = []
        for i, pred_ss_i in enumerate(pred_ss_split):
            loss_ss_i = F.nll_loss(torch.log(pred_ss_i + 1e-20), sample_ss[..., i])
            losses.append(loss_ss_i)
        loss_ss = torch.stack(losses).mean()
        return loss_a, loss_ss


class DiscreteNoiseMarginal(DiscreteNoise):
    def __init__(self, atom_marginals_path, ss_marginals_path, beta_scheduler):
        atom_type_prior = torch.load(atom_marginals_path, weights_only=True)
        site_symm_prior_per_sg = torch.load(ss_marginals_path, weights_only=True)
        P_ss = nn.ParameterList([nn.Parameter(
            site_symm_prior_per_sg[i].unsqueeze(-2).expand(NUM_SPACEGROUPS, SITE_SYMM_PGS, SITE_SYMM_PGS).clone(),
            requires_grad=False) for i in range(SITE_SYMM_AXES)])
        P_a = nn.Parameter(atom_type_prior.unsqueeze(0).expand(MAX_ATOMIC_NUM, -1).clone(), requires_grad=False)
        super().__init__(atom_type_prior, site_symm_prior_per_sg, beta_scheduler, P_ss, P_a)


class DiscreteNoiseMasked(DiscreteNoise):
    def __init__(self, beta_scheduler):
        atom_type_prior = torch.zeros(MAX_ATOMIC_NUM + 1)
        atom_type_prior[-1] = 1
        site_symm_prior = [torch.zeros(SITE_SYMM_PGS + 1) for i in range(SITE_SYMM_AXES)]
        for i in range(SITE_SYMM_AXES):
            site_symm_prior[i][-1] = 1
        site_symm_prior_per_sg = [site_symm_prior_i.expand(NUM_SPACEGROUPS, -1) for site_symm_prior_i in
                                  site_symm_prior]
        P_ss = nn.ParameterList([nn.Parameter(
            site_symm_prior[i].unsqueeze(-2).expand(NUM_SPACEGROUPS, SITE_SYMM_PGS + 1, SITE_SYMM_PGS + 1).clone(),
            requires_grad=False) for i in range(SITE_SYMM_AXES)])
        P_a = nn.Parameter(atom_type_prior.unsqueeze(0).expand(MAX_ATOMIC_NUM + 1, -1).clone(), requires_grad=False)
        super().__init__(atom_type_prior, site_symm_prior_per_sg, beta_scheduler, P_ss, P_a)
        self.max_atomic_num = MAX_ATOMIC_NUM + 1
        self.site_symm_pgs = SITE_SYMM_PGS + 1

    def sub_predictions(self, pred_a, pred_ss, atom_types, site_symms):
        # Modify atom and site symmetry predictions to account for masked tokens

        # Never predict masked tokens – zero them out
        mask_mask_atom = torch.zeros_like(pred_a) + 1
        mask_mask_atom[:, -1] = 0
        pred_a = pred_a * mask_mask_atom
        mask_mask_symm = torch.zeros_like(pred_ss) + 1
        self.reshape_ss(mask_mask_symm)[:, :, -1] = 0
        pred_ss = pred_ss * mask_mask_symm

        # If something is unmasked, keep it unmasked instead of predicting
        unmasked_atom = (atom_types[..., -1] == 0)[:, None].expand(-1, self.max_atomic_num)
        pred_a = torch.where(unmasked_atom, atom_types, pred_a)
        unmasked_symm = (
            (self.reshape_ss(site_symms)[:, :, -1] == 0)[:, :, None].expand(-1, -1, self.site_symm_pgs)).flatten(-2, -1)
        pred_ss = torch.where(unmasked_symm, site_symms, pred_ss)

        return pred_a, pred_ss


def find_num_atoms(dummy_ind, total_num_atoms):
    # num_atoms states how many atoms are there in each crystal (num_repr + dummy origin)
    actual_num_atoms = []
    atoms = 0
    for num in total_num_atoms:
        # find number of 0 in dummy_ind from atoms to atoms+num
        actual_num_atoms.append(torch.sum(dummy_ind[atoms:atoms + num] == 0).item())
        atoms += num

    return torch.tensor(actual_num_atoms)


def split_argmax_sitesymm(site_symm: torch.Tensor) -> np.ndarray:
    # site_symm : num_repr x 66
    return np.array(np.abs(1 - site_symm.cpu().detach().numpy()) < 0.1, dtype=float)


def modify_frac_coords_one(frac_coords, site_symm, atom_types, spacegroup):
    """
    优化版：增加了原子去重机制，防止对称性展开导致的原子重叠。
    """
    spacegroup = spacegroup.item()
    site_symm_axis = site_symm.reshape(-1, SITE_SYMM_AXES, SITE_SYMM_PGS).detach().cpu()

    # 获取该空间群下所有 Wyckoff 位置 (WP) 的位点对称性指纹
    wp_to_site_symm = SG_TO_WP_TO_SITE_SYMM[spacegroup]

    new_frac_coords = []
    new_atom_types = []
    new_site_symm = []

    min_ss_dists = []
    wp_projection_dists = []

    # 原子间距调整(分数坐标)
    MERGE_TOLERANCE = 0.25

    for (sym, frac_coord, atm_type) in zip(site_symm_axis, frac_coords, atom_types):
        frac_coord = frac_coord.cpu().detach().numpy()
        atm_type_np = atm_type.cpu().detach().numpy()
        sym_np = sym.cpu().detach().numpy()

        # 1. 寻找最匹配的 Wyckoff 位置
        wp_to_ss_dist = {wp: torch.norm(sym.flatten() - ss.flatten()) for wp, ss in wp_to_site_symm.items()}
        min_ss_dist = min(wp_to_ss_dist.values())
        min_ss_dists.append(min_ss_dist.item())
        closest_ss_wps = [wp for wp, dist in wp_to_ss_dist.items() if dist == min_ss_dist]

        # 2. 在物理空间中投影到最近的 Wyckoff 轨道
        closes = []
        for wp in closest_ss_wps:
            for orbit_index in range(len(wp.ops)):
                # 使用 PyXtal 投影坐标
                close = search_cloest_wp(SG_SYM[spacegroup], wp, wp.ops[orbit_index], frac_coord) % 1.
                # 计算周期性边界下的最小距离
                diff = (close - frac_coord) % 1.0
                diff = np.minimum(diff, 1.0 - diff)
                dist = np.linalg.norm(diff)

                closes.append((close, wp, orbit_index, dist))

        try:
            # 选择距离最近的投影
            closest = sorted(closes, key=lambda x: x[-1])[0]
            target_coord = closest[0]
            wyckoff = closest[1]
            repr_index = closest[2]
            wp_projection_dists.append(closest[3])

            # 3. 对称性展开 (Expansion) 并进行 去重 (Merging)
            # 这是一个关键修改：不直接 append，而是先检查是否已存在

            # 临时存储当前 Wyckoff 轨道生成的所有原子
            current_wp_atoms = []

            # 遍历该 Wyckoff 位置的所有对称操作
            for index in range(len(wyckoff)):
                # 生成新坐标
                gen_coord = wyckoff[(index + repr_index) % len(wyckoff)].operate(target_coord) % 1.

                # --- 核心去重逻辑 ---
                is_duplicate = False

                # A. 检查与【当前生成元生成的其他原子】是否重叠 (Self-collision)
                # 这种情况通常发生在一般位置靠近特殊位置时
                for existing_coord in current_wp_atoms:
                    diff = (gen_coord - existing_coord) % 1.0
                    diff = np.minimum(diff, 1.0 - diff)
                    if np.linalg.norm(diff) < MERGE_TOLERANCE:
                        is_duplicate = True
                        break

                if is_duplicate:
                    continue

                # B. 检查与【之前已经生成的其他原子】是否重叠 (Cross-collision)
                # 这种情况发生在两个不同的种子原子实际上占据了同一个位置
                if len(new_frac_coords) > 0:
                    # 将 list 转为 numpy 方便计算
                    existing_coords_array = np.array(new_frac_coords)
                    diff = (gen_coord - existing_coords_array) % 1.0
                    diff = np.minimum(diff, 1.0 - diff)
                    dists = np.linalg.norm(diff, axis=1)
                    if np.any(dists < MERGE_TOLERANCE):
                        is_duplicate = True

                if is_duplicate:
                    continue

                # 如果不是重复的，才添加
                current_wp_atoms.append(gen_coord)
                new_frac_coords.append(gen_coord)
                new_atom_types.append(atm_type_np)
                new_site_symm.append(sym_np)  # 这里 site_symm 只是简单的复制，实际上每个位置的指纹可能略有不同，但通常不影响后续

        except Exception as e:
            # Fallback: 如果出错，保留原始坐标 (不做展开，防止程序崩溃)
            # print(f'Warning: Projection failed for atom {atm_type_np}: {e}')
            new_frac_coords.append(frac_coord)
            new_atom_types.append(atm_type_np)
            new_site_symm.append(sym_np)

    # 堆叠结果
    if len(new_frac_coords) > 0:
        new_frac_coords = np.stack(new_frac_coords)
        new_atom_types = np.stack(new_atom_types)
        new_site_symm = np.stack(new_site_symm)
    else:
        # 防止空列表报错
        new_frac_coords = np.empty((0, 3))
        new_atom_types = np.empty((0,))
        new_site_symm = np.empty((0, SITE_SYMM_DIM))

    return new_frac_coords, len(new_frac_coords), new_atom_types, new_site_symm, min_ss_dists, wp_projection_dists

def modify_frac_coords(traj: Dict, spacegroups: List[int], num_repr: List[int]) -> Dict:
    device = traj['frac_coords'].device
    total_atoms = 0
    updated_frac_coords = []
    updated_num_atoms = []
    updated_atom_types = []
    updated_site_symm = []
    min_ss_dists, wp_projection_dists = [], []

    # 确保输入是 Tensor
    traj_frac_coords = traj['frac_coords']
    traj_atom_types = traj['atom_types']
    traj_site_symm = traj['site_symm']

    for index in range(len(num_repr)):
        if num_repr[index] > 0:
            # 提取当前晶体的种子原子
            current_frac = traj_frac_coords[total_atoms:total_atoms + num_repr[index]]
            current_symm = traj_site_symm[total_atoms:total_atoms + num_repr[index]]
            current_types = traj_atom_types[total_atoms:total_atoms + num_repr[index]]

            # 调用处理函数
            # 注意：传入的 spacegroups[index] 必须是 tensor 或 int，这里做一个兼容处理
            sg_val = spacegroups[index]
            if isinstance(sg_val, torch.Tensor):
                sg_val = sg_val.item()

            # modify_frac_coords_one 需要 spacegroup 是 tensor (为了 .item() 调用) 或者我们修改 one 函数
            # 这里为了兼容原始代码，我们重新封装成 tensor
            sg_tensor = torch.tensor(sg_val)

            new_frac_coords, new_num_atoms, new_atom_types, new_site_sym, min_ss_dist, wp_projection_dist = modify_frac_coords_one(
                current_frac,
                current_symm,
                current_types,
                sg_tensor,  # 传入 Tensor
            )

            if new_num_atoms > 0:
                updated_frac_coords.append(new_frac_coords)  # numpy array
                updated_num_atoms.append(new_num_atoms)
                updated_atom_types.append(new_atom_types)  # numpy array
                updated_site_symm.append(new_site_sym)
                min_ss_dists.append(min_ss_dist)
                wp_projection_dists.append(wp_projection_dist)

        total_atoms += num_repr[index]

    # === 关键修正：确保 frac_coords 和 atom_types 同时更新 ===
    if len(updated_frac_coords) > 0:
        # 转换为 Tensor 并覆盖原字典
        traj['frac_coords'] = torch.cat([torch.from_numpy(x).float() for x in updated_frac_coords]).to(device)
        traj['atom_types'] = torch.cat([torch.from_numpy(x).long() for x in updated_atom_types]).to(device)
        traj['num_atoms'] = torch.tensor(updated_num_atoms).to(device)
        traj['site_symm'] = torch.cat([torch.from_numpy(x).float() for x in updated_site_symm]).to(device)
        traj['min_ss_dists'] = min_ss_dists
        traj['wp_projection_dists'] = wp_projection_dists
    else:
        # 如果没有更新（比如出错了），保持原样，防止后续报错
        print("Warning: modify_frac_coords did not update any atoms!")

    return traj

class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()

### Model definition

class SinusoidalTimeEmbeddings(nn.Module):
    """ Attention is all you need. """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class CSPDiffusion(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        mask_token = 1 if self.hparams.prior == 'masked' else 0
        self.decoder = hydra.utils.instantiate(self.hparams.decoder,
                                               time_dim=self.hparams.time_dim + self.hparams.latent_dim,
                                               latent_dim=self.hparams.latent_dim, pred_type=True,
                                               pred_site_symm_type=True, smooth=True,
                                               max_atoms=MAX_ATOMIC_NUM + mask_token, mask_token=mask_token)
        self.beta_scheduler = hydra.utils.instantiate(self.hparams.beta_scheduler).to(self.device)
        self.sigma_scheduler = hydra.utils.instantiate(self.hparams.sigma_scheduler).to(self.device)
        self.time_dim = self.hparams.time_dim
        self.latent_dim = self.hparams.latent_dim
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.spacegroup_embedding = build_mlp(in_dim=SG_CONDITION_DIM, hidden_dim=128, fc_num_layers=2,
                                              out_dim=self.latent_dim)
        self.keep_lattice = self.hparams.cost_lattice < 1e-5
        self.keep_coords = self.hparams.cost_coord < 1e-5
        self.use_ks = self.hparams.use_ks
        self.discrete_noise = self.init_discrete_noise(self.hparams.prior)

    def init_discrete_noise(self, prior='marginal'):
        if prior == 'marginal':
            return DiscreteNoiseMarginal(self.hparams.data.datamodule.atom_marginals_path,
                                         self.hparams.data.datamodule.ss_marginals_path, self.beta_scheduler)
        elif prior == 'masked':
            return DiscreteNoiseMasked(self.beta_scheduler)

    def forward(self, batch):

        batch_size = batch.num_graphs
        atom_types, node_mask = to_dense_batch(batch.atom_types - 1, batch.batch, fill_value=0)
        if self.hparams.prior == 'masked':
            site_symm = torch.cat([batch.site_symm, torch.zeros_like(batch.site_symm)[..., :1]], dim=-1)
        else:
            site_symm = batch.site_symm
        site_symms, node_mask = to_dense_batch(site_symm.flatten(-2, -1), batch.batch, fill_value=0)
        times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        spacegroup_emb = self.spacegroup_embedding(batch.sg_condition.reshape(-1, SG_CONDITION_DIM))
        time_emb = torch.cat([self.time_embedding(times), spacegroup_emb], dim=-1)
        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]

        c0 = torch.sqrt(alphas_cumprod)
        c1 = torch.sqrt(1. - alphas_cumprod)

        sigmas = self.sigma_scheduler.sigmas[times]
        sigmas_norm = self.sigma_scheduler.sigmas_norm[times]

        ks = batch.ks
        if self.use_ks:
            lattices = lattice_ks_to_matrix_torch(batch.ks)
            ks_mask, ks_add = sg_to_ks_mask(batch.spacegroup)
        else:
            lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)

        frac_coords = batch.frac_coords

        rand_x = torch.randn_like(frac_coords)
        rand_ks = torch.randn_like(ks)
        rand_l = torch.randn_like(lattices)

        if self.use_ks:
            input_ks = c0[:, None] * ks + c1[:, None] * rand_ks
            input_ks = mask_ks(input_ks, ks_mask, ks_add)
            input_lattice = lattice_ks_to_matrix_torch(input_ks)
        else:
            input_lattice = c0[:, None, None] * lattices + c1[:, None, None] * rand_l

        sigmas_per_atom = sigmas.repeat_interleave(batch.num_atoms)[:, None]
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(batch.num_atoms)[:, None]
        input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.

        gt_atom_types_onehot = F.one_hot(atom_types, num_classes=self.discrete_noise.max_atomic_num).float()
        gt_site_symm_binary = site_symms

        atom_type_noised_probs = self.discrete_noise.apply_atom_noise(gt_atom_types_onehot, times)
        site_symm_noised_probs = self.discrete_noise.apply_site_symm_noise(gt_site_symm_binary, times, batch.spacegroup)
        atom_types_noised, site_symms_noised = self.discrete_noise.sample_discrete_features(atom_type_noised_probs,
                                                                                            site_symm_noised_probs,
                                                                                            node_mask)

        if self.keep_coords:
            input_frac_coords = frac_coords

        if self.keep_lattice:
            input_lattice = lattices
            input_ks = ks

        # pass noised site symmetries and behave similar to atom type probs
        lattice_feats = input_ks if self.use_ks else input_lattice
        symm_t = site_symms_noised[node_mask]
        atom_types_t = atom_types_noised[node_mask]
        pred_lattice, pred_x, pred_t_logit, pred_symm_logit = self.decoder(time_emb, atom_types_t, input_frac_coords,
                                                                           lattice_feats, input_lattice,
                                                                           batch.num_atoms,
                                                                           batch.batch, site_symm_probs=symm_t)

        pred_t = F.softmax(pred_t_logit, -1)
        pred_symm = F.softmax(self.discrete_noise.reshape_ss(pred_symm_logit), -1).flatten(-2, -1)
        if self.hparams.prior == 'masked':
            pred_t, pred_symm = self.discrete_noise.sub_predictions(pred_t, pred_symm, atom_types_t, symm_t)

        tar_x = d_log_p_wrapped_normal(sigmas_per_atom * rand_x, sigmas_per_atom) / torch.sqrt(sigmas_norm_per_atom)

        loss_lattice = F.mse_loss(pred_lattice, ks_mask * rand_ks) if self.use_ks else F.mse_loss(pred_lattice, rand_l)

        loss_coord = F.mse_loss(pred_x, tar_x)

        loss_type, loss_symm = self.discrete_noise.discrete_loss(batch.atom_types - 1, batch.site_symm.argmax(-1),
                                                                 pred_t, pred_symm)

        loss = (
                self.hparams.cost_lattice * loss_lattice +
                self.hparams.cost_coord * loss_coord +
                self.hparams.cost_type * loss_type +
                self.hparams.cost_symm * loss_symm
        )

        return {
            'loss': loss,
            'loss_lattice': loss_lattice,
            'loss_coord': loss_coord,
            'loss_type': loss_type,
            'loss_symm': loss_symm,
        }

    @torch.no_grad()
    def sample(
            self, batch, diff_ratio=1.0, step_lr=1e-5, return_traj=False,
            num_steps=None, fixed_atom_types=None, smc_callback=None, show_progress=True):
        """
        Sample new crystals from the model.

        Args:
            batch: The input batch (containing num_atoms, spacegroup, etc.)
            diff_ratio: Diffusion ratio (default 1.0)
            step_lr: Step size for Langevin dynamics on coordinates
            return_traj: Whether to return the full trajectory
            num_steps: Number of diffusion steps (if None, uses default)
            fixed_atom_types: (Optional) LongTensor of shape (total_nodes,).
                              If provided, locks the atom types to these values
                              and only generates coordinates/lattice.
            smc_callback: Optional inference-only callback that can return particle
                          ancestor indices for equal-size batch resampling.
            show_progress: Whether to render the tqdm diffusion progress bar.
        """
        batch = batch.to(self.device)

        batch_size = batch.num_graphs
        if smc_callback is not None:
            unique_atom_counts = torch.unique(batch.num_atoms)
            if len(unique_atom_counts) != 1:
                raise ValueError("SMC resampling requires equal atom counts in every particle")
            smc_atoms_per_particle = int(unique_atom_counts[0].item())
        ks_mask, ks_add = sg_to_ks_mask(batch.spacegroup)

        # 初始化噪声（t=T）
        k_T = torch.randn([batch_size, 6]).to(self.device)
        k_T = mask_ks(k_T, ks_mask, ks_add)
        l_T = lattice_ks_to_matrix_torch(k_T)
        x_T = torch.rand([batch.num_nodes, 3]).to(self.device)

        _, node_mask = to_dense_batch(batch.batch, batch.batch, fill_value=0)

        # 初始原子类型和位点对称性采样
        t_T, symm_T = self.discrete_noise.sample_limit_dist(node_mask, batch.spacegroup)
        t_T = t_T[node_mask]
        symm_T = symm_T[node_mask]

        if self.keep_coords:
            x_T = batch.frac_coords
        if self.keep_lattice:
            k_T = batch.ks
            l_T = lattice_ks_to_matrix_torch(k_T) if self.use_ks else lattice_params_to_matrix_torch(batch.lengths,
                                                                                                     batch.angles)

        # 当前时刻的状态（从噪声开始）
        x_t = x_T
        l_t = l_T
        k_t = k_T
        t_t = t_T
        symm_t = symm_T

        # 如果指定了 fixed_atom_types，预先准备 One-hot 格式
        fixed_atom_onehot = None
        if fixed_atom_types is not None:
            # 确保输入是 Dense 格式以便处理 Mask
            fixed_t_dense, _ = to_dense_batch(fixed_atom_types, batch.batch, fill_value=0)
            # 转换为 One-hot (Batch, N_max, Atom_Dim)
            fixed_atom_onehot = F.one_hot(fixed_t_dense, num_classes=self.discrete_noise.max_atomic_num).float()

        # 采样步数
        total_steps = self.beta_scheduler.timesteps
        if num_steps is not None:
            step_indices = torch.linspace(total_steps, 0, num_steps + 1, dtype=torch.long, device=self.device)
        else:
            step_indices = torch.arange(total_steps, 0, -1, device=self.device)

        traj = defaultdict(list)

        for i in tqdm(
                range(1, len(step_indices)), desc="Sampling", leave=False,
                disable=not show_progress):
            t = step_indices[i - 1].item()
            next_t = step_indices[i].item()

            # =================================================================
            # NEW: Fixed Atom Types Logic (Inpainting / Conditional Sampling)
            # =================================================================
            if fixed_atom_types is not None:
                # 计算当前时刻 t 下，fixed_atoms 应该呈现的噪声分布 q(x_t | x_0)
                times_now = torch.full((batch_size,), t, device=self.device)
                prob_a_t = self.discrete_noise.apply_atom_noise(fixed_atom_onehot, times_now)

                # 构造一个假的 prob_ss 用于调用 sample_discrete_features (只为了采样 atom)
                # 我们只关心返回的 atom_t，忽略 symm_t
                dummy_prob_ss = torch.ones(batch_size, node_mask.shape[1], SITE_SYMM_DIM, device=self.device)

                # 从分布中采样得到当前时刻的含噪原子类型
                t_t_dense, _ = self.discrete_noise.sample_discrete_features(prob_a_t, dummy_prob_ss, node_mask)
                t_t = t_t_dense[node_mask]

                # 可选：在最后几步去除噪声，确保输入给 Decoder 的是非常干净的类型
                if t < 5:
                    t_t = fixed_atom_onehot[node_mask]
            # =================================================================

            times = torch.full((batch_size,), t, device=self.device)

            spacegroup_emb = self.spacegroup_embedding(batch.sg_condition.reshape(-1, SG_CONDITION_DIM))
            time_emb = torch.cat([self.time_embedding(times), spacegroup_emb], dim=-1)

            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]
            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]

            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

            # --- Corrector step ---
            rand_x = torch.randn_like(x_t) if t > 1 else torch.zeros_like(x_t)
            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            std_x = torch.sqrt(2 * step_size)

            lattice_feats_t = k_t if self.use_ks else l_t

            # Decoder forward
            _, pred_x, _, _ = self.decoder(time_emb, t_t, x_t, lattice_feats_t, l_t, batch.num_atoms, batch.batch,
                                           site_symm_probs=symm_t)
            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t = x_t - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t

            # --- Predictor step ---
            rand_k = torch.randn_like(k_T) if t > 1 else torch.zeros_like(k_T)
            rand_x = torch.randn_like(x_t) if t > 1 else torch.zeros_like(x_t)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[next_t]
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))

            lattice_feats_t = k_t if self.use_ks else l_t

            # Decoder forward again
            pred_l, pred_x, pred_t_logit, pred_symm_logit = self.decoder(
                time_emb, t_t, x_t, lattice_feats_t, l_t, batch.num_atoms, batch.batch, site_symm_probs=symm_t
            )

            pred_t = F.softmax(pred_t_logit, dim=-1)
            pred_symm = F.softmax(self.discrete_noise.reshape_ss(pred_symm_logit), dim=-1).flatten(-2, -1)
            if self.hparams.prior == 'masked':
                pred_t, pred_symm = self.discrete_noise.sub_predictions(pred_t, pred_symm, t_t, symm_t)

            pred_x = pred_x * torch.sqrt(sigma_norm)

            # Update coordinates
            x_next = x_t - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t
            x_next = x_next % 1.0

            # Update lattice
            if self.use_ks:
                k_next = c0 * (k_t - c1 * pred_l) + sigmas * rand_k if not self.keep_lattice else k_t
                k_next = mask_ks(k_next, ks_mask, ks_add)
                l_next = lattice_ks_to_matrix_torch(k_next) if not self.keep_lattice else l_t
            else:
                l_next = c0 * (l_t - c1 * pred_l) + sigmas * rand_k if not self.keep_lattice else l_t
                k_next = k_t

            # Update discrete features (Atom Types & Site Symm)
            pred_t_dense, _ = to_dense_batch(pred_t, batch.batch, fill_value=0)
            pred_symm_dense, _ = to_dense_batch(pred_symm, batch.batch, fill_value=0)
            t_t_dense, _ = to_dense_batch(t_t, batch.batch, fill_value=0)
            symm_t_dense, _ = to_dense_batch(symm_t, batch.batch, fill_value=0)

            t_next, symm_next = self.discrete_noise.sample_zs_from_zt_and_pred(
                t_t_dense, symm_t_dense, pred_t_dense, pred_symm_dense,
                times, torch.full_like(times, next_t), node_mask,
                batch.spacegroup
            )
            t_next = t_next[node_mask]
            symm_next = symm_next[node_mask]

            # 更新当前状态
            x_t = x_next
            l_t = l_next
            k_t = k_next
            # 如果固定了原子，这里的 t_t 更新其实不重要，因为下一次循环开头会被重置
            # 但为了逻辑完整性保留它
            t_t = t_next
            symm_t = symm_next

            if smc_callback is not None:
                callback_result = smc_callback({
                    'step': i,
                    'diffusion_time': next_t,
                    'is_final_step': i == len(step_indices) - 1,
                    'frac_coords': x_t,
                    'lattices': l_t,
                    'ks': k_t,
                    'atom_types': t_t,
                    'site_symm': symm_t,
                })
                if callback_result and callback_result.get('ancestors') is not None:
                    ancestors = torch.as_tensor(
                        callback_result['ancestors'], device=self.device, dtype=torch.long
                    )
                    if ancestors.numel() != batch_size:
                        raise ValueError("SMC callback returned the wrong number of ancestors")

                    def resample_graph_tensor(tensor):
                        return tensor.index_select(0, ancestors)

                    def resample_node_tensor(tensor):
                        shape = (batch_size, smc_atoms_per_particle, *tensor.shape[1:])
                        return tensor.reshape(shape).index_select(0, ancestors).reshape_as(tensor)

                    x_t = resample_node_tensor(x_t)
                    t_t = resample_node_tensor(t_t)
                    symm_t = resample_node_tensor(symm_t)
                    l_t = resample_graph_tensor(l_t)
                    k_t = resample_graph_tensor(k_t)

            if return_traj:
                traj['x'].append(x_t.cpu())
                traj['l'].append(l_t.cpu())
                traj['t'].append(t_t.cpu())
                traj['s'].append(symm_t.cpu())

        # 最终输出处理
        final_atom_types = t_t
        if fixed_atom_types is not None:
            # 如果是固定模式，输出必须是 Clean 的 one-hot
            # fixed_atom_onehot 是 (Batch, N, Dim)，需要转回 (Total_Nodes, Dim)
            final_atom_types = fixed_atom_onehot[node_mask]

        final_output = {
            'num_atoms': batch.num_atoms,
            'atom_types': final_atom_types,
            'site_symm': symm_t,
            'frac_coords': x_t % 1.0,
            'lattices': l_t,
            'ks': k_t,
            'spacegroup': batch.spacegroup,
        }

        return final_output, traj if return_traj else None

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss_type = output_dict['loss_type']
        loss_symm = output_dict['loss_symm']
        loss = output_dict['loss']

        self.log_dict(
            {'train_loss': loss,
             'lattice_loss': loss_lattice,
             'coord_loss': loss_coord,
             'type_loss': loss_type,
             'symm_loss': loss_symm,
             },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        if loss.isnan():
            return None

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='val')

        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        if (self.current_epoch + 1) % self.hparams.data.eval_every_epoch == 0 and batch_idx == 0:
            # run a simpler evaluation
            self.simple_gen_evaluation()

        return loss

    def simple_gen_evaluation(self):
        import random

        import pandas as pd

        from scripts.compute_metrics import Crystal, GenEval, get_gt_crys_ori
        from scripts.eval_utils import lattices_to_params_shape, get_crystals_list
        from scripts.generation import SampleDataset

        eval_model_name_dataset = {
            "mp20": "mp",  # encompasses mp20, mpsa52
            "perovskite": "perov",
            "carbon": "carbon",
        }
        test_set = SampleDataset(
            eval_model_name_dataset[self.hparams.data.eval_model_name],
            self.hparams.data.eval_generate_samples,
            self.hparams.data.datamodule.datasets.train.save_path,
            self.hparams.data.datamodule.datasets.train.sg_info_path,
        )

        total_samples = self.hparams.data.eval_generate_samples
        print(f"INFO: Start generating {total_samples} crystals for evaluation (Epoch {self.current_epoch + 1})")

        frac_coords = []
        num_atoms = []
        atom_types = []
        lattices = []
        spacegroups = []
        site_symmetries = []

        for i in tqdm(range(total_samples), desc="Generating crystals"):
            data = test_set[i]
            data = data.to(self.device)
            batch = Batch.from_data_list([data]).to(self.device)

            gen_output, _ = self.sample(
                batch,
                step_lr=1e-5,
                return_traj=False,
                num_steps=300  # Adjust if needed for speed/quality
            )

            frac_coords.append(gen_output['frac_coords'].cpu())
            num_atoms.append(gen_output['num_atoms'].cpu())
            atom_types.append(gen_output['atom_types'].cpu())
            lattices.append(gen_output['lattices'].cpu())
            spacegroups.append(gen_output['spacegroup'].cpu())
            site_symmetries.append(gen_output['site_symm'].cpu())

            del gen_output
            if (i + 1) % 20 == 0:
                torch.cuda.empty_cache()
                import gc
                gc.collect()

        frac_coords = torch.cat(frac_coords, dim=0)
        num_atoms = torch.cat(num_atoms, dim=0)
        atom_types = torch.cat(atom_types, dim=0)
        lattices = torch.cat(lattices, dim=0)
        spacegroups = torch.cat(spacegroups, dim=0)
        site_symmetries = torch.cat(site_symmetries, dim=0)
        lengths, angles = lattices_to_params_shape(lattices)

        kwargs = {"spacegroups": spacegroups, "site_symmetries": site_symmetries}
        pred_crys_array_list = get_crystals_list(frac_coords, atom_types, lengths, angles, num_atoms, **kwargs)

        gen_crys = [Crystal(x) for x in tqdm(pred_crys_array_list, desc="Processing Crystals")]
        print(f"INFO: Done generating {total_samples} crystals (Epoch: {self.current_epoch + 1})")

        gt_path = self.hparams.data.datamodule.datasets.val[0].gt_crys_path
        subsample_size = 200  # subsample 减少内存
        if os.path.exists(gt_path):
            gt_crys_full = torch.load(gt_path)
            gt_crys = random.sample(gt_crys_full, min(subsample_size, len(gt_crys_full)))
            print(f"Loaded and subsampled {len(gt_crys)} GT crystals from cache")
        else:
            csv = pd.read_csv(self.hparams.data.datamodule.datasets.val[0].path)
            subsample_cif = random.sample(list(csv['cif']), subsample_size)
            gt_crys = [get_gt_crys_ori(cif) for cif in tqdm(subsample_cif, desc="Reading subsampled GT")]
            torch.save(gt_crys, gt_path + '_subsample.pt')  # 缓存 subsample

        print(f"INFO: Done reading ground truth crystals (Epoch: {self.current_epoch + 1})")

        gen_evaluator = GenEval(
            gen_crys,
            gt_crys,
            n_samples=0,
            eval_model_name=self.hparams.data.eval_model_name,
            gt_prop_eval_path=self.hparams.data.datamodule.datasets.val[0].gt_prop_eval_path
        )
        gen_metrics = gen_evaluator.get_metrics()
        print(gen_metrics)
        self.log_dict(gen_metrics)

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='test')

        self.log_dict(
            log_dict,
        )
        return loss

    def compute_stats(self, output_dict, prefix):

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss_type = output_dict['loss_type']
        loss_symm = output_dict['loss_symm']
        loss = output_dict['loss']

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_lattice_loss': loss_lattice,
            f'{prefix}_coord_loss': loss_coord,
            f'{prefix}_type_loss': loss_type,
            f'{prefix}_symm_loss': loss_symm,
        }

        return log_dict, loss

