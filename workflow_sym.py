import os
import sys
import torch
import numpy as np
import shutil
import argparse
import re

import torch.compiler

if not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda: False

# ASE & MACE
from ase.io import read, write
from ase.optimize import FIRE, BFGS
from ase.filters import ExpCellFilter
from mace.calculators import mace_mp

# PyTorch Geometric & Pymatgen
from torch_geometric.data import Data, Batch
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher

import datetime
from dataclasses import dataclass, field
from pathlib import Path


class DualLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


# ==========================================
# 0. 环境路径配置
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent
os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)
sys.path.append(str(PROJECT_ROOT))

from symmcd.pl_modules.mainmodelfun import CSPDiffusion, modify_frac_coords, SG_CONDITION_DIM, SITE_SYMM_DIM


@dataclass(frozen=True)
class WorkflowConfig:
    """Validated inference settings for one crystal-generation request."""

    formula: str
    spacegroup_number: int
    num_samples: int = 10
    checkpoint_path: Path = field(default_factory=lambda: PROJECT_ROOT / "epoch699.ckpt")
    output_root: Path = field(default_factory=lambda: PROJECT_ROOT / "generated_structures")
    enable_relax: bool = True
    mace_model: str = "medium"
    mace_device: str = "cpu"
    pre_relax_steps: int = 20
    fine_relax_steps: int = 100
    energy_cutoff: float = -200.0
    energy_upper_cutoff: float = 1000.0
    init_symprec: float = 0.2
    final_symprec: float = 0.1
    diffusion_steps: int = 500
    step_lr: float = 1e-5

    def validate(self) -> None:
        if not self.formula.strip():
            raise ValueError("formula cannot be empty")
        if not 1 <= self.spacegroup_number <= 230:
            raise ValueError("spacegroup_number must be between 1 and 230")
        if not 1 <= self.num_samples <= 1000:
            raise ValueError("num_samples must be between 1 and 1000")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"SymmCD checkpoint not found: {self.checkpoint_path}")
        if self.pre_relax_steps < 0 or self.fine_relax_steps < 0:
            raise ValueError("relaxation step counts cannot be negative")

    @property
    def file_prefix(self) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", self.formula.strip())


@dataclass
class WorkflowResult:
    formula: str
    spacegroup_number: int
    requested_samples: int
    unique_initial_structures: int
    accepted_poscars: list[str]
    output_directory: str
    rejected_directory: str
    log_file: str


_MODEL_CACHE: dict[tuple[str, str], CSPDiffusion] = {}
_MACE_CACHE: dict[tuple[str, str], object] = {}


def atomic_numbers_from_formula(formula: str) -> list[int]:
    """Expand an integer chemical formula such as BN or B2N2 to atomic numbers."""
    from pymatgen.core import Composition

    try:
        composition = Composition(formula)
    except Exception as exc:
        raise ValueError(f"invalid chemical formula: {formula}") from exc

    atomic_numbers: list[int] = []
    for element in composition.elements:
        amount = composition[element]
        rounded_amount = round(float(amount))
        if rounded_amount <= 0 or abs(float(amount) - rounded_amount) > 1e-8:
            raise ValueError("formula must contain positive integer stoichiometric amounts")
        atomic_numbers.extend([element.Z] * rounded_amount)

    if not atomic_numbers:
        raise ValueError(f"formula contains no elements: {formula}")
    return atomic_numbers


def _extract_inference_priors(checkpoint_path: Path) -> tuple[Path, Path]:
    """Materialize categorical priors embedded in the checkpoint state dict."""
    cache_dir = PROJECT_ROOT / ".inference_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_tag = f"{checkpoint_path.stem}_{checkpoint_path.stat().st_size}"
    atom_path = cache_dir / f"{cache_tag}_atom_marginals.pt"
    site_symm_path = cache_dir / f"{cache_tag}_site_symm_marginals.pt"
    if atom_path.is_file() and site_symm_path.is_file():
        return atom_path, site_symm_path

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", {})
    atom_transition = state_dict.get("discrete_noise.P_a")
    if atom_transition is None:
        raise KeyError("checkpoint does not contain discrete_noise.P_a")

    site_symm_priors = []
    index = 0
    while f"discrete_noise.P_ss.{index}" in state_dict:
        transition = state_dict[f"discrete_noise.P_ss.{index}"]
        site_symm_priors.append(transition[:, 0, :].clone())
        index += 1
    if not site_symm_priors:
        raise KeyError("checkpoint does not contain discrete_noise.P_ss.*")

    torch.save(atom_transition[0].clone(), atom_path)
    torch.save(site_symm_priors, site_symm_path)
    return atom_path, site_symm_path


def _load_generation_model(config: WorkflowConfig, device: str) -> CSPDiffusion:
    cache_key = (str(config.checkpoint_path.resolve()), device)
    if cache_key not in _MODEL_CACHE:
        from omegaconf import OmegaConf

        atom_path, site_symm_path = _extract_inference_priors(config.checkpoint_path)
        inference_data = OmegaConf.create(
            {
                "datamodule": {
                    "atom_marginals_path": str(atom_path),
                    "ss_marginals_path": str(site_symm_path),
                }
            }
        )
        model = CSPDiffusion.load_from_checkpoint(
            str(config.checkpoint_path),
            map_location=device,
            strict=False,
            data=inference_data,
        )
        _MODEL_CACHE[cache_key] = model.to(device).eval()
    return _MODEL_CACHE[cache_key]


def _load_mace_calculator(config: WorkflowConfig):
    cache_key = (config.mace_model, config.mace_device)
    if cache_key not in _MACE_CACHE:
        _MACE_CACHE[cache_key] = mace_mp(
            model=config.mace_model,
            default_dtype="float64",
            device=config.mace_device,
        )
    return _MACE_CACHE[cache_key]


# ==========================================
# 1. 生成模型辅助函数
# ==========================================
def build_constrained_batch(spacegroup_num, atom_list, device):
    num_atoms = len(atom_list)
    frac_coords = torch.zeros((num_atoms, 3), dtype=torch.float)
    atom_types = torch.tensor(atom_list, dtype=torch.long)
    site_symm = torch.zeros((num_atoms, SITE_SYMM_DIM), dtype=torch.float)
    sg_condition = torch.zeros((1, SG_CONDITION_DIM), dtype=torch.float)
    if 1 <= spacegroup_num <= 230:
        sg_condition[0, spacegroup_num - 1] = 1.0
    lengths = torch.tensor([[5.0, 5.0, 5.0]], dtype=torch.float)
    angles = torch.tensor([[90.0, 90.0, 90.0]], dtype=torch.float)
    ks = torch.zeros((1, 6), dtype=torch.float)
    data = Data(frac_coords=frac_coords, atom_types=atom_types, lengths=lengths, angles=angles,
                edge_index=torch.zeros((2, 0), dtype=torch.long), num_atoms=torch.tensor([num_atoms], dtype=torch.long),
                spacegroup=torch.tensor([spacegroup_num], dtype=torch.long), sg_condition=sg_condition,
                site_symm=site_symm, ks=ks, num_nodes=num_atoms)
    return Batch.from_data_list([data]).to(device)


def save_as_cif(output_dict, filename, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    frac_coords = output_dict['frac_coords'][0].cpu().numpy()
    lattice_matrix = output_dict['lattices'][0].cpu().numpy()
    atom_types_raw = output_dict['atom_types']
    if atom_types_raw.dim() == 3:
        atom_numbers = atom_types_raw[0].argmax(dim=-1).cpu().numpy() + 1
    else:
        atom_numbers = atom_types_raw[0].cpu().numpy()
    valid_mask = atom_numbers > 0
    if not np.any(valid_mask): return None
    try:
        struct = Structure(lattice=Lattice(lattice_matrix), species=atom_numbers[valid_mask],
                           coords=frac_coords[valid_mask], coords_are_cartesian=False)
        filepath = os.path.join(output_dir, filename)
        struct.to(filename=filepath)
        return filepath
    except Exception as e:
        return None


# ==========================================
# 2. MACE 通用弛豫引擎
# ==========================================
def run_mace_relaxation(input_cif, mace_calculator, output_name_base, max_steps, fmax_target=0.05, opt_algorithm=BFGS,
                        custom_poscar_name=None):
    algo_name = opt_algorithm.__name__
    print(f"  [MACE] 开始执行: {output_name_base} (算法: {algo_name}, 步数上限: {max_steps}, 目标 fmax: {fmax_target})")
    log_file = None
    try:
        atoms = read(input_cif)
        n_atoms = len(atoms)
        base_dir = os.path.dirname(input_cif)

        atoms.calc = mace_calculator
        e_init = atoms.get_potential_energy()

        ucf = ExpCellFilter(atoms)
        log_file = os.path.join(base_dir, f"{output_name_base}.log")

        opt = opt_algorithm(ucf, logfile=log_file)
        opt.run(fmax=fmax_target, steps=max_steps)

        e_final = atoms.get_potential_energy()

        out_struct_file = os.path.join(base_dir, f"{output_name_base}.cif")

        if custom_poscar_name:
            out_poscar = os.path.join(base_dir, custom_poscar_name)
        else:
            out_poscar = os.path.join(base_dir, f"{output_name_base}.POSCAR")

        write(out_struct_file, atoms, format="cif")
        write(out_poscar, atoms, format="vasp")

        print(f"  [MACE] 弛豫段落完成！")
        print(f"         - 初始能量: {e_init:.4f} eV")
        print(f"         - 最终能量: {e_final:.4f} eV")

        if log_file and os.path.exists(log_file):
            try:
                del opt
                os.remove(log_file)
            except Exception:
                pass

        return out_struct_file, e_final
    except Exception as e:
        print(f"  [MACE] 弛豫崩溃！错误信息: {e}")
        if log_file and os.path.exists(log_file):
            try:
                os.remove(log_file)
            except Exception:
                pass
        return None, None


# ==========================================
# 3. 主程序：高通量结构发现工作流
# ==========================================
def _run_generation(config: WorkflowConfig, output_dir: Path, others_dir: Path, log_path: Path) -> WorkflowResult:
    enable_relax = config.enable_relax
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    target_sg = config.spacegroup_number
    target_atoms = atomic_numbers_from_formula(config.formula)
    accepted_poscars: list[str] = []

    print("\n" + "*" * 60)
    print(f"[启动时间] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[生成条件] {config.formula}, 空间群 No. {target_sg}, 原子序数 {target_atoms}")
    print(f"[弛豫配置] enable_relax = {enable_relax}")
    print("*" * 60 + "\n")

    gen_model = _load_generation_model(config, DEVICE)

    if enable_relax:
        mace_calc = _load_mace_calculator(config)
    else:
        mace_calc = None

    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
    unique_structures_pool = []
    print("  [配置] 结构去重器已启动")

    unique_count = 0

    for i in range(config.num_samples):
        sample_id = i + 1
        current_prefix = f"{config.file_prefix}_sg{target_sg}_{sample_id:03d}"
        print(f"\n开始处理第 {sample_id}/{config.num_samples} 次采样 (当前已收集独立结构: {unique_count}个)")

        # --- 阶段 A: 生成与精确对称化标准化 ---
        torch.set_default_dtype(torch.float32)
        batch = build_constrained_batch(target_sg, target_atoms, DEVICE)
        with torch.no_grad():
            outputs, _ = gen_model.sample(
                batch,
                num_steps=config.diffusion_steps,
                step_lr=config.step_lr,
                fixed_atom_types=batch.atom_types - 1,
            )
            pred_indices = outputs['atom_types'].argmax(dim=-1) if outputs['atom_types'].dim() == 3 else outputs[
                'atom_types']
            traj_data = modify_frac_coords(
                {'frac_coords': outputs['frac_coords'], 'atom_types': pred_indices, 'site_symm': outputs['site_symm'],
                 'num_atoms': outputs['num_atoms']}, [torch.tensor(target_sg)], [len(target_atoms)])

            cif_filename = f"gen_{current_prefix}.cif"
            raw_cif_path = save_as_cif(
                {'frac_coords': traj_data['frac_coords'].unsqueeze(0), 'lattices': outputs['lattices'],
                 'atom_types': traj_data['atom_types'].unsqueeze(0)}, cif_filename, str(others_dir))

        if not raw_cif_path: continue

        # 🌟 修改：进行硬约束投影（Symmetry Snapping）
        # 晶格标准化与去重
        try:
            struct = Structure.from_file(raw_cif_path)

            # ========================================================
            # 🌟 强力防御机制：防止 spglib 因恶性结构发生 Access Violation 崩溃
            # ========================================================
            # 1. 检查是否存在 NaN 或 Inf 无效数值
            if np.isnan(struct.lattice.matrix).any() or np.isnan(struct.frac_coords).any():
                print("  [晶体学] 警告：结构中包含 NaN 无效值，跳过该结构以防止崩溃。")
                continue

            # 2. 检查晶格体积是否塌陷 (体积过小说明晶格不合法)
            if struct.volume < 1.0:  # 体积小于 1 Å³ 绝对是异常结构
                print(f"  [晶体学] 警告：晶格体积过小 ({struct.volume:.2f} A^3)，判定为塌陷结构，跳过。")
                continue

            # 3. 检查原子是否严重重叠 (距离极近)
            if len(struct) > 1:
                dist_matrix = struct.distance_matrix  # 考虑周期性边界条件的距离矩阵
                np.fill_diagonal(dist_matrix, 100.0)  # 将对角线（自身距离）填一个大数排除
                min_dist = np.min(dist_matrix)
                if min_dist < 0.4:  # 两个原子的距离如果小于 0.4 埃，通常会导致 spglib 划分网格崩溃
                    print(f"  [晶体学] 警告：检测到原子严重重叠 (最小间距 {min_dist:.2f} A)，跳过以防止 spglib 崩溃。")
                    continue
            # ========================================================

            # 只有通过上述体检的结构，才允许进入 SpacegroupAnalyzer
            sga = SpacegroupAnalyzer(struct, symprec=config.init_symprec)

            # 【核心修改点 1】：强制精确精炼结构，将微小的浮点数偏差卡回完美的对称位置
            refined_struct = sga.get_refined_structure()

            # 再从完美的对称结构中获取初基原胞用于去重和后续计算
            sga_refined = SpacegroupAnalyzer(refined_struct, symprec=config.final_symprec)
            std_struct = sga_refined.get_primitive_standard_structure()

            is_duplicate = False
            for existing_struct in unique_structures_pool:
                if matcher.fit(std_struct, existing_struct):
                    is_duplicate = True
                    break

            if is_duplicate:
                print("  [去重] 发现重复生成结构，跳过。")
                continue

            unique_structures_pool.append(std_struct)
            unique_count += 1

            std_cif_path = raw_cif_path.replace('.cif', '_std.cif')
            std_struct.to(filename=std_cif_path)
            work_cif_path = std_cif_path
            print(
                f"  [晶体学] 初始结构收录成功！强制投影后空间群: {sga_refined.get_space_group_symbol()} (No. {sga_refined.get_space_group_number()})")
        except Exception as e:
            print(f"  [晶体学] 初始分析或强制对称化出错: {e}")
            continue

        final_path = None

        if enable_relax:
            # --- 阶段 B: MACE 预弛豫 (FIRE) ---
            torch.set_default_dtype(torch.float64)
            prerelax_prefix = f"{current_prefix}_prerelax"
            prerelaxed_path, e_final_pre = run_mace_relaxation(work_cif_path, mace_calc, prerelax_prefix,
                                                               config.pre_relax_steps, 0.5, FIRE)

            bypass_fine_relax = False
            if e_final_pre is not None:
                if e_final_pre < config.energy_cutoff:
                    final_path = prerelaxed_path
                    bypass_fine_relax = True
                elif e_final_pre > config.energy_upper_cutoff:
                    print(f"  [判定] 预弛豫能量过高，放弃任务。")
                    continue

            # --- 阶段 C: 精细弛豫 (BFGS) ---
            if prerelaxed_path and not bypass_fine_relax:
                final_path, e_final_bfgs = run_mace_relaxation(
                    input_cif=prerelaxed_path, mace_calculator=mace_calc,
                    output_name_base=f"{current_prefix}_finerelax",
                    max_steps=config.fine_relax_steps, fmax_target=0.05, opt_algorithm=BFGS
                )
        else:
            final_path = work_cif_path

        # --- 🌟 阶段 D: 空间群二次硬约束校准与最终分流 ---
        if final_path:
            try:
                final_struct = Structure.from_file(final_path)

                # 【核心修改点 2】：因为 MACE 弛豫会严重破坏对称性，在最终判定前，必须再次用容差尝试将其“扶正”
                final_sga_pre = SpacegroupAnalyzer(final_struct, symprec=config.init_symprec)
                final_refined_struct = final_sga_pre.get_refined_structure()

                # 用严格的容差做最终的严格体检
                final_sga = SpacegroupAnalyzer(final_refined_struct, symprec=config.final_symprec)
                final_sg_num = final_sga.get_space_group_number()
                final_sg_symbol = final_sga.get_space_group_symbol()

                actual_elements = set(final_refined_struct.atomic_numbers)
                target_elements = set(target_atoms)

                is_sg_match = (final_sg_num == target_sg)
                is_elem_match = (actual_elements == target_elements)

                mace_poscar_path = final_path.replace('.cif', '.POSCAR')

                if is_sg_match and is_elem_match:
                    # 【完全符合要求】 提取完美的、严格对称的初基原胞并输出
                    final_std_struct = final_sga.get_primitive_standard_structure()
                    target_poscar_name = f"POSCAR_{current_prefix}"
                    target_poscar_path = os.path.join(output_dir, target_poscar_name)
                    final_std_struct.to(filename=target_poscar_path, fmt="poscar")
                    accepted_poscars.append(str(Path(target_poscar_path).resolve()))

                    print("  [判定结果] 结构符合初始对称性要求。")
                    print(f"         - 最终空间群: {final_sg_symbol} (No. {final_sg_num})")
                    print(f"         - 成功导出严格对称初基原胞: {target_poscar_path}")
                else:
                    # 【不符合要求】 无论怎么扶都扶不回来的结构（畸变过大），扔进 others 文件夹
                    others_cif_name = f"{current_prefix}_wrong_sg{final_sg_num}.cif"
                    others_poscar_name = f"POSCAR_{current_prefix}_wrong_sg{final_sg_num}"

                    others_cif_path = os.path.join(others_dir, others_cif_name)
                    others_poscar_path = os.path.join(others_dir, others_poscar_name)

                    if os.path.exists(final_path) and final_path != others_cif_path:
                        shutil.move(final_path, others_cif_path)
                    if os.path.exists(mace_poscar_path) and mace_poscar_path != others_poscar_path:
                        shutil.move(mace_poscar_path, others_poscar_path)

                    print("  [判定结果] 结构发生严重对称性破缺，无法恢复。")
                    print(f"         - 实际空间群: {final_sg_symbol} (No. {final_sg_num}) [期望: {target_sg}]")
                    print(f"         - 文件已归档于目录: {others_dir}/")

            except Exception as e:
                print(f"  [晶体学] 最终结构验证或文件管理发生错误: {e}")

    print("\n" + "*" * 60)
    print(f"运行结束。总采样次数: {config.num_samples} | 捕获独特初始结构: {unique_count} 个")
    print("*" * 60 + "\n")

    return WorkflowResult(
        formula=config.formula,
        spacegroup_number=target_sg,
        requested_samples=config.num_samples,
        unique_initial_structures=unique_count,
        accepted_poscars=accepted_poscars,
        output_directory=str(output_dir.resolve()),
        rejected_directory=str(others_dir.resolve()),
        log_file=str(log_path.resolve()),
    )


def run_workflow(config: WorkflowConfig) -> WorkflowResult:
    """Run one generation job while keeping model instances cached for later chat turns."""
    config.validate()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    job_dir = (
        config.output_root
        / f"{config.file_prefix}_sg{config.spacegroup_number}"
        / f"run_{timestamp}"
    )
    output_dir = job_dir / "accepted"
    others_dir = job_dir / "others"
    output_dir.mkdir(parents=True, exist_ok=True)
    others_dir.mkdir(parents=True, exist_ok=True)

    log_path = others_dir / f"workflow_{timestamp}.log"
    original_stdout = sys.stdout
    logger = DualLogger(str(log_path))
    try:
        sys.stdout = logger
        return _run_generation(config, output_dir, others_dir, log_path)
    finally:
        sys.stdout = original_stdout
        logger.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate symmetry-constrained crystal structures")
    parser.add_argument("--formula", default="CaCuBi")
    parser.add_argument("--spacegroup", type=int, default=225)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "epoch699.ckpt")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "generated_structures")
    parser.add_argument("--mace-model", default=os.getenv("MACE_MODEL", "medium"))
    parser.add_argument("--mace-device", default=os.getenv("MACE_DEVICE", "cpu"))
    parser.add_argument("--no-relax", action="store_true")
    args = parser.parse_args()

    result = run_workflow(
        WorkflowConfig(
            formula=args.formula,
            spacegroup_number=args.spacegroup,
            num_samples=args.samples,
            checkpoint_path=args.checkpoint,
            output_root=args.output_root,
            enable_relax=not args.no_relax,
            mace_model=args.mace_model,
            mace_device=args.mace_device,
        )
    )
    print(f"Accepted structures: {len(result.accepted_poscars)}")
    print(f"Output directory: {result.output_directory}")


if __name__ == "__main__":
    main()
