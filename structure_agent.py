"""Conversational Pydantic AI interface for symmetry-constrained generation."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime
import io
import os
import sys
import threading
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dotenv import dotenv_values, load_dotenv


# These optional-dependency and deprecation warnings are known and do not affect this CLI.
warnings.filterwarnings(
    "ignore",
    message=r"You are using `torch\.load` with `weights_only=False`.*",
    category=FutureWarning,
)
warnings.filterwarnings("ignore", category=UserWarning, module=r"torchvision\.io\.image")
warnings.filterwarnings("ignore", category=UserWarning, module=r"pyxtal\.molecule")


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PROJECT_ENV_FILE = PROJECT_ROOT / ".env.agent"
LEGACY_ENV_FILE = PROJECT_ROOT.parent / ".env.agent"
ACTIVE_ENV_FILE = PROJECT_ENV_FILE if PROJECT_ENV_FILE.is_file() else LEGACY_ENV_FILE
if PROJECT_ENV_FILE.is_file():
    load_dotenv(PROJECT_ENV_FILE, override=False)
elif LEGACY_ENV_FILE.is_file():
    legacy_values = dotenv_values(LEGACY_ENV_FILE)
    for variable_name in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        if value := legacy_values.get(variable_name):
            os.environ.setdefault(variable_name, value)


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _mace_model_value(value: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return str(candidate.resolve())
    project_candidate = PROJECT_ROOT / candidate
    return str(project_candidate.resolve()) if project_candidate.is_file() else value


def _configure_console_output() -> None:
    """Keep model responses from crashing legacy Windows code-page consoles."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


PYDANTIC_AI_REPO = _project_path(
    os.getenv("PYDANTIC_AI_SOURCE", PROJECT_ROOT / "vendor" / "pydantic-ai")
)
LOCAL_PACKAGE_METADATA = PROJECT_ROOT / ".local_packages"
BUNDLED_MACE_MODEL = PROJECT_ROOT / "macemodel" / "2023-12-03-mace-128-L1_epoch-199.model"

# Use the adjacent source checkout without copying framework code into this project.
for source_dir in (
    LOCAL_PACKAGE_METADATA,
    PYDANTIC_AI_REPO / "pydantic_ai_slim",
    PYDANTIC_AI_REPO / "pydantic_graph",
):
    if source_dir.is_dir() and str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

with contextlib.redirect_stdout(io.StringIO()):
    from band_result_analysis import (
        DEFAULT_RESULTS_ROOT,
        BandAccidentalDegeneracyReport as BandResultAnalysis,
        analyze_band_result,
    )
    from band_workflow import BandWorkflowConfig, check_band_environment, run_band_workflow
    from emergent_particles import DEFAULT_INDEX_PATH, lookup_emergent_particles
    from inverse_design.smc import SMCConfig, run_smc_generation
    from mace_energy import MaceEnergyConfig, calculate_mace_energy
    from workflow_sym import WorkflowConfig, atomic_numbers_from_formula, run_workflow


class CrystalGenerationRequest(BaseModel):
    formula: str = Field(
        min_length=1,
        description="Chemical formula using element symbols, for example BN, B2N2, CaCuBi, or LiFePO4.",
    )
    spacegroup_number: int = Field(
        ge=1,
        le=230,
        description="International Tables for Crystallography space-group number.",
    )
    num_samples: int | None = Field(
        default=None,
        ge=1,
        le=1000,
        description="Number of diffusion samples. Omit it to use the server default.",
    )


class GeneratedStructureReport(BaseModel):
    structure_number: int
    sample_index: int
    poscar_path: str
    actual_spacegroup_number: int
    actual_spacegroup_symbol: str
    mace_total_energy_ev: float | None
    mace_energy_per_atom_ev: float | None
    energy_description: str


class CrystalGenerationReport(BaseModel):
    formula: str
    atomic_numbers: list[int]
    spacegroup_number: int
    requested_samples: int
    unique_initial_structures: int
    accepted_count: int
    accepted_poscars: list[str]
    output_directory: str
    rejected_directory: str
    log_file: str
    structures: list[GeneratedStructureReport] = Field(default_factory=list)


class ParticleGuidedGenerationReport(BaseModel):
    formula: str
    spacegroup_number: int
    particle: str
    requested_particles: int
    method: str
    checkpoint_frozen: bool
    surrogate_validated_for_smc: bool
    valid_output_count: int
    composition_retained_count: int
    requested_spacegroup_retained_count: int
    condition_valid_count: int
    resampling_event_count: int
    probability_min: float
    probability_mean: float
    probability_max: float
    output_directory: str
    candidates_csv: str
    events_file: str
    summary_file: str
    scientific_scope: str


class GenerationAndBandReport(CrystalGenerationReport):
    band_requested_count: int
    band_completed_count: int
    band_failed_count: int
    band_images: list[str]
    bands_directory: str | None
    band_report_file: str | None
    band_log_file: str | None
    band_failures: list[str]


class StructureEnergyRequest(BaseModel):
    structure_filename: str = Field(
        min_length=1,
        description=(
            "Filename of a CIF or POSCAR structure placed directly in the configured "
            "structure input directory, for example graphene.cif or POSCAR_graphene."
        ),
    )


class MaceEnergyReport(BaseModel):
    structure_filename: str
    structure_path: str
    formula: str
    atom_count: int
    total_energy_ev: float
    energy_per_atom_ev: float
    calculation_type: str
    mace_model: str
    mace_device: str
    output_directory: str
    report_file: str


class HighSymmetryPathItem(BaseModel):
    line_label: str
    path: str
    source_pdf_page: int


class EmergentParticleItem(BaseModel):
    abbreviation: str
    name: str
    paths: list[HighSymmetryPathItem] | None = None


class EmergentParticleReport(BaseModel):
    spacegroup_number: int
    soc: bool
    essential: list[EmergentParticleItem]
    accidental: list[EmergentParticleItem]
    all_particles: list[EmergentParticleItem]
    source_title: str
    source_file: str
    source_table: str
    source_pdf_page: int
    path_source_section: str


class ParticlePathCandidateItem(BaseModel):
    abbreviation: str
    name: str
    line_label: str
    path: str
    source_pdf_page: int


class AccidentalCrossingItem(BaseModel):
    crossing_number: int
    branch: str
    high_symmetry_path: str
    line_label: str | None
    k_point_interval: tuple[int, int]
    k_point_coordinates: tuple[list[float], list[float]]
    band_indices: tuple[int, int]
    irreps_swapped: tuple[str, str]
    energy_relative_to_fermi_ev: float
    classification: str
    candidates: list[ParticlePathCandidateItem]


class PathMatchSummaryItem(BaseModel):
    high_symmetry_path: str
    crossing_count: int
    classification: str
    candidates: list[ParticlePathCandidateItem]


class EncyclopediaPathComparisonItem(BaseModel):
    high_symmetry_path: str
    line_label: str
    crossing_count: int
    abbreviation: str
    name: str
    classification: str
    source_pdf_page: int


class BandAccidentalDegeneracyReport(BaseModel):
    result_name: str
    result_directory: str
    band_job_directory: str
    band_image: str | None
    spacegroup_number: int
    spacegroup_symbol: str
    soc: bool
    detector: str
    energy_window_ev: float
    gap_tolerance_ev: float
    crossing_count: int
    confirmed_particle_types: list[str]
    path_compatible_particle_types: list[str]
    path_summaries: list[PathMatchSummaryItem]
    encyclopedia_path_table: list[EncyclopediaPathComparisonItem]
    crossings: list[AccidentalCrossingItem]
    source_title: str
    source_file: str
    source_table: str
    source_table_pdf_page: int
    source_path_section: str
    report_file: str
    scientific_scope: str


@dataclass
class AgentSessionState:
    last_generation: CrystalGenerationReport | None = None


@dataclass(frozen=True)
class AgentDependencies:
    checkpoint_path: Path
    output_root: Path
    mace_model: str
    mace_device: str
    enable_relax: bool
    default_samples: int
    band_analyzer_root: Path = PROJECT_ROOT / "band_analysis"
    band_python: Path = Path(sys.executable)
    band_timeout_seconds: int = 0
    band_output_root: Path | None = None
    structure_input_dir: Path = PROJECT_ROOT / "input_structures"
    energy_output_root: Path = PROJECT_ROOT / "calculation_results" / "mace_energy"
    emergent_particle_index: Path = DEFAULT_INDEX_PATH
    calculation_results_root: Path = DEFAULT_RESULTS_ROOT
    particle_model_root: Path = PROJECT_ROOT / "research_models" / "particles"
    smc_output_root: Path = PROJECT_ROOT / "generated_structures" / "particle_guided"
    smc_particles: int = 32
    smc_diffusion_steps: int = 500
    smc_resample_interval: int = 50
    smc_guidance_start_fraction: float = 0.5
    smc_alpha: float = 3.0
    smc_ess_threshold: float = 0.8
    smc_seed: int = 20260813
    smc_device: str = "auto"
    session_state: AgentSessionState = field(default_factory=AgentSessionState)


GENERATION_INSTRUCTIONS = """
你是一个晶体结构生成、MACE 能量与能带计算助手。把用户的自然语言请求转换为严格的工具参数。

规则：
1. 提取规范化学式、1 到 230 的空间群号，以及用户明确要求的采样数。
2. 中文材料名必须转换为化学式。例如“氮化硼”是 BN，“碳化硅”是 SiC；NaBi 保持为 NaBi。
3. 如果用户要求计算输入目录中已有 CIF 或 POSCAR 文件的“能量”“energy”，只调用
   calculate_structure_energy，传递用户提到的文件名；不要生成新结构，也不要调用能带工具。
4. 已有结构的 MACE 计算是未弛豫单点势能。工具完成后必须同时报告总能量（eV）和每原子能量（eV/atom），
   并说明这是 MACE 预测值。不要把它称为 DFT 能量、形成能或 energy above hull (E_hull)。
5. 如果用户要求“能带”“band”或生成后进行能带计算，只调用
   generate_structures_and_calculate_bands，不要再调用 generate_crystal_structures。
6. 如果用户只要求生成结构，只调用 generate_crystal_structures。
7. 用户没有给出样本数时省略 num_samples，让服务使用默认值。
8. 生成请求缺少化学式或空间群号，或能量请求缺少文件名时，先用一句简短问题补充信息，不调用工具。
9. 每个明确请求只调用一次对应工具，不要编造结构、计算结果或文件路径。
10. 工具完成后，准确报告采样数、通过验收的 POSCAR 数、完成的能带图片数、输出目录和失败信息。
11. 用户追问“第几个结构”的能量、形成能或空间群时，调用 get_generated_structure_details，
    structure_number 是最近一次生成任务中通过验收结构的 1-based 编号；不要重新生成或猜测。
12. 生成结构记录中的能量是弛豫后的 MACE 势能，不是严格热力学形成能。用户询问形成能时，
    明确说明当前未计算严格形成能，然后报告 MACE 总势能、每原子势能和实际空间群。
13. 用户询问某个空间群可能存在的“演生粒子”“emergent particle”或类似百科知识时，调用
    lookup_spacegroup_emergent_particles。用户明确说“考虑 SOC”“含 SOC”时 soc=true，明确说
    “不考虑 SOC”“无 SOC”时 soc=false；SOC 条件不明确时先询问，不要猜测。
14. 演生粒子查询结果必须分别列出 essential（本征/对称性强制）和 accidental（偶然简并）类别，
    再给出去重后的总列表，并注明补充材料表号和 PDF 页码。不要把 accidental 漏掉。
15. 用户询问偶然简并对应的“高对称路径”“高对称线”时，必须对每种 accidental 粒子列出工具返回的
    全部 paths，包括线路符号 line_label、端点 path 和 source_pdf_page；不要凭常见布里渊区路径补充或猜测。
16. 当前 paths 只索引 accidental 路径。essential 粒子的 paths=null 表示“本工具未索引”，不代表不存在路径；
    不要把它报告成“无”。用户只询问偶然简并时应聚焦 accidental 表，可省略 essential 对照表。
17. 用户要求“分析某材料结果中的偶然简并”或类似已完成能带结果判定时，只调用
    analyze_calculated_band_accidental_degeneracies，result_name 传材料名或结果目录名，不要重新计算能带，
    也不要仅调用百科查询工具。该工具会自动定位 calculation_results 下的结果、复现 band 图红圈使用的
    表示对换检测，并与对应空间群和 SOC 模式的百科路径索引比对。
18. 对结果判定必须区分 classification：confirmed_by_unique_path 表示该高对称路径在索引中只匹配一种
    accidental 粒子，可报告为路径唯一确认；path_compatible_ambiguous 表示同一路径对应多种粒子，只能列为
    路径相容候选，不能擅自唯一命名；not_indexed_for_this_path 表示检测到表示对换但索引无对应路径。
19. 回答结果分析时先汇总空间群、SOC、红圈/表示对换总数，再按高对称路径列出数量、候选粒子。在该汇总表
    后必须单独输出“百科全书高对称路径对照表”，逐行完整展示 encyclopedia_path_table，不得合并或省略行；
    表格列至少包括高对称路径、线标、红圈数、百科允许粒子缩写、英文名称、匹配结论和补充材料 PDF 页码。
    然后列出每个 crossing 的相对费米能量、能带编号和 irreps_swapped，并注明报告文件与补充材料表号/路径章节。
20. 空间群符号只能逐字使用工具返回的 spacegroup_symbol，不要根据编号自行回忆或补写符号。
21. 用户要求“可能具有 DP/DNL”“面向 DP/DNL”或“粒子引导生成”时，只调用
    generate_particle_guided_structures。particle 必须是 DP 或 DNL；样本数传给 num_particles。
22. 粒子引导结果是冻结 SymmCD checkpoint 的 TDS-inspired SMC 筛选结果，不是拓扑确认。
    必须报告代理概率范围、有效结构数、目标空间群保持数和输出目录，并说明仍需 SOC DFT/IRVSP 验证。
""".strip()


_generation_lock = threading.Lock()


def _run_structure_energy_request(
    deps: AgentDependencies,
    structure_filename: str,
) -> MaceEnergyReport:
    request = StructureEnergyRequest(structure_filename=structure_filename)
    result = calculate_mace_energy(
        MaceEnergyConfig(
            structure_filename=request.structure_filename,
            input_directory=deps.structure_input_dir,
            output_root=deps.energy_output_root,
            mace_model=deps.mace_model,
            mace_device=deps.mace_device,
        )
    )
    return MaceEnergyReport(**asdict(result))


def _run_emergent_particle_lookup(
    deps: AgentDependencies,
    spacegroup_number: int,
    soc: bool,
) -> EmergentParticleReport:
    result = lookup_emergent_particles(
        spacegroup_number=spacegroup_number,
        soc=soc,
        index_path=deps.emergent_particle_index,
    )
    report_data = asdict(result)
    for key in ("essential", "accidental", "all_particles"):
        report_data.pop(key)
    return EmergentParticleReport(
        **report_data,
        essential=[EmergentParticleItem(**asdict(item)) for item in result.essential],
        accidental=[EmergentParticleItem(**asdict(item)) for item in result.accidental],
        all_particles=[EmergentParticleItem(**asdict(item)) for item in result.all_particles],
    )


def _run_band_result_analysis(
    deps: AgentDependencies,
    result_name: str,
) -> BandAccidentalDegeneracyReport:
    result: BandResultAnalysis = analyze_band_result(
        result_name=result_name,
        results_root=deps.calculation_results_root,
        emergent_index=deps.emergent_particle_index,
        analyzer_root=deps.band_analyzer_root,
    )
    return BandAccidentalDegeneracyReport.model_validate(asdict(result))


def _run_generation_request(
    deps: AgentDependencies,
    formula: str,
    spacegroup_number: int,
    num_samples: int | None,
) -> CrystalGenerationReport:
    request = CrystalGenerationRequest(
        formula=formula,
        spacegroup_number=spacegroup_number,
        num_samples=num_samples,
    )
    atoms = atomic_numbers_from_formula(request.formula)
    sample_count = request.num_samples or deps.default_samples
    config = WorkflowConfig(
        formula=request.formula,
        spacegroup_number=request.spacegroup_number,
        num_samples=sample_count,
        checkpoint_path=deps.checkpoint_path,
        output_root=deps.output_root,
        enable_relax=deps.enable_relax,
        mace_model=deps.mace_model,
        mace_device=deps.mace_device,
    )

    if not _generation_lock.acquire(blocking=False):
        raise RuntimeError("another crystal-generation job is already running")
    try:
        result = run_workflow(config)
    finally:
        _generation_lock.release()

    report = CrystalGenerationReport(
        formula=result.formula,
        atomic_numbers=atoms,
        spacegroup_number=result.spacegroup_number,
        requested_samples=result.requested_samples,
        unique_initial_structures=result.unique_initial_structures,
        accepted_count=len(result.accepted_poscars),
        accepted_poscars=result.accepted_poscars,
        output_directory=result.output_directory,
        rejected_directory=result.rejected_directory,
        log_file=result.log_file,
        structures=[GeneratedStructureReport(**asdict(item)) for item in result.structures],
    )
    deps.session_state.last_generation = report
    return report


def _run_particle_guided_generation_request(
    deps: AgentDependencies,
    formula: str,
    spacegroup_number: int,
    particle: str,
    num_particles: int | None,
) -> ParticleGuidedGenerationReport:
    normalized_particle = particle.strip().upper()
    if normalized_particle not in {"DP", "DNL"}:
        raise ValueError("particle 必须是 DP 或 DNL")
    particle_count = num_particles or deps.smc_particles
    if not 2 <= particle_count <= 1000:
        raise ValueError("num_particles 必须在 2 到 1000 之间")
    model_path = deps.particle_model_root / (
        f"particle_{normalized_particle.lower()}_surrogate.joblib"
    )
    if not model_path.is_file():
        raise FileNotFoundError(
            f"尚未训练 {normalized_particle} 粒子代理模型: {model_path}"
        )
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = (
        deps.smc_output_root
        / f"{normalized_particle}_{formula}_sg{spacegroup_number}_{timestamp}"
    )
    config = SMCConfig(
        formula=formula,
        spacegroup_number=spacegroup_number,
        model_path=model_path,
        output_dir=output_dir,
        checkpoint_path=deps.checkpoint_path,
        particles=particle_count,
        diffusion_steps=deps.smc_diffusion_steps,
        resample_interval=deps.smc_resample_interval,
        guidance_start_fraction=deps.smc_guidance_start_fraction,
        alpha=deps.smc_alpha,
        ess_threshold=deps.smc_ess_threshold,
        seed=deps.smc_seed,
        device=deps.smc_device,
    )
    if not _generation_lock.acquire(blocking=False):
        raise RuntimeError("another crystal-generation job is already running")
    try:
        summary = run_smc_generation(config)
    finally:
        _generation_lock.release()
    return ParticleGuidedGenerationReport(
        formula=formula,
        spacegroup_number=spacegroup_number,
        particle=normalized_particle,
        requested_particles=particle_count,
        method=summary["method"],
        checkpoint_frozen=summary["checkpoint_frozen"],
        surrogate_validated_for_smc=summary["surrogate_validated_for_smc"],
        valid_output_count=summary["valid_output_count"],
        composition_retained_count=summary["composition_retained_count"],
        requested_spacegroup_retained_count=summary["requested_spacegroup_retained_count"],
        condition_valid_count=summary["condition_valid_count"],
        resampling_event_count=summary["resampling_event_count"],
        probability_min=summary["probability_min"],
        probability_mean=summary["probability_mean"],
        probability_max=summary["probability_max"],
        output_directory=str(output_dir.resolve()),
        candidates_csv=str((output_dir / "smc_candidates.csv").resolve()),
        events_file=str((output_dir / "smc_events.json").resolve()),
        summary_file=str((output_dir / "smc_summary.json").resolve()),
        scientific_scope=(
            "代理模型引导的候选排序，不构成拓扑确认；必须继续进行 SOC DFT 能带和 IRVSP 验证。"
        ),
    )


def _get_generated_structure_details(
    deps: AgentDependencies,
    structure_number: int,
) -> GeneratedStructureReport:
    generation = deps.session_state.last_generation
    if generation is None:
        raise RuntimeError("当前会话中还没有结构生成结果，请先生成结构")
    if structure_number < 1 or structure_number > len(generation.structures):
        raise ValueError(
            f"结构编号必须在 1 到 {len(generation.structures)} 之间；"
            f"最近一次任务通过验收的结构数为 {len(generation.structures)}"
        )
    return generation.structures[structure_number - 1]


def _build_model(model_name: Any = None) -> Any:
    if model_name is not None and not isinstance(model_name, str):
        return model_name

    base_url = os.getenv("LLM_BASE_URL")
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    configured_model = model_name or os.getenv("LLM_MODEL", "gpt-5.2")

    if base_url:
        provider = OpenAIProvider(base_url=base_url, api_key=api_key)
        return OpenAIChatModel(configured_model, provider=provider)

    if ":" in configured_model:
        return configured_model
    return f"openai:{configured_model}"


def create_agent(model_name: Any = None) -> Agent[AgentDependencies, str]:
    agent = Agent(
        _build_model(model_name),
        deps_type=AgentDependencies,
        instructions=GENERATION_INSTRUCTIONS,
        retries=2,
    )

    @agent.tool
    def calculate_structure_energy(
        ctx: RunContext[AgentDependencies],
        structure_filename: str,
    ) -> MaceEnergyReport:
        """Calculate an unrelaxed MACE single-point energy for an input CIF or POSCAR file."""
        return _run_structure_energy_request(ctx.deps, structure_filename)

    @agent.tool
    def lookup_spacegroup_emergent_particles(
        ctx: RunContext[AgentDependencies],
        spacegroup_number: int,
        soc: bool,
    ) -> EmergentParticleReport:
        """Look up all essential and accidental emergent particles for one space group."""
        return _run_emergent_particle_lookup(ctx.deps, spacegroup_number, soc)

    @agent.tool
    def analyze_calculated_band_accidental_degeneracies(
        ctx: RunContext[AgentDependencies],
        result_name: str,
    ) -> BandAccidentalDegeneracyReport:
        """Analyze plotted irrep-exchange crossings and classify them using the local encyclopedia."""
        return _run_band_result_analysis(ctx.deps, result_name)

    @agent.tool
    def generate_crystal_structures(
        ctx: RunContext[AgentDependencies],
        formula: str,
        spacegroup_number: int,
        num_samples: int | None = None,
    ) -> CrystalGenerationReport:
        """Generate and optionally relax crystals for a formula and space-group number."""
        return _run_generation_request(
            ctx.deps,
            formula=formula,
            spacegroup_number=spacegroup_number,
            num_samples=num_samples,
        )

    @agent.tool
    def generate_particle_guided_structures(
        ctx: RunContext[AgentDependencies],
        formula: str,
        spacegroup_number: int,
        particle: str,
        num_particles: int | None = None,
    ) -> ParticleGuidedGenerationReport:
        """Generate DP- or DNL-enriched candidates using frozen-checkpoint SMC guidance."""
        return _run_particle_guided_generation_request(
            ctx.deps,
            formula=formula,
            spacegroup_number=spacegroup_number,
            particle=particle,
            num_particles=num_particles,
        )

    @agent.tool
    def get_generated_structure_details(
        ctx: RunContext[AgentDependencies],
        structure_number: int,
    ) -> GeneratedStructureReport:
        """Return energy and actual space group for an accepted structure from the latest generation."""
        return _get_generated_structure_details(ctx.deps, structure_number)

    @agent.tool
    def generate_structures_and_calculate_bands(
        ctx: RunContext[AgentDependencies],
        formula: str,
        spacegroup_number: int,
        num_samples: int | None = None,
    ) -> GenerationAndBandReport:
        """Generate accepted POSCAR files, then calculate and plot their electronic bands."""
        check_band_environment(ctx.deps.band_analyzer_root, ctx.deps.band_python)
        generation = _run_generation_request(
            ctx.deps,
            formula=formula,
            spacegroup_number=spacegroup_number,
            num_samples=num_samples,
        )
        generation_job_dir = Path(generation.output_directory).parent
        if ctx.deps.band_output_root is None:
            band_output_root = generation_job_dir / "band_analysis"
        else:
            band_output_root = (
                ctx.deps.band_output_root
                / generation_job_dir.parent.name
                / generation_job_dir.name
            )
        if not generation.accepted_poscars:
            return GenerationAndBandReport(
                **generation.model_dump(),
                band_requested_count=0,
                band_completed_count=0,
                band_failed_count=0,
                band_images=[],
                bands_directory=str((band_output_root / "bands").resolve()),
                band_report_file=None,
                band_log_file=None,
                band_failures=[],
            )

        band_result = run_band_workflow(
            BandWorkflowConfig(
                structure_paths=[Path(path) for path in generation.accepted_poscars],
                output_root=band_output_root,
                analyzer_root=ctx.deps.band_analyzer_root,
                python_executable=ctx.deps.band_python,
                timeout_seconds=ctx.deps.band_timeout_seconds,
            )
        )
        return GenerationAndBandReport(
            **generation.model_dump(),
            band_requested_count=band_result.requested_count,
            band_completed_count=band_result.completed_count,
            band_failed_count=band_result.failed_count,
            band_images=band_result.band_images,
            bands_directory=band_result.bands_directory,
            band_report_file=band_result.report_file,
            band_log_file=band_result.log_file,
            band_failures=band_result.failures,
        )

    return agent


def build_dependencies(args: argparse.Namespace) -> AgentDependencies:
    return AgentDependencies(
        checkpoint_path=args.checkpoint.resolve(),
        output_root=args.output_root.resolve(),
        mace_model=_mace_model_value(args.mace_model),
        mace_device=args.mace_device,
        enable_relax=not args.no_relax,
        default_samples=args.default_samples,
        band_analyzer_root=args.band_analyzer_root.resolve(),
        band_python=args.band_python.resolve(),
        band_timeout_seconds=args.band_timeout_seconds,
        band_output_root=args.band_output_root.resolve() if args.band_output_root else None,
        structure_input_dir=args.structure_input_dir.resolve(),
        energy_output_root=args.energy_output_root.resolve(),
        emergent_particle_index=args.emergent_index.resolve(),
        calculation_results_root=args.calculation_results_root.resolve(),
        particle_model_root=args.particle_model_root.resolve(),
        smc_output_root=args.smc_output_root.resolve(),
        smc_particles=args.smc_particles,
        smc_diffusion_steps=args.smc_diffusion_steps,
        smc_resample_interval=args.smc_resample_interval,
        smc_guidance_start_fraction=args.smc_guidance_start_fraction,
        smc_alpha=args.smc_alpha,
        smc_ess_threshold=args.smc_ess_threshold,
        smc_seed=args.smc_seed,
        smc_device=args.smc_device,
    )


async def _run_interactive_cli(
    agent: Agent[AgentDependencies, str],
    deps: AgentDependencies,
) -> None:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from rich.console import Console
    from rich.markdown import Markdown

    history_path = PROJECT_ROOT / ".agent_history" / "prompt-history.txt"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.touch(exist_ok=True)
    session = PromptSession(history=FileHistory(str(history_path)))
    console = Console()
    messages = []

    while True:
        try:
            prompt = await session.prompt_async("pydantic ➤ ")
        except EOFError:
            return
        except KeyboardInterrupt:
            console.print("[dim]已取消当前输入。输入 /exit 退出。[/dim]")
            continue

        prompt = prompt.strip()
        if not prompt:
            continue
        command = prompt.lower().replace(" ", "-")
        if command in {"/exit", "/quit"}:
            return
        if command == "/clear":
            messages = []
            console.print("[dim]当前对话上下文已清空。[/dim]")
            continue
        if command == "/help":
            console.print("[dim]可用命令：/clear 清空上下文，/exit 退出。[/dim]")
            continue
        if command.startswith("/"):
            console.print(f"[yellow]未知命令：{prompt}[/yellow]")
            continue

        console.print("[dim]正在处理请求...[/dim]")
        try:
            result = await agent.run(prompt, message_history=messages, deps=deps)
        except asyncio.CancelledError:
            console.print("[dim]请求已取消。[/dim]")
            continue
        except Exception as exc:
            console.print(f"[red]{type(exc).__name__}:[/red] {exc}")
            if cause := getattr(exc, "__cause__", None):
                console.print(f"[dim]Caused by: {cause}[/dim]")
            continue

        messages = result.all_messages()
        console.print(Markdown(str(result.output), code_theme="monokai"))


def main() -> None:
    _configure_console_output()
    parser = argparse.ArgumentParser(description="Chat with the SymmCD structure-generation agent")
    parser.add_argument("--prompt", help="Run one request instead of opening interactive chat")
    parser.add_argument("--model", help="Pydantic AI model name or OpenAI-compatible model ID")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "epoch699.ckpt")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "generated_structures")
    parser.add_argument("--default-samples", type=int, default=10)
    parser.add_argument(
        "--particle-model-root",
        type=Path,
        default=PROJECT_ROOT / "research_models" / "particles",
    )
    parser.add_argument(
        "--smc-output-root",
        type=Path,
        default=PROJECT_ROOT / "generated_structures" / "particle_guided",
    )
    parser.add_argument("--smc-particles", type=int, default=32)
    parser.add_argument("--smc-diffusion-steps", type=int, default=500)
    parser.add_argument("--smc-resample-interval", type=int, default=50)
    parser.add_argument("--smc-guidance-start-fraction", type=float, default=0.5)
    parser.add_argument("--smc-alpha", type=float, default=3.0)
    parser.add_argument("--smc-ess-threshold", type=float, default=0.8)
    parser.add_argument("--smc-seed", type=int, default=20260813)
    parser.add_argument("--smc-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--mace-model",
        default=os.getenv(
            "MACE_MODEL",
            str(BUNDLED_MACE_MODEL) if BUNDLED_MACE_MODEL.is_file() else "medium",
        ),
    )
    parser.add_argument("--mace-device", default=os.getenv("MACE_DEVICE", "cpu"))
    parser.add_argument(
        "--structure-input-dir",
        type=Path,
        default=_project_path(os.getenv("STRUCTURE_INPUT_DIR", PROJECT_ROOT / "input_structures")),
        help="Directory containing user-provided CIF and POSCAR files.",
    )
    parser.add_argument(
        "--energy-output-root",
        type=Path,
        default=_project_path(
            os.getenv("ENERGY_OUTPUT_ROOT", PROJECT_ROOT / "calculation_results" / "mace_energy")
        ),
        help="Root directory for MACE single-point energy reports.",
    )
    parser.add_argument(
        "--emergent-index",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help="Prebuilt emergent-particle index derived from supplemental Tables S1/S2.",
    )
    parser.add_argument(
        "--calculation-results-root",
        type=Path,
        default=_project_path(
            os.getenv("CALCULATION_RESULTS_ROOT", PROJECT_ROOT / "calculation_results")
        ),
        help="Root containing completed material band-result directories.",
    )
    parser.add_argument(
        "--band-analyzer-root",
        type=Path,
        default=_project_path(os.getenv("BAND_ANALYZER_ROOT", PROJECT_ROOT / "band_analysis")),
    )
    parser.add_argument(
        "--band-python",
        type=Path,
        default=Path(os.getenv("BAND_PYTHON", sys.executable)),
        help="Python executable from the environment containing atomate2 and jobflow.",
    )
    parser.add_argument(
        "--band-timeout-seconds",
        type=int,
        default=int(os.getenv("BAND_TIMEOUT_SECONDS", "0")),
        help="Timeout for the complete band workflow; 0 means no timeout.",
    )
    parser.add_argument(
        "--band-output-root",
        type=Path,
        default=_project_path(os.environ["BAND_OUTPUT_ROOT"]) if os.getenv("BAND_OUTPUT_ROOT") else None,
        help="Optional root directory for band jobs and images; defaults to each generation run.",
    )
    parser.add_argument("--no-relax", action="store_true")
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Validate local paths and print non-secret runtime settings without calling an API.",
    )
    parser.add_argument(
        "--check-band",
        action="store_true",
        help="Validate the configured atomate2/jobflow/IRVSP environment without running a calculation.",
    )
    args = parser.parse_args()
    deps = build_dependencies(args)

    if not 1 <= deps.default_samples <= 1000:
        parser.error("--default-samples must be between 1 and 1000")
    if not 2 <= deps.smc_particles <= 1000:
        parser.error("--smc-particles must be between 2 and 1000")
    if deps.smc_diffusion_steps < 2 or deps.smc_resample_interval < 1:
        parser.error("SMC diffusion steps and resample interval must be positive")
    if not 0 <= deps.smc_guidance_start_fraction < 1:
        parser.error("--smc-guidance-start-fraction must be in [0, 1)")
    if deps.smc_alpha < 0 or not 0 < deps.smc_ess_threshold <= 1:
        parser.error("SMC alpha must be nonnegative and ESS threshold must be in (0, 1]")
    if not deps.checkpoint_path.is_file():
        parser.error(f"checkpoint not found: {deps.checkpoint_path}")
    if deps.band_timeout_seconds < 0:
        parser.error("--band-timeout-seconds cannot be negative")
    if not deps.structure_input_dir.is_dir():
        parser.error(f"structure input directory not found: {deps.structure_input_dir}")
    if not deps.emergent_particle_index.is_file():
        parser.error(f"emergent-particle index not found: {deps.emergent_particle_index}")
    if not deps.calculation_results_root.is_dir():
        parser.error(f"calculation-results directory not found: {deps.calculation_results_root}")

    if args.show_config:
        print(asdict(deps))
        print(f"agent_env_file={ACTIVE_ENV_FILE if ACTIVE_ENV_FILE.is_file() else 'not found'}")
        print(f"pydantic_ai_source={PYDANTIC_AI_REPO}")
        return
    if args.check_band:
        try:
            print(check_band_environment(deps.band_analyzer_root, deps.band_python))
        except (FileNotFoundError, RuntimeError) as exc:
            parser.error(str(exc))
        return

    agent = create_agent(args.model)
    if args.prompt:
        result = agent.run_sync(args.prompt, deps=deps)
        print(result.output)
    else:
        asyncio.run(_run_interactive_cli(agent, deps))


if __name__ == "__main__":
    main()
