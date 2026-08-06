"""Conversational Pydantic AI interface for symmetry-constrained generation."""

from __future__ import annotations

import argparse
import os
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env.agent", override=False)
PYDANTIC_AI_REPO = Path(
    os.getenv("PYDANTIC_AI_SOURCE", str(PROJECT_ROOT / "vendor" / "pydantic-ai"))
).resolve()
LOCAL_PACKAGE_METADATA = PROJECT_ROOT / ".local_packages"

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

from band_workflow import BandWorkflowConfig, check_band_environment, run_band_workflow
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


class GenerationAndBandReport(CrystalGenerationReport):
    band_requested_count: int
    band_completed_count: int
    band_failed_count: int
    band_images: list[str]
    bands_directory: str | None
    band_report_file: str | None
    band_log_file: str | None
    band_failures: list[str]


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


GENERATION_INSTRUCTIONS = """
你是一个晶体结构生成与能带计算助手。把用户的自然语言请求转换为严格的工具参数。

规则：
1. 提取规范化学式、1 到 230 的空间群号，以及用户明确要求的采样数。
2. 中文材料名必须转换为化学式。例如“氮化硼”是 BN，“碳化硅”是 SiC；NaBi 保持为 NaBi。
3. 如果用户要求“能带”“band”或生成后进行能带计算，只调用
   generate_structures_and_calculate_bands，不要再调用 generate_crystal_structures。
4. 如果用户只要求生成结构，只调用 generate_crystal_structures。
5. 用户没有给出样本数时省略 num_samples，让服务使用默认值。
6. 化学式或空间群号缺失时，先用一句简短问题补充信息，不调用工具。
7. 每个明确请求只调用一次对应工具，不要编造结构、计算结果或文件路径。
8. 工具完成后，准确报告采样数、通过验收的 POSCAR 数、完成的能带图片数、输出目录和失败信息。
""".strip()


_generation_lock = threading.Lock()


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

    return CrystalGenerationReport(
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
    )


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
        mace_model=args.mace_model,
        mace_device=args.mace_device,
        enable_relax=not args.no_relax,
        default_samples=args.default_samples,
        band_analyzer_root=args.band_analyzer_root.resolve(),
        band_python=args.band_python.resolve(),
        band_timeout_seconds=args.band_timeout_seconds,
        band_output_root=args.band_output_root.resolve() if args.band_output_root else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the SymmCD structure-generation agent")
    parser.add_argument("--prompt", help="Run one request instead of opening interactive chat")
    parser.add_argument("--model", help="Pydantic AI model name or OpenAI-compatible model ID")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "epoch699.ckpt")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "generated_structures")
    parser.add_argument("--default-samples", type=int, default=10)
    parser.add_argument("--mace-model", default=os.getenv("MACE_MODEL", "medium"))
    parser.add_argument("--mace-device", default=os.getenv("MACE_DEVICE", "cpu"))
    parser.add_argument(
        "--band-analyzer-root",
        type=Path,
        default=Path(os.getenv("BAND_ANALYZER_ROOT", PROJECT_ROOT / "band_analysis")),
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
        default=Path(os.environ["BAND_OUTPUT_ROOT"]) if os.getenv("BAND_OUTPUT_ROOT") else None,
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
    if not deps.checkpoint_path.is_file():
        parser.error(f"checkpoint not found: {deps.checkpoint_path}")
    if deps.band_timeout_seconds < 0:
        parser.error("--band-timeout-seconds cannot be negative")

    if args.show_config:
        print(asdict(deps))
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
        agent.to_cli_sync(deps=deps, prog_name="symmcd-agent")


if __name__ == "__main__":
    main()
