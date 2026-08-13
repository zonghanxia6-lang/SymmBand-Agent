import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import structure_agent

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from band_workflow import BandWorkflowResult
from mace_energy import MaceEnergyResult
from workflow_sym import GeneratedStructureRecord, WorkflowResult, atomic_numbers_from_formula


class AgentTests(unittest.TestCase):
    def test_integrated_runtime_sources_are_inside_project(self):
        project_root = Path(structure_agent.__file__).resolve().parent
        self.assertEqual(
            structure_agent.PYDANTIC_AI_REPO,
            project_root / "vendor" / "pydantic-ai",
        )
        self.assertTrue(
            (project_root / "band_analysis" / "agent_runner.py").is_file()
        )

    def test_formula_expansion(self):
        self.assertEqual(atomic_numbers_from_formula("BN"), [5, 7])
        self.assertEqual(atomic_numbers_from_formula("B2N2"), [5, 5, 7, 7])

    def test_interactive_cli_prints_each_turn_once_and_keeps_history(self):
        class FakeSession:
            def __init__(self):
                self.prompts = iter(["第一轮", "第二轮", "/exit"])

            async def prompt_async(self, _prompt):
                return next(self.prompts)

        class FakeConsole:
            def __init__(self):
                self.output = []

            def print(self, value):
                self.output.append(str(value))

        class FakeResult:
            def __init__(self, output, messages):
                self.output = output
                self._messages = messages

            def all_messages(self):
                return self._messages

        class FakeAgent:
            def __init__(self):
                self.calls = []

            async def run(self, prompt, message_history, deps):
                self.calls.append((prompt, list(message_history), deps))
                turn = len(self.calls)
                return FakeResult(f"回答{turn}", [f"history-{turn}"])

        fake_session = FakeSession()
        fake_console = FakeConsole()
        fake_agent = FakeAgent()
        deps = object()

        with (
            patch("prompt_toolkit.PromptSession", return_value=fake_session),
            patch("prompt_toolkit.history.FileHistory"),
            patch("rich.console.Console", return_value=fake_console),
            patch("rich.markdown.Markdown", side_effect=lambda content, code_theme: content),
        ):
            asyncio.run(structure_agent._run_interactive_cli(fake_agent, deps))

        self.assertEqual(fake_agent.calls[0][1], [])
        self.assertEqual(fake_agent.calls[1][1], ["history-1"])
        processing_lines = [line for line in fake_console.output if "正在处理请求" in line]
        self.assertEqual(len(processing_lines), 2)
        self.assertEqual(sum("回答1" in line for line in fake_console.output), 1)
        self.assertEqual(sum("回答2" in line for line in fake_console.output), 1)

    def test_soc_emergent_particle_index_for_spacegroup_194(self):
        report = structure_agent._run_emergent_particle_lookup(
            structure_agent.AgentDependencies(
                checkpoint_path=Path("epoch699.ckpt").resolve(),
                output_root=Path("generated_structures").resolve(),
                mace_model="medium",
                mace_device="cpu",
                enable_relax=True,
                default_samples=10,
            ),
            spacegroup_number=194,
            soc=True,
        )

        self.assertEqual(
            [item.abbreviation for item in report.essential],
            ["DNL", "DNL net"],
        )
        self.assertEqual(
            [item.abbreviation for item in report.accidental],
            ["DP", "QDP", "DNL"],
        )
        self.assertEqual(
            [item.abbreviation for item in report.all_particles],
            ["DNL", "DNL net", "DP", "QDP"],
        )
        self.assertEqual(report.source_table, "S2")
        self.assertEqual(report.source_pdf_page, 8)

    def test_soc_accidental_paths_for_spacegroup_216(self):
        report = structure_agent._run_emergent_particle_lookup(
            structure_agent.AgentDependencies(
                checkpoint_path=Path("epoch699.ckpt").resolve(),
                output_root=Path("generated_structures").resolve(),
                mace_model="medium",
                mace_device="cpu",
                enable_relax=True,
                default_samples=10,
            ),
            spacegroup_number=216,
            soc=True,
        )

        paths = {
            item.abbreviation: [(entry.line_label, entry.path) for entry in item.paths]
            for item in report.accidental
        }
        self.assertEqual(paths["C-1 WP"], [("Z", "XW")])
        self.assertEqual(paths["TP"], [("Λ", "ΓL")])
        self.assertEqual(paths["WNL"], [("Λ", "ΓL"), ("Σ", "ΓΣ"), ("S", "XS")])
        self.assertEqual(paths["WNL net"], [("Λ", "ΓL")])
        self.assertTrue(
            all(
                path.source_pdf_page == 1070
                for item in report.accidental
                for path in item.paths
            )
        )
        self.assertEqual(report.path_source_section, "S8B")

    def test_agent_dispatches_emergent_particle_lookup(self):
        def model_function(messages, _info):
            model_steps = sum(isinstance(message, ModelResponse) for message in messages)
            if model_steps == 0:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "lookup_spacegroup_emergent_particles",
                            {"spacegroup_number": 194, "soc": True},
                        )
                    ]
                )
            return ModelResponse(
                parts=[TextPart("含 SOC 时共有 DNL、DNL net、DP 和 QDP。")]
            )

        deps = structure_agent.AgentDependencies(
            checkpoint_path=Path("epoch699.ckpt").resolve(),
            output_root=Path("generated_structures").resolve(),
            mace_model="medium",
            mace_device="cpu",
            enable_relax=True,
            default_samples=10,
        )
        agent = structure_agent.create_agent(FunctionModel(model_function))
        result = agent.run_sync(
            "我想知道194号空间群考虑SOC时所有可能存在的演生粒子有哪些",
            deps=deps,
        )

        self.assertEqual(result.output, "含 SOC 时共有 DNL、DNL net、DP 和 QDP。")

    def test_agent_dispatches_bn_spacegroup_194(self):
        captured = []

        def fake_workflow(config):
            captured.append(config)
            return WorkflowResult(
                formula=config.formula,
                spacegroup_number=config.spacegroup_number,
                requested_samples=config.num_samples,
                unique_initial_structures=1,
                accepted_poscars=[str(output_root / "POSCAR_BN_sg194_001")],
                output_directory=str(output_root / "accepted"),
                rejected_directory=str(output_root / "others"),
                log_file=str(output_root / "workflow.log"),
            )

        def model_function(messages, _info):
            model_steps = sum(isinstance(message, ModelResponse) for message in messages)
            if model_steps == 0:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "generate_crystal_structures",
                            {"formula": "BN", "spacegroup_number": 194},
                        )
                    ]
                )
            return ModelResponse(parts=[TextPart("已完成氮化硼结构生成。")])

        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            deps = structure_agent.AgentDependencies(
                checkpoint_path=Path("epoch699.ckpt").resolve(),
                output_root=output_root,
                mace_model="medium",
                mace_device="cpu",
                enable_relax=True,
                default_samples=10,
            )
            agent = structure_agent.create_agent(FunctionModel(model_function))
            with patch.object(structure_agent, "run_workflow", side_effect=fake_workflow):
                result = agent.run_sync("我要生成194号空间群的氮化硼结构", deps=deps)

        self.assertEqual(result.output, "已完成氮化硼结构生成。")
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].formula, "BN")
        self.assertEqual(captured[0].spacegroup_number, 194)
        self.assertEqual(captured[0].num_samples, 10)

    def test_agent_dispatches_dp_guided_generation(self):
        captured = []

        def fake_smc(config):
            captured.append(config)
            config.output_dir.mkdir(parents=True)
            return {
                "method": "TDS-inspired sequential Monte Carlo twisting",
                "checkpoint_frozen": True,
                "surrogate_validated_for_smc": True,
                "valid_output_count": 8,
                "composition_retained_count": 8,
                "requested_spacegroup_retained_count": 7,
                "condition_valid_count": 7,
                "resampling_event_count": 2,
                "probability_min": 0.2,
                "probability_mean": 0.6,
                "probability_max": 0.9,
            }

        def model_function(messages, _info):
            model_steps = sum(isinstance(message, ModelResponse) for message in messages)
            if model_steps == 0:
                return ModelResponse(parts=[ToolCallPart(
                    "generate_particle_guided_structures",
                    {
                        "formula": "BN",
                        "spacegroup_number": 194,
                        "particle": "DP",
                        "num_particles": 10,
                    },
                )])
            return ModelResponse(parts=[TextPart("DP 引导生成完成。")])

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_root = root / "models"
            model_root.mkdir()
            (model_root / "particle_dp_surrogate.joblib").touch()
            deps = structure_agent.AgentDependencies(
                checkpoint_path=Path("epoch699.ckpt").resolve(),
                output_root=root / "generated",
                mace_model="medium",
                mace_device="cpu",
                enable_relax=False,
                default_samples=10,
                particle_model_root=model_root,
                smc_output_root=root / "smc",
            )
            agent = structure_agent.create_agent(FunctionModel(model_function))
            with patch.object(structure_agent, "run_smc_generation", side_effect=fake_smc):
                result = agent.run_sync(
                    "生成10个可能具有DP的194号空间群BN结构", deps=deps
                )

        self.assertEqual(result.output, "DP 引导生成完成。")
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].particles, 10)
        self.assertEqual(captured[0].spacegroup_number, 194)
        self.assertEqual(captured[0].alpha, 3.0)

    def test_follow_up_returns_third_generated_structure_from_session(self):
        def fake_workflow(config):
            structures = [
                GeneratedStructureRecord(
                    structure_number=index,
                    sample_index=index + 1,
                    poscar_path=str(output_root / f"POSCAR_BN_sg194_{index:03d}"),
                    actual_spacegroup_number=194,
                    actual_spacegroup_symbol="P6_3/mmc",
                    mace_total_energy_ev=-20.0 - index,
                    mace_energy_per_atom_ev=(-20.0 - index) / 2,
                )
                for index in range(1, 4)
            ]
            return WorkflowResult(
                formula=config.formula,
                spacegroup_number=config.spacegroup_number,
                requested_samples=config.num_samples,
                unique_initial_structures=3,
                accepted_poscars=[item.poscar_path for item in structures],
                output_directory=str(output_root / "accepted"),
                rejected_directory=str(output_root / "others"),
                log_file=str(output_root / "workflow.log"),
                structures=structures,
            )

        def model_function(messages, _info):
            model_steps = sum(isinstance(message, ModelResponse) for message in messages)
            if model_steps == 0:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "generate_crystal_structures",
                            {"formula": "BN", "spacegroup_number": 194, "num_samples": 10},
                        )
                    ]
                )
            if model_steps == 1:
                return ModelResponse(parts=[TextPart("已生成并记录通过验收的结构。")])
            if model_steps == 2:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "get_generated_structure_details",
                            {"structure_number": 3},
                        )
                    ]
                )
            return ModelResponse(
                parts=[TextPart("第三个结构的 MACE 总势能为 -23 eV，实际空间群为 194。")]
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            deps = structure_agent.AgentDependencies(
                checkpoint_path=Path("epoch699.ckpt").resolve(),
                output_root=output_root,
                mace_model="medium",
                mace_device="cpu",
                enable_relax=True,
                default_samples=10,
            )
            agent = structure_agent.create_agent(FunctionModel(model_function))
            with patch.object(structure_agent, "run_workflow", side_effect=fake_workflow):
                first = agent.run_sync("生成10个194号空间群的BN结构", deps=deps)
                second = agent.run_sync(
                    "给出生成的第三个结构的形成能和空间群",
                    deps=deps,
                    message_history=first.all_messages(),
                )

        third = structure_agent._get_generated_structure_details(deps, 3)
        self.assertEqual(third.structure_number, 3)
        self.assertEqual(third.sample_index, 4)
        self.assertEqual(third.mace_total_energy_ev, -23.0)
        self.assertEqual(third.actual_spacegroup_number, 194)
        self.assertIn("-23 eV", second.output)

    def test_agent_dispatches_existing_structure_energy_calculation(self):
        captured = []

        def fake_energy(config):
            captured.append(config)
            output_dir = energy_root / "graphene_test"
            return MaceEnergyResult(
                structure_filename="graphene.cif",
                structure_path=str(input_dir / "graphene.cif"),
                formula="C2",
                atom_count=2,
                total_energy_ev=-18.0,
                energy_per_atom_ev=-9.0,
                calculation_type="MACE single-point potential energy (unrelaxed)",
                mace_model=config.mace_model,
                mace_device=config.mace_device,
                output_directory=str(output_dir),
                report_file=str(output_dir / "mace_energy.json"),
            )

        def model_function(messages, _info):
            model_steps = sum(isinstance(message, ModelResponse) for message in messages)
            if model_steps == 0:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "calculate_structure_energy",
                            {"structure_filename": "graphene.cif"},
                        )
                    ]
                )
            return ModelResponse(parts=[TextPart("graphene.cif 的 MACE 单点能量已计算完成。")])

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input_structures"
            input_dir.mkdir()
            energy_root = root / "calculation_results" / "mace_energy"
            deps = structure_agent.AgentDependencies(
                checkpoint_path=Path("epoch699.ckpt").resolve(),
                output_root=root / "generated_structures",
                mace_model="local.model",
                mace_device="cpu",
                enable_relax=True,
                default_samples=10,
                structure_input_dir=input_dir,
                energy_output_root=energy_root,
            )
            agent = structure_agent.create_agent(FunctionModel(model_function))
            with patch.object(structure_agent, "calculate_mace_energy", side_effect=fake_energy):
                result = agent.run_sync("我要计算这个 graphene.cif 的能量", deps=deps)

        self.assertEqual(result.output, "graphene.cif 的 MACE 单点能量已计算完成。")
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].structure_filename, "graphene.cif")
        self.assertEqual(captured[0].input_directory, input_dir)
        self.assertEqual(captured[0].output_root, energy_root)

    def test_agent_dispatches_nabi_generation_and_band_workflow(self):
        generation_configs = []
        band_configs = []

        def fake_generation(config):
            generation_configs.append(config)
            accepted_dir = output_root / "NaBi_sg194" / "run_test" / "accepted"
            return WorkflowResult(
                formula=config.formula,
                spacegroup_number=config.spacegroup_number,
                requested_samples=config.num_samples,
                unique_initial_structures=2,
                accepted_poscars=[str(accepted_dir / "POSCAR_NaBi_sg194_001")],
                output_directory=str(accepted_dir),
                rejected_directory=str(accepted_dir.parent / "others"),
                log_file=str(accepted_dir.parent / "others" / "workflow.log"),
            )

        def fake_band_workflow(config):
            band_configs.append(config)
            bands_dir = config.output_root / "bands"
            return BandWorkflowResult(
                requested_count=1,
                completed_count=1,
                failed_count=0,
                output_root=str(config.output_root),
                bands_directory=str(bands_dir),
                band_images=[str(bands_dir / "band_001_NaBi_sg194_001.png")],
                report_file=str(config.output_root / "band_report.json"),
                log_file=str(config.output_root / "band_workflow.log"),
                failures=[],
            )

        def model_function(messages, _info):
            model_steps = sum(isinstance(message, ModelResponse) for message in messages)
            if model_steps == 0:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "generate_structures_and_calculate_bands",
                            {
                                "formula": "NaBi",
                                "spacegroup_number": 194,
                                "num_samples": 10,
                            },
                        )
                    ]
                )
            return ModelResponse(parts=[TextPart("已完成 NaBi 结构生成和能带计算。")])

        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            deps = structure_agent.AgentDependencies(
                checkpoint_path=Path("epoch699.ckpt").resolve(),
                output_root=output_root,
                mace_model="medium",
                mace_device="cpu",
                enable_relax=True,
                default_samples=10,
            )
            agent = structure_agent.create_agent(FunctionModel(model_function))
            with (
                patch.object(structure_agent, "run_workflow", side_effect=fake_generation),
                patch.object(structure_agent, "check_band_environment", return_value={"ready": True}),
                patch.object(structure_agent, "run_band_workflow", side_effect=fake_band_workflow),
            ):
                result = agent.run_sync(
                    "我要生成10个194号空间群的NaBi结构，然后计算它的能带",
                    deps=deps,
                )

        self.assertEqual(result.output, "已完成 NaBi 结构生成和能带计算。")
        self.assertEqual(len(generation_configs), 1)
        self.assertEqual(generation_configs[0].formula, "NaBi")
        self.assertEqual(generation_configs[0].spacegroup_number, 194)
        self.assertEqual(generation_configs[0].num_samples, 10)
        self.assertEqual(len(band_configs), 1)
        self.assertEqual(len(band_configs[0].structure_paths), 1)
        self.assertEqual(band_configs[0].output_root.name, "band_analysis")

    def test_agent_dispatches_completed_band_result_analysis(self):
        captured = []

        def fake_analysis(deps, result_name):
            captured.append((deps, result_name))
            return structure_agent.BandAccidentalDegeneracyReport(
                result_name="NaBi_sg186_007",
                result_directory="calculation_results/NaBi_sg186_007",
                band_job_directory="calculation_results/NaBi_sg186_007/job_band",
                band_image=None,
                spacegroup_number=186,
                spacegroup_symbol="P6_3mc",
                soc=True,
                detector="irrep exchange",
                energy_window_ev=1.0,
                gap_tolerance_ev=0.03,
                crossing_count=9,
                confirmed_particle_types=["DP"],
                path_compatible_particle_types=["DP", "TP", "WNL", "WNL net"],
                path_summaries=[],
                encyclopedia_path_table=[
                    structure_agent.EncyclopediaPathComparisonItem(
                        high_symmetry_path="KH",
                        line_label="P",
                        crossing_count=1,
                        abbreviation="TP",
                        name="Triple point",
                        classification="path_compatible_ambiguous",
                        source_pdf_page=1058,
                    )
                ],
                crossings=[],
                source_title="Supplemental Material",
                source_file="supplement.pdf",
                source_table="S2",
                source_table_pdf_page=8,
                source_path_section="S8B",
                report_file="accidental_degeneracy_report.json",
                scientific_scope="Path-only classification scope.",
            )

        def model_function(messages, _info):
            model_steps = sum(isinstance(message, ModelResponse) for message in messages)
            if model_steps == 0:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "analyze_calculated_band_accidental_degeneracies",
                            {"result_name": "NaBi"},
                        )
                    ]
                )
            return ModelResponse(parts=[TextPart("NaBi结果中确认DP，并有TP/WNL/WNL net路径候选。")])

        deps = structure_agent.AgentDependencies(
            checkpoint_path=Path("epoch699.ckpt").resolve(),
            output_root=Path("generated_structures").resolve(),
            mace_model="medium",
            mace_device="cpu",
            enable_relax=True,
            default_samples=10,
        )
        agent = structure_agent.create_agent(FunctionModel(model_function))
        with patch.object(
            structure_agent,
            "_run_band_result_analysis",
            side_effect=fake_analysis,
        ):
            result = agent.run_sync("分析NaBi结果中偶然简并都有哪些", deps=deps)

        self.assertEqual(result.output, "NaBi结果中确认DP，并有TP/WNL/WNL net路径候选。")
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0][1], "NaBi")


if __name__ == "__main__":
    unittest.main()
