import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import structure_agent

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from band_workflow import BandWorkflowResult
from workflow_sym import WorkflowResult, atomic_numbers_from_formula


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


if __name__ == "__main__":
    unittest.main()
