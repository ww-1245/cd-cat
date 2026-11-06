"""CD-CAT: 认知诊断计算机自适应测验 (Cognitive Diagnostic Computerized Adaptive Testing)

一个功能完整的 CD-CAT 选题与仿真系统。

主要组件
--------
- ``qmatrix``       Q 矩阵构建、校验与诊断
- ``models``        认知诊断响应模型 (DINA / DINO / G-DINA / RRUM)
- ``calibration``   题目参数标定 (MMLE/EM，支持四种模型)
- ``diagnostics``   拟合指标、被试诊断、模型比较与检验
- ``pipeline``      一体化标定分析流程
- ``estimation``    属性掌握模式估计 (MAP / EAP / MLE)
- ``selection``     选题准则 (KL / PWKL / MPWKL / GDI / SHE / CDI)
- ``termination``   终止规则 (后验阈值 / 最大题量 / 稳定性)
- ``simulation``    蒙特卡洛仿真引擎
- ``evaluation``    性能评估指标 (PMR / AMR / 题量 / 曝光率)

快速开始
--------
>>> from cdcat import QMatrix, CDCATSimulator, PWKL
>>> q = QMatrix.from_random(n_items=60, n_attrs=5, seed=42)
>>> sim = CDCATSimulator(q, model='DINA', selector='PWKL')
>>> result = sim.run(n_examinees=500, n_replications=10)
"""

from .qmatrix import (
    QMatrix,
    IdentifiabilityReport,
    analyse_identifiability,
    QValidationReport,
    QSuggestion,
    validate_q_matrix,
    AttributeHierarchy,
    HierarchyConsistencyReport,
    check_hierarchy_consistency,
)
from .models import ACDM, DINA, DINO, GDINA, LLM, RRUM, create_model
from .estimation import (
    AttributeEstimator,
    NONPARAMETRIC_METHODS,
    NonparametricResult,
    classify_nonparametric,
)
from .selection import (
    SelectionCriterion,
    KL,
    PWKL,
    MPWKL,
    GDI,
    SHE,
    CDI,
    MIV,
    IdealResponseCriterion,
    NPSHE,
    ERHD,
    AMI,
    create_selector,
    ExposureControl,
    SympsonHetter,
    RandomesqueExposure,
    MaxExposureLimit,
    StratifiedExposure,
    NoExposureControl,
    create_exposure_control,
    AssemblyResult,
    assemble_test_bank,
)
from .termination import (
    TerminationRule,
    PosteriorThreshold,
    FixedLength,
    StabilityRule,
    AttributeConfidence,
    PatternConfidence,
    CompositeRule,
    create_rule,
)
from .simulation import CDCATSimulator, SimulationResult
from .evaluation import (
    Evaluator,
    EvaluationReport,
    ItemAnalysisResult,
    analyse_items,
    ExamineeReport,
    build_examinee_report,
    GroupReport,
    build_group_report,
)
from .calibration import (
    EMCalibrator,
    CalibrationResult,
    calibrate,
    NewItemResult,
    OnlineCalibrationReport,
    calibrate_new_items,
)
from .diagnostics import (
    FitIndices,
    compute_fit_indices,
    saturated_log_likelihood,
    PatternDiagnosis,
    diagnose_patterns,
    ModelComparison,
    compare_models,
    likelihood_ratio_test,
    parameter_recovery,
    ItemFitResult,
    item_fit,
    LocalDependenceResult,
    local_dependence,
    AbsoluteFitResult,
    analyse_absolute_fit,
    DIFItem,
    DIFReport,
    dif_analysis,
    PPMCResult,
    posterior_predictive_check,
    HierarchyCandidate,
    HierarchyComparisonReport,
    compare_hierarchies,
)
from .pipeline import CDMAnalysis, run_cdm_analysis
from .report import HTMLReport, build_cdm_report
from .presets import PRESET_BANKS, build_preset_bank, list_presets
from .utils import (
    enumerate_attribute_patterns,
    ideal_response_pattern,
    random_item_params,
    build_test_items,
    pattern_to_str,
    str_to_pattern,
)

# 多级记分（顺序 G-DINA）—— 放在最后，确保 models / qmatrix 已就绪
from .polytomous import (
    PolytomousQMatrix,
    SequentialGDINA,
    expand_to_steps,
    collapse_step_responses,
    calibrate_polytomous,
    build_polytomous_bank,
    PolySelectionCriterion,
    PolyKL,
    PolyPWKL,
    PolyGDI,
    create_poly_selector,
)

__version__ = "1.1.1"
__all__ = [
    "QMatrix",
    "IdentifiabilityReport",
    "analyse_identifiability",
    "QValidationReport",
    "QSuggestion",
    "validate_q_matrix",
    "AttributeHierarchy",
    "HierarchyConsistencyReport",
    "check_hierarchy_consistency",
    "DINA",
    "DINO",
    "GDINA",
    "RRUM",
    "ACDM",
    "LLM",
    "create_model",
    "AttributeEstimator",
    "NONPARAMETRIC_METHODS",
    "NonparametricResult",
    "classify_nonparametric",
    "SelectionCriterion",
    "KL",
    "PWKL",
    "MPWKL",
    "GDI",
    "SHE",
    "CDI",
    "MIV",
    "IdealResponseCriterion",
    "NPSHE",
    "ERHD",
    "AMI",
    "create_selector",
    "ExposureControl",
    "SympsonHetter",
    "RandomesqueExposure",
    "MaxExposureLimit",
    "StratifiedExposure",
    "NoExposureControl",
    "create_exposure_control",
    "TerminationRule",
    "PosteriorThreshold",
    "FixedLength",
    "StabilityRule",
    "AttributeConfidence",
    "PatternConfidence",
    "CompositeRule",
    "create_rule",
    "CDCATSimulator",
    "SimulationResult",
    "Evaluator",
    "EvaluationReport",
    "ItemAnalysisResult",
    "analyse_items",
    "ExamineeReport",
    "build_examinee_report",
    "GroupReport",
    "build_group_report",
    "EMCalibrator",
    "CalibrationResult",
    "calibrate",
    "FitIndices",
    "compute_fit_indices",
    "saturated_log_likelihood",
    "PatternDiagnosis",
    "diagnose_patterns",
    "ModelComparison",
    "compare_models",
    "likelihood_ratio_test",
    "parameter_recovery",
    "ItemFitResult",
    "item_fit",
    "LocalDependenceResult",
    "local_dependence",
    "AbsoluteFitResult",
    "analyse_absolute_fit",
    "CDMAnalysis",
    "run_cdm_analysis",
    "PRESET_BANKS",
    "build_preset_bank",
    "list_presets",
    "enumerate_attribute_patterns",
    "ideal_response_pattern",
    "random_item_params",
    "build_test_items",
    "pattern_to_str",
    "str_to_pattern",
    "PolytomousQMatrix",
    "SequentialGDINA",
    "expand_to_steps",
    "collapse_step_responses",
    "calibrate_polytomous",
    "build_polytomous_bank",
    "PolySelectionCriterion",
    "PolyKL",
    "PolyPWKL",
    "PolyGDI",
    "create_poly_selector",
]
