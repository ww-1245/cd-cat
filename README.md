# CD-CAT 认知诊断计算机自适应测验系统

一个功能完整、技术扎实的 **CD-CAT（Cognitive Diagnostic Computerized Adaptive Testing）**
选题与仿真系统，纯 Python 实现，自带交互式 Web 界面。

> 传统 CAT 估计连续能力值 θ；**CD-CAT 估计离散的属性掌握模式 α**
> ——它以尽可能少的题目，精准识别每个人的知识结构。

---

## 目录

- [核心特性](#核心特性)
- [快速开始](#快速开始)
- [命令行工具](#命令行工具cliedcat)
- [系统架构](#系统架构)
- [数学原理](#数学原理)
- [非参数分类（距离判别）](#非参数分类距离判别)
- [结构可辨识性：为什么 PMR 达不到 1.0](#结构可辨识性为什么-pmr-达不到-10)
- [属性层级关系 AHM](#属性层级关系-ahm)
- [Q 矩阵自动修正（Qval / PVAF）](#q-矩阵自动修正qval--pvaf)
- [多级评分 CDM（顺序 G-DINA）](#多级评分-cdm顺序-g-dina)
- [绝对拟合与局部依赖](#绝对拟合与局部依赖)
- [后验预测检验 PPMC](#后验预测检验ppmc)
- [属性层级结构比较](#属性层级结构比较)
- [题库装配](#题库装配从大题库到短测验)
- [新题在线标定](#新题在线标定)
- [单文件 HTML 报告](#单文件-html-报告)
- [一页式质量看板与明暗主题](#一页式质量看板与明暗主题)
- [题目分析与冗余检测](#题目分析与冗余检测)
- [个人诊断报告](#个人诊断报告)
- [群体 / 班级诊断报告](#群体--班级诊断报告)
- [曝光控制](#曝光控制)
- [使用方法](#使用方法)
  - [多模型 CDM 分析流水线](#多模型-cdm-分析流水线推荐入口)
- [实测性能](#实测性能)
- [API 速查](#api-速查)
- [参考文献](#参考文献)

---

## 核心特性

| 模块 | 已实现 |
|---|---|
| **响应模型** | DINA（合取）、DINO（析取）、G-DINA（饱和）、RRUM（乘法）、**ACDM（加性）**、**LLM（线性 logistic）**，**+ 顺序 G-DINA（多级评分）** |
| **参数标定** | DINA / DINO / G-DINA / RRUM / ACDM / LLM 六模型的 MMLE-EM 估计，**经验贝叶斯属性先验 π(α) 估计**，含 AIC/BIC/AICc；**多级评分专用条件 EM** |
| **模型拟合指标** | −2LL、deviance、n_params、AIC、BIC、AICc、饱和模型 LL、CFI、TLI、RMSEA、LRT |
| **属性层级 AHM** | 有向先备关系、Warshall 可达闭包、合法模式枚举、层级一致性检查、**上限提升量化**；已接入辨识性分析 / 估计器 / 仿真器 |
| **Q 矩阵自动修正** | **Qval / PVAF** 逐题搜索最优属性子集，自动给出「漏标属性」与「多余属性」建议并可直接应用 |
| **绝对拟合诊断** | 题目级 Pearson χ² / 似然比 G² 检验、残差效应量、Yen **Q3** 局部依赖矩阵与超标题目对 |
| **缺失值处理** | 作答矩阵缺失读取为 `NaN`；**标定按题边际化**（MAR，不当作答错），诊断 / 绝对拟合 / Q3 逐题掩码 + 配对删除，不会被静默污染 |
| **模型比较** | 多模型标定、AIC/BIC/AICc 择优、嵌套模型似然比检验（自实现 χ² 上尾概率，无 scipy 依赖）；**非嵌套组合自动提示 ΔBIC 而非 p 值** |
| **被试诊断** | MAP / EAP 双路知识状态、完整后验矩阵、属性边际概率、属性级信度、模式频次分布、MAP 置信度 |
| **非参数诊断** | **距离判别（汉明 / 广义距离 GDD）**：不估计任何题目参数即可分类，可作为参数模型的零参数基线；支持层级约束与缺失作答 |
| **群体报告** | **班级/群体掌握率排行、薄弱知识点 TOP-N、典型模式频次、分组对比、个体 vs 群体百分位、四联报告图** |
| **能力估计** | MAP（最大后验）、EAP（期望后验）、MLE（最大似然） |
| **选题准则** | 参数化六种：PWKL、MPWKL、KL、GDI、SHE、CDI；**非参数四种（只读 Q 矩阵，无需标定）：MIV、NPSHE、ERHD、AMI**；多级评分对应 PolyKL / PolyPWKL / PolyGDI |
| **终止规则** | 模式级置信、后验阈值、固定题量、估计稳定性、属性熵、可任意组合 |
| **曝光控制** | 硬上限、随机化选题、分层曝光、Sympson-Hetter（含自动标定） |
| **仿真引擎** | 单被试施测、固定题量仿真、蒙特卡洛批量仿真（**`n_jobs` 多进程并行**）、多准则公平对比；多级评分模型可直接接入 |
| **性能评估** | PMR / AMR / 平均题量 / 题库利用率 / 曝光率 / χ² 均匀性检验 / 置信区间 |
| **结构可辨识性** | 检出不可区分模式、可互换属性、PMR 理论上限，**支持在层级约束下重算上限** |
| **题目分析** | 区分度、信息量、结构冗余检测、最弱/最强题识别 |
| **数据 I/O** | 纯 Python 读写 Q 矩阵 / 作答矩阵 / 题库 / 参数（CSV/TSV/JSON），自动嗅探分隔符与表头 |
| **结果报告** | 个人诊断报告（TXT/JSON）、群体报告（TXT/JSON/PNG）、评估报告、被试明细、对比总表、多 sheet Excel 导出 |
| **Q 矩阵诊断** | 属性覆盖、重复题检测、关联图连通性、属性共现相关 |
| **命令行** | **`cdcat` 十五个子命令**：`presets` / `list` / `demo` / `calibrate` / `simulate` / `diagnose` / `ppmc` / `hier-test` / `assemble` / `online-calib` / `dif` / `nonparam` / `validate-q` / `report` / `report-html` |
| **可视化** | **18 张交互式 Plotly 图**（界面）+ **13 张 matplotlib 静态图**（报告），含 6 面板标定报告、3 面板绝对拟合报告、群体四联图、收敛三连图 |
| **交互界面** | Streamlit **九大功能页**，含「质量看板」（一屏 KPI 仪表 + 知识地图 + 逐题体检）与「多级评分」页；**浅色/深色双主题**，图表与页面配色一起切换 |
| **测试** | **486 个测试**（含 20 个界面端到端冒烟测试、16 个 CLI 端到端测试），全部通过 |

---

## 快速开始

### 1. 安装依赖

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. 启动交互式界面

```bash
streamlit run demo/app.py
```

浏览器会自动打开，包含九个功能页：

| 页面 | 作用 |
|---|---|
| **质量看板** | 一次「体检」把自适应诊断精度、知识地图、逐题拟合 / 局部依赖 / 区分度压到一屏，并导出自包含 HTML 报告 |
| **在线施测** | 模拟一名考生逐题作答，实时看属性后验如何收敛，导出个人诊断报告 |
| **蒙特卡洛仿真** | 批量仿真，评估 PMR/AMR、题量、曝光控制，并与结构上限对比 |
| **准则对比** | 同一批被试下横向比较各选题准则，选出最优方案 |
| **题库诊断** | Q 矩阵校验、属性覆盖热图、关联图连通性、结构可辨识性、题目信息量与冗余 |
| **群体报告** | 班级/群体掌握率排行、薄弱知识点、确定性分档、分组对比、个体 vs 群体常模；内置 **Q 矩阵自动修正** 与 **属性层级** 选择器 |
| **参数标定** | EM 算法从作答数据估计题目参数与属性先验，输出绝对拟合 / Q3 诊断并验证恢复精度 |
| **多级评分** | **顺序 G-DINA**：多级题库构造 → 条件 EM 标定 → 参数恢复 → 档位概率曲线 → 多级自适应施测 |
| **系统说明** | 方法、使用流程与参考文献 |

> **深色界面**：在启动命令后加 `--theme.base dark`（或把 `.streamlit/config.toml`
> 里的 `base` 改成 `dark`）。页面配色来自一套设计令牌，图表另有 `cdc` / `cdc_dark`
> 两套 Plotly 模板，应用会读 `st.get_option("theme.base")` 自动选配套的那一套 ——
> 不会出现"深底浅图"的割裂感。项目自带的 `config.toml` 刻意**只写 `primaryColor`
> 与 `font`、不写 `base`**：一旦写死 `base`，命令行参数就会被配置文件覆盖。

### 3. 或用命令行完整演示

```bash
python demo/run_demo.py
```

会走完「建库 → 标定 → 施测 → 准则对比 → 仿真 → 结构分析 → 出图 → 多模型 CDM 分析 → 绝对拟合诊断 → 落地工具链」共 10 步，
并在 `demo/output/` 生成 10 张 PNG 图表、1 份多 sheet Excel 报告、1 份自包含 HTML 报告。

### 4. 或直接用命令行工具

```bash
pip install -e .        # 安装后获得 cdcat 命令
cdcat --help
cdcat demo              # 一条命令跑完整链路并落盘产物
```

详见 [命令行工具](#命令行工具cliedcat)。

### 5. 跑测试

```bash
pytest tests/ -v
```

共 **486 个测试**：466 个核心库单元/集成测试 + 20 个 Streamlit 界面测试
（后者用 `streamlit.testing.v1.AppTest` 在无浏览器环境下驱动真实页面，
端到端跑通「生成数据 → 多模型标定 → 指标渲染」，可捕获页面级运行时异常；
新模块另配 16 个 CLI 端到端测试，逐条验证落盘产物）。

---

## 命令行工具（CLI）：`cdcat`

不想写 Python、不想开浏览器时，`cdcat` 把整套能力做成了命令行 —— **15 个子命令**：

```bash
cdcat presets                                   # 列出 4 个内置题库
cdcat list                                      # 列出可用模型 / 准则 / 终止规则 / 曝光策略

# ① 从作答数据标定参数（含经验贝叶斯先验、拟合指标、题目参数落盘）
cdcat calibrate -q q.csv -r resp.csv -m DINA -o out/calib --json --figures

# ② 蒙特卡洛仿真（单准则或多准则对比，支持并行）
cdcat simulate --preset "分数运算 (4 属性)" -s PWKL \
               --compare KL PWKL GDI -n 300 -R 5 -j 8 -o out/sim --figures

# ③ 题目级诊断（绝对拟合 + Q3 局部依赖 + 题目分析）
cdcat diagnose -q q.csv -r resp.csv -m DINA -o out/diag --json --figures

# ④ 检查 Q 矩阵是否需要修正（Qval / PVAF）
cdcat validate-q -q q.csv -r resp.csv --pvaf 0.95 --top 10 -o out/qval

# ⑤ 群体 / 班级诊断报告（TXT + JSON + 四联图，可选分组对比）
cdcat report -q q.csv -r resp.csv --groups grade.csv \
             --missing skip --label "高一(3)班" --top 3 -o out/report --figures

# ⑥ 后验预测检验：模型能否再现数据的二阶结构（SRMSR / Q3 / 联合答对率）
cdcat ppmc -q q.csv -r resp.csv -m DINA --replicates 500 -o out/ppmc --json

# ⑦ 属性层级结构比较：none / linear / convergent / divergent 谁最省参数又不失拟合
cdcat hier-test -q q.csv -r resp.csv -o out/hier --json

# ⑧ 题库装配：从大题库里挑一份满足覆盖与题量约束的短测验
cdcat assemble -q q.csv --params out/calib/item_params.json -k 16 \
              --min-per-attr 2 --max-per-attr 6 --must 3 17 -o out/asm

# ⑨ 新题在线标定：新题没参数，借已有模型的后验反推 slip / guess
cdcat online-calib -q q.csv -r resp.csv --new-items 41 42 43 44 45 -o out/online

# ⑩ DIF 检验：分班后逐题检查是否存在组间功能差异
cdcat dif -q q.csv -r resp.csv --groups class.csv --alpha 0.05 -o out/dif

# ⑪ 单文件 HTML 报告：图表内联，可离线归档 / 邮件发送
cdcat report-html -q q.csv -r resp.csv --models DINA G-DINA \
                  --title "高一(3)班 · 认知诊断报告" -o out/html

# ⑫ 非参数分类：不标定参数，用距离判别直接出知识状态
cdcat nonparam -q q.csv -r resp.csv --methods gdd hamming \
               --truth true_patterns.csv -o out/np --json

# ⑬ 一条命令的完整演示
cdcat demo --preset "分数运算 (4 属性)" -n 300 -R 3 -o out/demo --figures
```

| 参数 | 说明 |
|---|---|
| `--preset` / `--bank` | 用内置题库，或读入 JSON 题库文件（含题目参数） |
| `-q` / `-r` | Q 矩阵与作答矩阵（CSV/TSV，自动嗅探分隔符与表头） |
| `-m` | 响应模型：`DINA` / `DINO` / `G-DINA` / `RRUM` / `ACDM` / `LLM`（`calibrate` 每次跑一个，横向比较请多次运行） |
| `-s` / `--compare` | 单选准则 / 多准则横向对比（`simulate`） |
| `-n` / `-R` / `-j` | 被试数 / 重复次数 / 并行进程数 |
| `--estimate-prior` | 估计经验贝叶斯属性先验 π(α)（`calibrate`/`diagnose` 默认开启） |
| `--pvaf` / `--top` | Q 矩阵修正的 PVAF 阈值 / 最多展示条数 |
| `--groups` | 分组表（`被试ID,组名` 或单列组名），自动对齐作答行序 |
| `--methods` | 非参数判别方法：`gdd`（方差加权）/ `hamming`（等权），可多选 |
| `--hierarchy` | 层级约束：`none` / `linear` / `convergent` / `divergent`（非参数分类） |
| `--truth` | 真实属性模式 CSV（行=被试，列=属性），给出则计算 PMR / AMR |
| `--missing {skip,wrong,ignore}` | 缺失作答：跳过该题 / 按答错 / 报错 |
| `--json` / `--figures` | 额外输出机器可读 JSON / matplotlib 报告图 |
| `--replicates` / `--no-refit` | PPMC 复制次数 / 复制侧是否重估题目参数（默认重估，两侧口径才对称） |
| `--new-items` / `--old-items` | 在线标定中的新题题号（1 起）/ 锚题题号（默认用其余全部题目） |
| `-k` / `--must` | 装配目标题量 / 必须入选的题号（1 起） |
| `--min-per-attr` / `--max-per-attr` | 每个属性至少 / 至多被几道题考查（不可行时直接报错，不静默放宽） |
| `--structures` | 层级比较的候选结构（默认四种全跑） |
| `-o` | 产物目录，自动创建 |

每个子命令都会打印结构化摘要并把全部结果**落盘**（CSV / JSON / PNG），
可直接接进批处理脚本或 CI。`cdcat/io.py` 与 `cdcat/presets.py` 是纯 Python 实现，
**不依赖 streamlit**，因此 CLI 可在无界面服务器上独立运行。

---

## 系统架构

```
cdcat/
├── io.py           纯 Python 数据 I/O：Q 矩阵 / 作答矩阵 / 题库 / 参数（CSV·TSV·JSON）
├── presets.py      内置题库定义（Web 界面与 CLI 共用同一份，不再漂移）
├── cli.py          `cdcat` 命令行：15 个子命令（标定 / 仿真 / 诊断 / 装配 / 报告……）
├── figures.py      matplotlib 报告级图：属性掌握率 / 群体四联图 / 收敛三连图
├── utils/          属性模式枚举、理想反应、随机题库生成
├── qmatrix/        Q 矩阵：校验、诊断、理想反应矩阵 E、结构可辨识性
│   ├── hierarchy.py     属性层级 AHM：先备关系、Warshall 可达闭包、合法模式
│   └── refinement.py    Q 矩阵自动修正：Qval / PVAF 逐题最优属性子集搜索
├── models/         响应模型：DINA / DINO / G-DINA / RRUM / ACDM / LLM
├── polytomous/     多级评分：顺序 G-DINA、专用条件 EM、PolyKL/PolyPWKL/PolyGDI
├── calibration/    MMLE/EM 参数标定（六模型）+ 经验贝叶斯属性先验 π(α) + 拟合指标
│   └── online.py        新题在线标定（后验加权 MMLE + Beta 解析 CI）
├── diagnostics/    模型拟合指标、被试诊断、模型比较、LRT、参数恢复
│   ├── absolute_fit.py  题目级 χ²/G² 绝对拟合 + Yen Q3 局部依赖（缺失值配对删除）
│   ├── ppmc.py          后验预测检验（SRMSR / Q3max / 联合答对率 / 总分 SD）
│   └── hierarchy_test.py 属性层级结构比较（BIC 择优，先验零点锁死）
├── pipeline.py     多模型 CDM 分析流水线（端到端一站式入口）
├── report.py       自包含 HTML 报告（CSS 内联 + base64 图，八个小节）
├── calibration_plots.py  matplotlib 静态报告图（10 图 + 6 面板标定报告 + 3 面板绝对拟合报告）
├── estimation/     属性后验估计：MAP / EAP / MLE（支持层级约束，非法模式先验清零）
│   └── nonparametric.py 非参数距离判别：汉明 / GDD + 层级约束 + 伪后验（0 参数）
├── selection/      选题准则：KL / PWKL / MPWKL / GDI / SHE / CDI + 非参数 MIV / NPSHE / ERHD / AMI
│   ├── exposure.py 曝光控制：硬上限 / 随机化 / 分层 / Sympson-Hetter
│   └── assembly.py 题库装配：maximin 对称 KL + 覆盖约束 + 成对交换局部搜索
├── termination/    终止规则与组合器
├── simulation/     蒙特卡洛仿真引擎（支持 joblib 多进程并行、层级约束）
└── evaluation/     性能评估、题目分析、个人报告
    └── group.py    群体 / 班级诊断报告（属性掌握率、薄弱知识点、分组对比）

demo/
├── app.py          Streamlit 交互界面（九大功能页，双套设计令牌 + 明暗主题）
├── cdm_charts.py   交互式 Plotly 图表（20 张，含知识地图 DAG 与半环仪表）
├── bank_utils.py   题库构建与导入导出（转发库内定义）
└── run_demo.py     CLI 完整演示（十步 + 十图 + Excel + HTML 导出）

tests/                       486 项（466 核心 + 20 界面）
├── test_cdcat.py            核心库单元与集成测试
├── test_cdm_analysis.py     CDM 标定 / 诊断 / 比较 / 可视化
├── test_cdm_diagnostics.py  先验估计 / 并行仿真 / 绝对拟合 / Q3
├── test_new_models.py       ACDM / LLM 的结构性质与工厂投影
├── test_ideal_selection.py  非参数准则 NPSHE / ERHD / AMI 的数值正确性
├── test_missing_data.py     缺失作答的边际化与参数恢复
├── test_new_features.py     I/O / 群体报告 / Q 修正 / 层级 / 多级评分 / CLI
├── test_assembly.py 等      题库装配 / 在线标定 / 报告 / 非参数诊断
└── test_app_smoke.py        界面端到端冒烟测试（AppTest，20 项）
```

数据流：

```
Q 矩阵 + 题目参数
      ↓
 [结构可辨识性分析] → PMR 理论上限
      ↓
 [选题准则] ← 当前后验分布
      ↓         ↖ [曝光控制]（可选）
   选定题目
      ↓
 [响应模型] 生成作答（仿真）/ 考生作答（真实施测）
      ↓
 [估计器] 更新属性后验
      ↓
 [终止规则] 判断是否停止 → 未停止则回到选题
      ↓
 [评估/报告] PMR vs 上限 · 曝光均匀性 · 个人诊断报告
```

标定‑诊断链路（作答矩阵 → 参数与分类结果）：

```
作答矩阵 X (n × J)
      ↓
 [EMCalibrator] 对每个候选模型跑 MMLE-EM
      ↓  E 步：后验责任 h_i(α)
      ↓  M 步：解析更新题目参数 + 经验贝叶斯更新属性先验 π(α)
 参数 (slip/guess · β,γ · π*,r*) + π(α)
      ↓
 [compute_fit_indices] −2LL / AIC / BIC / AICc / CFI / TLI / RMSEA（含先验参数计数）
      ↓
 [compare_models] 择优（默认 BIC）  →  [likelihood_ratio_test] 嵌套检验
      ↓
 [diagnose_patterns] MAP + EAP 知识状态 · 后验矩阵 · 属性信度
      ↓
 [analyse_absolute_fit] 题目级 χ²/G² 绝对拟合 · 残差效应量 · Yen Q3 局部依赖
      ↓
 [可视化] 6 面板标定报告图 + 3 面板绝对拟合报告图 / 8 张交互式 Plotly 图
      ↓
 [导出] 多 sheet Excel（拟合指标 / 题目参数 / 被试诊断 / 绝对拟合 / Q3 / 属性先验）
```

---

## 数学原理

### Q 矩阵与理想反应

Q 矩阵 $Q$ 为 $J \times K$ 的 0/1 矩阵，$q_{jk}=1$ 表示第 $j$ 题考查第 $k$ 个属性。
对每个属性模式 $\alpha$（共 $2^K$ 个），定义**理想反应**：

- **连接型（DINA）**：$\eta_j(\alpha) = \prod_{k \in K_j^*} \alpha_k$ —— 需掌握全部所需属性
- **析取型（DINO）**：$\omega_j(\alpha) = 1 - \prod_{k \in K_j^*} (1-\alpha_k)$ —— 掌握任一即可

### 响应模型

**DINA / DINO**

$$
P(X_j=1 \mid \alpha) = (1-s_j)^{\eta_j(\alpha)} \, g_j^{1-\eta_j(\alpha)}
$$

$s_j$ 为失误概率（slip），$g_j$ 为猜测概率（guess）。

**G-DINA（本题采用 logit 指示函数基）**

$$
\operatorname{logit}\big[P(X_j=1\mid\alpha)\big]
= \lambda_{j,0} + \sum_{\varnothing \neq S \subseteq K_j^*} \lambda_{j,S} \cdot \mathbb{1}\!\left[\alpha_k=1,\ \forall k\in S\right]
$$

由于指示基可精确表达任意阶梯函数，`GDINA.from_dina()` 通过解线性方程组
$\lambda = X^{-1}\operatorname{logit}(P_{\text{group}})$ 得到与 DINA
**数值上完全一致**的概率矩阵（实测误差 $< 10^{-16}$）。

**ACDM（加性模型，de la Torre 2011）**

$$
P(X_j=1\mid\alpha) = \delta_{j0} + \sum_{k \in K_j^*} \delta_{jk}\,\alpha_k
$$

每个掌握的属性各自贡献一份增量，因此是**完全补偿**的：掌握 A 或掌握 B
都能把概率推高。约束 $\delta_{jk}\ge 0$（单调）与
$\delta_{j0}+\sum_k\delta_{jk}\le 1$（值域）。因为它在概率尺度上是线性的，
M 步对分组答对率做**加权最小二乘**即可闭式求解，不需要迭代。

**LLM（线性 logistic 模型，Maris 1999）**

$$
\operatorname{logit}\big[P(X_j=1\mid\alpha)\big]
= \beta_{j0} + \sum_{k \in K_j^*} \beta_{jk}\,\alpha_k
$$

这是 **LCDM 只保留主效应**时的特化形式（Henson, Templin & Willse 2009）：
同样是加性、补偿的，但 logit 链接自带值域保证，不会像 ACDM 那样越界；
代价是参数不再有"概率增量"的直观解释。M 步用分组二项 **IRLS**（加权
logistic 回归），斜率强制非负。

> ACDM 与 LLM 每题都是 $1+r_j$ 个参数，正好卡在 DINA（2 个）与
> G-DINA（$2^{r_j}$ 个）之间。它们与 DINA 的差别不在参数多少，而在
> **能不能表达"全掌握才跳变"**：二者都是加性的，表达不了合取台阶。
> 实测在合取型数据上，ACDM 的 BIC 比 DINA 高 14521（见「实测性能」），
> 这不是标定失败——EM 的解比"把 DINA 概率直接投影到 ACDM"的初值 logL 还高
> 490，说明优化到位，差的是结构本身表达不了合取台阶。

### 选题准则（核心）

**PWKL**（后验加权 KL，当前主流）：

$$
\mathrm{PWKL}_j = \sum_{\alpha} \pi(\alpha)
\sum_{x \in \{0,1\}} P(X_j=x \mid \alpha) \,
\log \frac{P(X_j=x \mid \alpha)}{P(X_j=x \mid \hat\alpha)}
$$

其中 $\pi(\alpha)$ 为当前后验，$\hat\alpha$ 为当前估计模式。

**MPWKL**：对后验施加 $M$ 次扰动再取平均，避免 PWKL 过早锁定单一模式：

$$
\mathrm{MPWKL}_j = \frac{1}{M}\sum_{m=1}^{M}\sum_{\alpha}\pi_m(\alpha)\,
\mathrm{KL}\big(P(X_j|\alpha) \,\|\, P(X_j|\hat\alpha_m)\big)
$$

> MPWKL 是六个准则里唯一带随机性的（扰动哪几个属性是抽出来的），所以它的
> 默认种子**必须是固定值**。早期版本写成 `seed=None`，`default_rng(None)` 会从
> 操作系统熵源取种，结果同一份数据两次运行 PMR 能差半个百分点，
> 而 `compare_criteria(seed=0)` 的文档却承诺"每个准则面对完全相同的随机流"——
> 用户完全无法复现。改成 `seed=0` 后逐位一致，需要随机化时显式传别的种子。

**GDI**（与当前估计无关，适合作为前几题的准则）：

$$
\mathrm{GDI}_j = 2\left[\,\mathbb{E}_{\pi}[P_j^2] - \big(\mathbb{E}_{\pi}[P_j]\big)^2\right]
$$

其余准则（KL / SHE / CDI）分别从后验 KL 散度、Shannon 熵缩减、
属性边际方差缩减三个角度度量题目信息量。

#### 非参数准则：不读题目参数

下面四个准则只用到 Q 矩阵的理想反应 $\eta_j(\alpha)$，**完全不碰
slip / guess**。做法是把"该答对 / 该答错"当成确定性观察，把后验劈成两支：

$$
w_1 = \pi(\eta_j=1),\quad \pi_1(\alpha) = \frac{\pi(\alpha)\,\mathbb{1}[\eta_j(\alpha)=1]}{w_1},
\qquad
w_0 = 1-w_1,\quad \pi_0(\alpha) = \frac{\pi(\alpha)\,\mathbb{1}[\eta_j(\alpha)=0]}{w_0}
$$

然后各自用不同的信息量衡量"劈开之后不确定性降了多少"：

$$
\begin{aligned}
\text{NPSHE}_j &= H(\pi) - \bigl[w_1 H(\pi_1) + w_0 H(\pi_0)\bigr] \\[2pt]
\text{ERHD}_j &= \mathrm{EHD}(\pi) - \bigl[w_1 \mathrm{EHD}(\pi_1) + w_0 \mathrm{EHD}(\pi_0)\bigr],
\quad \mathrm{EHD}(\pi)=\textstyle\sum_\alpha \pi(\alpha)\,d_H(\alpha,\hat\alpha) \\[2pt]
\text{AMI}_j &= \textstyle\sum_k H(\alpha_k) - \bigl[w_1 \sum_k H(\alpha_{k}\mid X_j{=}1) + w_0 \sum_k H(\alpha_k\mid X_j{=}0)\bigr] \\[2pt]
\text{MIV}_j &= p_j(1-p_j), \qquad p_j = \mathbb{E}_{\pi}\bigl[\eta_j(\alpha)\bigr]
\end{aligned}
$$

四者的分工：

| 准则 | 度量什么 | 什么时候用 |
|---|---|---|
| **NPSHE** | 模式级熵缩减 | SHE 的无参数替代，最接近"信息量"的直觉 |
| **ERHD** | 期望汉明距离缩减 | 直接优化属性误判个数，与 AMR 同口径 |
| **AMI** | 属性边际熵缩减 | Q 矩阵有等价类时仍能区分，不会追不可辨识的方向 |
| **MIV** | 理想反应方差 | 最省算力，只需一次矩阵乘法 |

因为它们与题目参数无关，同一 Q 矩阵下换任何参数，准则值逐位不变
（四个准则都有测试钉住）。代价是丢掉了"这题有多大把握答对"这一层：
理想反应把所有题都当成 $\text{slip}=\text{guess}=0$，因而区分不出
"大概率答对"和"一定答对"。

### 参数标定（MMLE/EM）

E 步计算每名考生属于各属性模式的后验责任：

$$
h_{i}(\alpha) = \frac{\pi(\alpha) L(\mathbf{x}_i \mid \alpha)}{\sum_{\alpha'} \pi(\alpha') L(\mathbf{x}_i \mid \alpha')}
$$

M 步按模型解析更新参数。**DINA / DINO** 由掌握组/未掌握组的期望计数直接得到 slip / guess：

$$
s_j = \frac{\sum_i h_i(\alpha{:}\,\eta_j=1)(1-x_{ij})}{\sum_i h_i(\alpha{:}\,\eta_j=1)},
\qquad
g_j = \frac{\sum_i h_i(\alpha{:}\,\eta_j=0)\,x_{ij}}{\sum_i h_i(\alpha{:}\,\eta_j=0)}
$$

**G-DINA** 对每题、每个潜在组先算期望答对率 $P(X_j=1\mid\alpha_c)$，再取 logit，按指示函数设计矩阵 $X$ 解线性方程 $X\theta=\mathrm{logit}(P)$ 得到 $\beta_j$ 与 $\gamma_j$（每题 $2^{r_j}$ 个参数）。**RRUM** 由全掌握组得 $\pi_j^*$，再由"仅缺属性 $k$"组的期望计数比得折损系数 $r_{jk}^*$。**ACDM** 与 **LLM** 共用同一套分组统计量（各组后验权重 $w_g$ 与加权答对数 $c_g$），ACDM 解加权最小二乘、LLM 跑分组二项 IRLS，都在 1 秒内完成 60 题 / 1500 人的标定。

六模型的自由参数计数：

$$
\text{DINA/DINO}: 2J, \qquad
\text{G-DINA}: \sum_j 2^{r_j}, \qquad
\text{RRUM/ACDM/LLM}: \sum_j (1+r_j)
$$

#### 缺失作答：按题边际化

缺失用 `np.nan` 标记。E 步只对已作答题连乘 $P(x_{ij}\mid\alpha)^{obs_{ij}}$，
各 M 步的分子分母同时剔除该题——等价于 MAR 假设下的边际化，而不是"答错"。

把缺失当 0 会让已掌握属性者被记一次失误，偏差方向单一且量级很大：
同一份模拟数据（真值 slip = 0.10），20% 随机缺失下旧实现估出 0.29、
40% 时顶到 0.499 上界；边际化后分别是 0.10 / 0.11，
slip 的 MAE 从 0.186 / 0.358 降到 0.013 / 0.017（`tests/test_missing_data.py` 锁定）。
整列缺失时该题参数不可识别，只发 `RuntimeWarning` 并保持初值。

#### 经验贝叶斯属性先验 π(α)

标准 MMLE 不只估计题目参数——M 步还可一并估计**属性模式先验** π(α)。
若把 π(α) 写死为均匀分布，当被试总体的属性分布本身不均匀时，
后验会被系统性地向均匀方向拉偏，属性掌握率与模式判准率双双受损。

本系统在 M 步按**经验贝叶斯**方式更新（Bock & Aitkin, 1981）：

$$
\pi^{(t+1)}(\alpha) = \frac{\sum_i h_i^{(t)}(\alpha)}{\sum_{\alpha'} \sum_i h_i^{(t)}(\alpha')}
$$

即"当前期望属于模式 α 的被试比例"。为防止某模式的责任和恰为 0 后被**永久锁死**，
引入 Dirichlet 平滑（默认 $\varepsilon = 10^{-6}$）：

$$
\pi(\alpha) \leftarrow \frac{c_\alpha + \varepsilon}{\sum_{\alpha'} c_{\alpha'} + S\varepsilon},
\qquad c_\alpha = \sum_i h_i(\alpha),\quad S = 2^K
$$

估计 π(α) 引入 $2^K-1$ 个自由参数，**必须计入 AIC/BIC**，否则"固定先验 vs 估计先验"
两种设定不可比。因此系统把参数拆成两部分：

$$
k = k_{\text{item}} + k_{\text{prior}},
\qquad k_{\text{prior}} = \begin{cases} 2^K-1 & \text{估计先验} \\ 0 & \text{固定先验}\end{cases}
$$

```python
EMCalibrator(q, "DINA")                        # 默认估计 π(α)（prior 未给定）
EMCalibrator(q, "DINA", estimate_prior=False)  # 退回均匀先验
EMCalibrator(q, "DINA", prior=my_prior)        # 固定为自定义先验
res.n_item_params, res.n_prior_params, res.n_params, res.attr_prior
```

### 三项扩展能力的数学基础

**① 属性层级（AHM）**：有向边 $a \to b$ 表示 $a$ 是 $b$ 的**先备**。合法模式集合为

$$
\mathcal{A}_{\text{legal}} = \Big\{\alpha \in \{0,1\}^K :\; \forall (a,b)\in E,\; \alpha_b = 1 \Rightarrow \alpha_a = 1\Big\}
$$

只需检查**直接边**即可 —— 传递性由可达矩阵保证。可达矩阵用 Warshall 布尔闭包：

$$
R^{(t+1)}_{ij} = R^{(t)}_{ij} \;\lor\; \big(R^{(t)}_{ik} \land R^{(t)}_{kj}\big)
$$

线型 K=4 层级把 $2^4=16$ 个模式压缩到 5 个：
$0000 \to 1000 \to 1100 \to 1110 \to 1111$。

**② Q 矩阵修正（Qval / PVAF）**：对第 $j$ 题、候选属性子集 $s$，理想反应为
$\eta_{is} = \prod_{k \in s}\alpha_{ik}$。把被试按 $\eta_{is}$ 分成两组的方差解释比例

$$
\mathrm{PVAF}(s) = 1 - \frac{\sum_{\eta \in \{0,1\}} n_\eta \big(\hat{P}_\eta - \eta\big)^2}{\sum_i (X_{ij} - \bar{X}_j)^2},
\qquad \hat{P}_\eta = \frac{\sum_{i:\,\eta_{is}=\eta} X_{ij}}{n_\eta}
$$

在全部 $2^{K}-1$ 个非空子集中：

$$
s^* = \begin{cases}
\arg\min\limits_{s:\,\mathrm{PVAF}(s) \ge \tau} |s| & \text{有子集达标（奥卡姆：取属性最少者）} \\[4pt]
\arg\max\limits_{s} \mathrm{PVAF}(s) & \text{否则}
\end{cases}
$$

**③ 多级评分（顺序 G-DINA）**：把第 $j$ 题拆成 $M_j$ 个**顺序步骤**。
第 $m$ 步的 slip 为 $s_{jm}(\alpha)$，则

$$
P(X_j = k \mid \alpha) = \Big(\prod_{l \le k} s_{jl}(\alpha)\Big)\big(1 - s_{j,k+1}(\alpha)\big),
\qquad s_{j,M_j+1} \equiv 0
$$

闭环式 M 步只用"已经走到第 $m$ 步"（$X_{ij} \ge m-1$）的被试：

$$
\hat{p}_{m,g} = \frac{\sum_i \sum_{\alpha \in g} h_i(\alpha)\,\mathbb{1}[X_{ij} \ge m]}
{\sum_i \sum_{\alpha \in g} h_i(\alpha)\,\mathbb{1}[X_{ij} \ge m-1]}
\;\Longrightarrow\;
\begin{cases}
s_{jm}^{(1)} = 1 - \hat{p}_{m,1} & \text{（未掌握组）}\\
g_{jm}^{(0)} = \hat{p}_{m,0} & \text{（掌握组）}
\end{cases}
$$

> ⚠️ **建模陷阱**：若把多级题朴素展开成二分题 $Y_{jm}=\mathbb{1}[X_j \ge m]$ 再喂给普通 DINA EM，
> 等价于假设 $P(Y_{jm}=1\mid\alpha)$ 只依赖 $q_{jm}$ —— 但真实值是 $\prod_{l\le m} s_{jl}(\alpha)$，
> **含前面所有步骤的乘积**，不服从该假设（只有 $m=1$ 是特例）。
> 实测这样做参数恢复相关系数只有 0.16 且**不随样本量改善**；改用上式条件 EM 后升至 0.98。

### 模型拟合指标

设数据在模型下的边际对数似然为 $\ell$，被试数 $n$，自由参数 $k$：

$$
\mathrm{AIC} = -2\ell + 2k, \qquad
\mathrm{BIC} = -2\ell + k\ln n, \qquad
\mathrm{AICc} = \mathrm{AIC} + \frac{2k(k+1)}{n-k-1}
$$

相对拟合指数以独立模型（各题仅估一个答对率）为零模型、饱和模型为上界：

$$
\mathrm{CFI} = 1 - \frac{\max(D_{\text{model}}-df_{\text{model}},\,0)}{D_{\text{null}}-df_{\text{null}}},
\qquad
\mathrm{RMSEA} = \sqrt{\frac{\max(D_{\text{model}}-df_{\text{model}},\,0)}{df_{\text{model}}\,(n-1)}}
$$

嵌套模型用似然比检验 $LR = 2(\ell_{\text{full}}-\ell_{\text{restricted}}) \sim \chi^2_{df}$，其中 $df=k_{\text{full}}-k_{\text{restricted}}$。χ² 上尾概率由正则化不完全伽马函数（级数展开 + Lentz 连分式）自实现，与 `scipy.stats.chi2` 对比误差 $< 3\times10^{-15}$，**不引入 scipy 依赖**。

### 绝对拟合：题目级 χ² / G² 与残差

AIC/BIC/CFI 是**相对**拟合指标，只能回答"哪个模型更好"，不能回答"这个模型够不够好"。
**绝对拟合**按"MAP 属性组 × 剩余得分分箱"把被试划分为单元，逐题比较观测与预测答对率：

$$
\chi_j^2 = \sum_c \frac{(O_{jc} - E_{jc})^2}{E_{jc}},
\qquad
G_j^2 = 2\sum_c O_{jc}\ln\frac{O_{jc}}{E_{jc}},
\qquad
df_j = \#\text{单元} - 1
$$

其中 $E_{jc} = \sum_{i \in c} P(X_j = 1 \mid \hat\alpha_i)$。为规避大样本把 p 值放大到"必然显著"，
系统同时报告**效应量**——单元残差的最大绝对值与均方根：

$$
\text{max resid}_j = \max_c \left| \frac{O_{jc} - E_{jc}}{\sqrt{E_{jc}(1 - E_{jc}/n_c)}} \right|
$$

> **判读建议**：优先看效应量而非 p 值。题目参数与数据同源估计，χ²/G² 的 p 值仅作近似参考。

### 局部依赖：Yen Q3

条件于属性掌握模式后，题间残差应当近似不相关。若两题共享 Q 矩阵**未记录**的隐藏属性，
则条件于 α 后仍会同向波动——这就是局部依赖。记残差

$$
d_{ij} = X_{ij} - P(X_j=1 \mid \hat\alpha_i)
$$

Yen **Q3** 即两题残差的 Pearson 相关：

$$
Q3_{jj'} = \operatorname{corr}_i\big(d_{ij},\ d_{ij'}\big)
$$

经验判据：$|Q3| > 0.2$ 提示可能存在局部依赖，应检查 Q 矩阵是否漏标了共同属性。

```python
from cdcat import analyse_absolute_fit, item_fit, local_dependence

af = analyse_absolute_fit(model, X, prior=res.attr_prior, n_bins=5)
af.item_fit.poorly_fitting(0.05)     # 不合格题目索引
af.item_fit.worst_items(5)           # 残差最大的题目
af.local_dependence.top_pairs(5)     # Q3 最高的题目对
af.local_dependence.n_flagged        # 超过阈值 (默认 0.2) 的对数
af.summary()
```

### 评估指标

- **PMR**（Pattern Match Ratio）：整条属性模式完全判对的比例
- **AMR**（Attribute Match Ratio）：逐属性一致的平均比例
- **题库利用率**：至少被使用一次的题目占比
- **曝光均匀性**：题目曝光次数的 $\chi^2$ 检验
- **PMR 理论上限**：由 Q 矩阵结构可辨识性决定的精度天花板

---

## 非参数分类（距离判别）

参数化 CDM 必须先标定 slip / guess，再据此算后验。标定会带来误差：样本太小、
题目刚上线、Q 矩阵有误时，参数估计不稳，分类精度随之下降。**距离判别**绕开这一步：
只用 Q 矩阵决定的理想反应模式 $\eta_j(\alpha)$ 与作答向量比对，**不估计任何题目参数**。

对属性模式 $\alpha$ 与第 $i$ 个被试的作答向量 $\mathbf{x}_i$：

$$
d_i(\alpha) = \sum_{j=1}^{J} w_j \,\bigl| x_{ij} - \eta_j(\alpha) \bigr|,
\qquad
\hat\alpha_i = \arg\min_{\alpha \in \mathcal{A}} d_i(\alpha)
$$

两种权重，对应两条经典路线：

| 方法 | 权重 | 说明 |
|---|---|---|
| `hamming` | $w_j = 1$ | 等权汉明距离，最简单；对区分度弱的题一视同仁 |
| `gdd` | $w_j \propto 1/[p_j(1-p_j)]$ | 广义距离判别（Leighton, Gierl & Hunka, 2004）：靠近 0.5 的题方差大、权重高，极端难/易题自动降权 |

$p_j$ 取样本通过率（拉普拉斯平滑后截断到 $[0.02, 0.98]$）。它是**样本统计量而非模型参数**，
所以 `gdd` 仍属非参数范畴。权重会归一化到均值 1，使距离的单位是"等效题数"。

**属性层级是天然约束。** 给定层级结构时，非法模式的距离直接置为 $+\infty$，
等价于 AHM 里"只在合法模式集中搜索"的做法 —— 不需要改似然、不需要重参数化：

$$
d_i(\alpha) = +\infty \quad \text{若 } \alpha \notin \mathcal{A}_{\text{legal}}
$$

**缺失作答**按"该题不提供信息"处理：$\sum_j$ 只对已作答的题求和，
既不是按 0 分算、也不是按 1 分算。

### 距离怎么变成"把握度"

报告与绘图需要一个 0~1 刻度，这里用

$$
\tilde\pi(\alpha) \propto \exp\!\bigl(-d(\alpha)/T\bigr)
$$

$T=1$ 表示"距离每多 1 个等效题，相对权重衰减 $e$ 倍"。

> ⚠️ **它是伪后验，不是真实后验**：没有似然、没有先验，只是距离的柔性排序。
> 它的用途是给出"判别把握"与连续色谱，**不能**和参数模型的后验概率直接比较，
> 也不该用它做贝叶斯决策。代码里字段叫 `posterior` 是为了复用现有绘图组件，
> 文档与界面文案统一写作"伪后验 / 判别把握"。

### 它到底有什么用

实测（60 题 / 5 属性 / 1500 人，真实模型 DINA，与上文拟合比较同一份数据）：

| 方法 | 自由参数 | PMR | AMR | 判别把握 |
|---|---|---|---|---|
| DINA（MMLE-EM） | 151 | **0.9607** | **0.9917** | — |
| G-DINA | 217 | 0.9607 | 0.9917 | — |
| 汉明（非参数） | **0** | 0.9573 | 0.9909 | 0.91 |
| GDD（非参数） | **0** | 0.9480 | 0.9891 | 0.91 |

两个值得注意的事实：

1. **零参数的距离判别只落后 0.3~1.3 个百分点。** 这就是"标定带来的真实增益"——
   在这份干净数据上很小。参数模型的价值主要体现在小样本、Q 矩阵有误、
   或需要题目参数做等值/DIF 等下游分析时。
2. **GDD 在这份题库上反而略低于等权汉明。** 原因是 GDD 的 $1/[p(1-p)]$ 是
   **方差标准化**权重：它压低中等难度题的比重、抬高极端难度题。当题库难度本就接近
   （本演示的题目都在 0.5~0.75 一带）时，这种"拉平"没有信息可拉，反而放大了噪声。
   **实践建议：先跑两种方法看 PMR，再决定用哪套权重**；难度跨度大时 GDD 通常更稳。

三个用途：

1. **兜底**：标定不可行（样本太少、参数未收敛）时仍能立刻给出知识状态；
2. **基线**：这个百分点差值就是标定的真实收益；若参数模型追不平非参数方法，
   问题多半在 Q 矩阵或模型设定，而不在优化算法；
3. **教学**：把"参数估计到底贡献了什么"变成一个可量化的差值。

### 非参数选题：MIV / NPSHE / ERHD / AMI

CD-CAT 的选题准则（KL / PWKL / …）都依赖题目参数。当参数不可得时（新题未标定、
样本太小、参数未收敛），可以用一组**只读 Q 矩阵**的准则。做法都是把后验按
"该题该答对 / 该答错"劈成两支，再衡量不确定性降了多少，区别只在用什么衡量：

| 准则 | 判据 | 直观解释 |
|---|---|---|
| **MIV** | $p_j(1-p_j)$，$p_j=\mathbb{E}_\pi[\eta_j]$ | 挑"最拿不准该不该答对"的题 |
| **NPSHE** | 模式级熵缩减 | SHE 的无参数版，最接近"信息量"直觉 |
| **ERHD** | 期望汉明距离缩减 | 直接优化"还会判错几个属性" |
| **AMI** | 属性边际熵缩减 | 只看每个属性是否确定，绕开模式不可辨识 |

四个准则与题目参数**逐位无关**（有测试钉住）：同一 Q 矩阵下换任何 slip/guess，
准则值完全不变。

> **实测提醒（见「实测性能」）**：四个非参数准则的 PMR 在 0.928~0.948 之间，
> 与参数化准则的差距落在 600 次施测的置信区间内。但 ERHD 的题库利用率只有
> 0.950、曝光标准差最高——直接对齐最终指标（属性误判个数）反而会把题量压在
> 少数几道题上。**目标函数与评价指标同口径，不等于它一定选得更好。**

---

## 结构可辨识性：为什么 PMR 达不到 1.0

这是本系统最重要的方法学贡献之一。

**问题**：在 `slip = guess = 0`（完美确定性作答）的理想条件下，
PMR 仍然只有 0.935 而非 1.0。这与算法无关，而是 **Q 矩阵本身的结构缺陷**。

**原理**：两个掌握模式 $\alpha, \alpha'$ 若对**所有题目**产生**完全相同**的理想反应向量
$\eta(\alpha) = \eta(\alpha')$，则无论作答多少题目、用什么算法，
观测数据都无法区分它们 —— 它们属于同一**等价类**。

$$
\alpha \equiv \alpha' \iff \forall j:\ \eta_j(\alpha) = \eta_j(\alpha')
$$

**理论上限**：

$$
\text{PMR}_{\max} = \frac{\#\{\text{可唯一辨识的模式}\}}{2^K}
$$

**接口**：

```python
from cdcat import analyse_identifiability

rep = analyse_identifiability(q)
print(rep.summary())
print(f"PMR 理论上限 = {rep.expected_pmr_ceiling:.4f}")
print(f"Q 矩阵是否完备 = {rep.is_complete}")

# 仿真结果与上限对比
gap = result.pmr_vs_ceiling()
print(gap)
# {'pmr': 0.83, 'ceiling': 0.875, 'gap': 0.045, 'efficiency': 0.949,
#  'n_unidentifiable': 4, 'verdict': '...'}
```

`analyse_identifiability` 会检出四类结构问题：

| 问题 | 含义 | 修复建议 |
|---|---|---|
| `equivalence_classes` | 理想反应相同的模式组 | 增加能区分它们的题目 |
| `uncovered_attrs` | 无任何题目考查的属性 | 补充该属性的题目 |
| `singleton_attrs` | 有独立测量题目的属性 | 属性有独立题才可单独标定 |
| `interchangeable_attr_pairs` | 无法互相区分的属性对 | 增加只考查其中之一的题目 |

> **实践建议**：报告 PMR 时应**始终与理论上限对比**。PMR=0.83 对于上限 0.875 的题库
> 已是 95% 的效率；而对上限 1.0 的题库则仍有优化空间。脱离上限谈 PMR 没有意义。

---

## 属性层级关系 AHM

心理学上属性之间几乎从不互相独立：不会通分就谈不上分数加减。
**AHM（Attribute Hierarchy Method，Leighton, Gierl & Hunka, 2004）**
把这种先备结构显式写进模型，收益是双重的 ——
**合法模式从 $2^K$ 塌缩到少数几个**，辨识性上限大幅抬升，同时诊断更贴合真实认知结构。

```python
from cdcat import AttributeHierarchy, QMatrix, CDCATSimulator

h = AttributeHierarchy.linear(4, ["A1", "A2", "A3", "A4"])   # 完全线型：A1→A2→A3→A4
h = AttributeHierarchy.convergent(4)                          # 汇聚型
h = AttributeHierarchy.divergent(4)                           # 发散型
h = AttributeHierarchy.from_edges(4, [(0, 1), (1, 2)])        # 自定义先备边
h = AttributeHierarchy.from_adjacency(matrix)                 # 由邻接矩阵构造
h = AttributeHierarchy.none(4)                                # 等价于不作约束

h.n_legal_patterns        # 合法模式数（线型 K=4 → 5）
h.edges                   # 直接先备边
h.reachability()          # Warshall 可达矩阵
h.all_prerequisites(2)    # 属性 2 的全部（传递）先备
h.levels                  # 每个属性的层级深度
h.legal_patterns()        # 全部合法模式
h.is_legal([1, 1, 0, 0])  # 模式是否合法
h.project([0, 1, 0, 0])   # 越界模式投影回最近合法模式 -> [1, 1, 0, 0]
h.reduced_q_matrix(q)     # 用层级化简 Q 矩阵
h.to_dict() / AttributeHierarchy.from_dict(d)
```

接进流水线只需一个参数：

```python
# ① 辨识性分析：在合法模式集合内重算上限
rep = analyse_identifiability(q, hierarchy=h)
rep.expected_pmr_ceiling          # 层级下的上限
rep.expected_pmr_ceiling_no_hierarchy   # 无层级时的上限（用于对比）
rep.hierarchy_structure, rep.n_legal_patterns
rep.summary()                     # 会显示层级带来的收益

# ② 估计器：非法模式的先验直接清零，绝不进入后验
est = AttributeEstimator(model, method="MAP", hierarchy=h)

# ③ 仿真器：只在合法模式上抽真实知识状态
sim = CDCATSimulator(q, model="DINA", model_params=..., hierarchy=h)

# ④ 层级 × Q 矩阵一致性：层级要求掌握 A4 的人必须掌握 A3，但题库里根本没有题能
#    同时测出二者 —— 这种矛盾会被显式报出来
check_hierarchy_consistency(q, h)
```

### 实测收益

**① 上限抬升**（构造 A4 只与 A3 同现的不完备 Q 矩阵，K=4）：

| 设定 | 合法模式数 | PMR 理论上限 |
|---|---|---|
| 无层级 | 16 | 0.7500 |
| 线型 / 汇聚层级 | **5** | **1.0000** |

> 同一份 Q 矩阵，只是承认了"A4 蕴含 A3"这一先备关系，理论上限就从 0.75 抬到 **1.00（+0.25）**。

**② 仿真精度与效率**（30 题 / 4 属性 / 500 被试）：

| 指标 | 无层级 | 线型层级 |
|---|---|---|
| PMR | 0.8278 | **0.9611** |
| AMR | 0.9514 | **0.9869** |
| 平均用题量 | 6.51 | **5.21** |

> PMR 提升 **13.3 个百分点**，同时**少用 1.3 道题**。
> 这不是"更严格的先验换精度"的取舍，而是**减少假设**带来的纯收益：
> 排除了现实中不存在的知识状态，等于免费获得了一批先验信息。

---

## Q 矩阵自动修正（Qval / PVAF）

Q 矩阵是 CDM 的命门，但它通常由学科专家**手工编写**，出错率高且难以自查：
题目真正考了 3 个属性，专家只标了 2 个；或者把"运算能力"顺手标上，其实并不需要。
**Qval**（de la Torre & Chiu, 2016）与 **PVAF**（Chiu, 2013）
用**作答数据反向验证 Q 矩阵**：对每题穷举所有属性子集，找出最能解释该题得分的那个。

```python
from cdcat import validate_q_matrix, calibrate

# 先标定模型（Qval 要用它推断每名被试的属性模式）
model, _ = calibrate(q, responses, "DINA")

rep = validate_q_matrix(q, responses, model, pvaf_threshold=0.95)
print(rep.summary(top_n=10))

rep.n_suspect          # 需修正的题目数
rep.n_additions        # 建议「加属性」的题数
rep.n_removals         # 建议「删属性」的题数
rep.mean_pvaf          # 全库平均 PVAF（Q 矩阵整体质量）
rep.to_dataframe()     # 逐题：原 Q 行 / 建议 Q 行 / 原 PVAF / 最优 PVAF / 改善量

for s in rep.suggestions[:3]:
    print(s.describe())
    # 新增 通分；删除 分数乘除
    s.current_attrs_str(), s.suggested_attrs_str()   # "通分+分数加减" -> "通分+约分+分数加减"
    s.added, s.removed, s.improvement                # 属性索引 + PVAF 增益

q_fixed = rep.apply(q, min_improvement=0.10)   # 应用建议（生成新矩阵，不改原对象）
```

### 实测：漏标检测与误报控制

**① 人为漏标一道题**（3000 被试 / 30 题 / 4 属性，把第 5 题的一个属性抹掉）：

| 项目 | 结果 |
|---|---|
| 检出题目数 | **1**（精确命中第 5 题） |
| 建议属性 | `A → A+B` —— 与真实行**完全一致** |
| PVAF 变化 | 0.438 → **0.903** |

> 全库 30 题只报了这 1 题，**没有误报**。

**② 正确 Q 矩阵上跑**：`n_suspect = 0`（阈值 0.95）—— 不制造虚假告警。

**③ 整库体检**（48 题 / 800 被试）：平均 PVAF **0.8846**，未发现需修正项。

> **判读建议**：优先看 PVAF 的**绝对水平**和**改进幅度**。
> PVAF < 0.6 说明该题的属性标注几乎解释不了作答差异，是最该人工复核的题目。

---

## 多级评分 CDM（顺序 G-DINA）

现实测验大量使用**多级评分**：0/1/2/3 分的过程性评分、Likert 式理解程度、
"完全不会 / 部分掌握 / 完全掌握"。顺序 G-DINA（Ma & de la Torre, 2016）
把第 $j$ 题建模为一串**顺序步骤** —— 只有走到第 $m-1$ 步的人才可能走到第 $m$ 步。

```python
from cdcat import (
    PolytomousQMatrix, SequentialGDINA, calibrate_polytomous,
    build_polytomous_bank, create_poly_selector,
    expand_to_steps, collapse_step_responses,
)

# ① 构造多级 Q 矩阵：每题 2~4 档
pq = PolytomousQMatrix.sequential(matrix, n_categories)   # 属性即步骤
pq = PolytomousQMatrix.from_steps(step_q, item_index)     # 显式指定每题步骤
pq = PolytomousQMatrix.from_binary_qmatrix(q, n_categories)

pq.n_items, pq.n_steps, pq.n_categories, pq.max_categories
pq.step_slice(j)              # 第 j 题占用的步骤行
pq.union_q_matrix()           # 合并成二分 Q 矩阵（供结构分析）

# ② 标定（专用条件 EM，非二分展开）
model, res = calibrate_polytomous(pq, responses, max_iter=500, tol=1e-6)
res.n_steps, res.n_categories, res.aic, res.bic, res.aicc
model.step_probs()            # (2^K, S) 各步骤答对概率
model.category_probs()        # (2^K, J, Cmax) 各档位概率

# ③ 接入完整的 CD-CAT 链路：多级模型可直接传给既有仿真器
sim = CDCATSimulator(pq.union_q_matrix(), model=model, ...)
sim.run(n_examinees=300, n_replications=3)

# ④ 多级选题准则
sel = create_poly_selector("PWKL", model)     # 另有 PolyKL / PolyGDI
```

### 实测：参数恢复（顺序 G-DINA，K=4 / 20 题 / 2~4 档）

| 被试数 | slip 相关系数 | guess 相关系数 | slip MAE | guess MAE | PMR |
|---|---|---|---|---|---|
| 400 | 0.77 | 0.78 | 0.024 | 0.026 | 0.98 |
| 1500 | 0.91 | 0.92 | 0.014 | 0.015 | 0.99 |
| 4000 | **0.98** | **0.96** | **0.007** | **0.008** | 0.98 |

> 恢复精度**随样本量单调改善**，符合 MMLE 的渐近性质。BIC 随 N 上升（正常，惩罚项变重）。

### 实测：全链路自适应施测（模型实例直接接入既有仿真器）

| 选题准则 | PMR | AMR | 平均题量 |
|---|---|---|---|
| PolyPWKL | 0.8644 | >0.965 | ~5.5 |
| PolyKL | 0.8822 | >0.965 | ~5.5 |
| PolyGDI | 0.8722 | >0.965 | ~5.5 |

> 该题库的可辨识上限为 0.875，PolyKL 已达 **0.8822**（效率 100.8%）。
> 终止原因以"模式后验与属性边际均达到 0.8"为主，说明多级评分提供的额外信息
> 让后验收敛得更快。

> **界面**：「多级评分」页可从零构造多级题库 → 一键条件 EM 标定 → 查看步骤参数恢复散点与
> 档位概率曲线 → 直接跑多级自适应施测，全程无需写代码。

> **往返一致性保证**：`expand_to_steps` 把多级分数展开成步骤作答、
> `collapse_step_responses` 再收回来，二者严格互逆（缺失值在展开后被正确标记为"未走到该步"）。

---

## 绝对拟合与局部依赖

相对拟合（AIC/BIC/CFI）只能**在候选模型间排序**，不能告诉你"现有模型对这份数据到底够不够好"。
绝对拟合填补了这个缺口：它把测验的**每个单元**的观测答对率与该模型预测的答对率逐一比对。

系统提供了两个互补的诊断，均已接入 `run_cdm_analysis` 流水线并可在界面一键查看：

| 诊断 | 回答的问题 | 关键输出 |
|---|---|---|
| **题目级绝对拟合** | 模型能不能重现每题在各能力组上的答对率？ | Pearson χ²、似然比 G²、df、p、残差效应量 |
| **Yen Q3 局部依赖** | 条件于属性模式后，题间残差是否仍相关？ | Q3 矩阵、超标题目对、平均/最大 Q3 |

### 判别力实测（真实模型 = DINO，2000 名考生，30 题）

| 拟合模型 | 判定 | 不合格题目 | 最大残差均值 | Q3>0.2 对数 | 最大 Q3 |
|---|---|---|---|---|---|
| **DINO** | 设定正确 | **3 / 30（10%）** | **0.1225** | **0** | 0.0664 |
| DINA | 模型误设 | **21 / 30（70%）** | **0.2667** | **32** | 0.4966 |

> 同一份数据，只是把拟合模型从正确的 DINO 换成 DINA，
> 不合格题目占比就从 10% 飙到 70%、Q3 超标对数从 0 升到 32。
> 这说明该诊断对"模型-数据不匹配"非常敏感，可作为建模前的**体检**步骤。

### 检出隐藏属性（局部依赖）实测

在模拟数据中人为让第 6 题与第 19 题（0 基索引 5、18）同时受一个 Q 矩阵未记录的隐藏属性支配
（隐藏属性掌握者两题高概率答对、未掌握者两题低概率答对）：

| 指标 | 数值 |
|---|---|
| 全体平均 Q3 | 0.0009 |
| 最大 Q3 | **0.4817** |
| Q3 最高的题目对 | **I6 × I19（Q3 = 0.4817）** |
| 超出阈值（>0.2）的对数 | 1 |

> Q3 矩阵在**满屏约 0 的噪声中**精准地把 (I6, I19) 顶成唯一超标对 —— 这正是
> "Q 矩阵漏标了共同属性"的典型信号。其余题目对的 Q3 都低于 0.07，没有误报。

### 使用方式

```python
from cdcat import analyse_absolute_fit

res = EMCalibrator(q, "DINA").fit(X)
model = res.to_model(q)

af = analyse_absolute_fit(model, X, prior=res.attr_prior, n_bins=5)
print(af.summary())

af.item_fit.to_dataframe()          # 每题一行：chi_square / g_square / df / p / 残差
af.item_fit.poorly_fitting(0.05)    # p < 0.05 的题目索引
af.item_fit.worst_items(5)          # 残差效应量最大的 5 题

af.local_dependence.q3              # (J, J) Q3 矩阵
af.local_dependence.top_pairs(5)    # Q3 最高的题目对
af.local_dependence.flagged_pairs() # 仅超阈值（默认 0.2）的对
```

在 `run_cdm_analysis` 里默认开启（`absolute_fit=True`），结果挂在 `analysis.absolute_fit`，
并随 `to_excel()` 一起导出为「题目绝对拟合」「局部依赖Q3」「属性先验」三张 sheet。

> **重要提醒**：题目参数与这批数据是同一批估计出来的，χ²/G² 的 **p 值会被大样本放大**，
> 几乎必然会"显著"。**判读请优先看效应量（最大残差、Q3 大小）**，p 值只作辅助参考。

---

## 题目分析与冗余检测

```python
from cdcat import analyse_items, create_model

model = create_model("DINA", q, slip=item_slip, guess=item_guess)
ia = analyse_items(model, n_simulate=4000, seed=2024)

print(ia.discrimination)          # 每题区分度 (2^K 加权期望)
print(ia.information)             # 每题信息量
print(ia.strongest_items[:5])     # 最强题索引
print(ia.weakest_items[:5])       # 最弱题索引
print(ia.structural_duplicates)   # Q 向量完全相同的冗余题组
```

**冗余检测的两个层次**：

1. **结构冗余**（`structural_duplicates`，主判据）：Q 向量完全相同 → 诊断信息严格等价。
2. **统计冗余**（`redundant_pairs`，辅判据）：理想反应高度相关 → 信息重叠。

> 注意：Q 向量相同的题目，其**实际作答**相关系数通常只有 ~0.6，
> 因为每题的 slip/guess 噪声相互独立。因此**必须用结构判据**，不能只看作答相关。

---

## 个人诊断报告

```python
from cdcat import build_examinee_report

rec = sim.administer(true_pattern, rng=rng)
report = build_examinee_report(
    rec, q.attr_names, q.item_names, q, include_truth=True
)

print(report.to_text())        # 人类可读报告
data = report.to_dict()        # 结构化数据（可入库）
print(report.mastered)         # 已掌握属性索引（property）
print(report.accuracy)         # 属性判准率（property）
print(report.correct_count)    # 答对题数（property）
print(report.low_confidence_attrs(threshold=0.6))  # 需复核的属性
```

报告包含：用题量、答对数、诊断模式与置信度、逐属性掌握概率与置信度条、
**低置信度属性警告**、完整作答明细。

---

## 群体 / 班级诊断报告

个人报告回答"这个学生哪里不会"；**群体报告回答"这个班/这个群体整体哪里不会"**
—— 后者才是教师调整教学进度的直接依据。

```python
from cdcat import build_group_report

# 路径 A：从真实作答矩阵出发（内部逐人跑 MAP/EAP）
rep = build_group_report(
    attr_names=q.attr_names,
    q_matrix=q,
    responses=X,
    examinee_ids=ids,
    group_labels=grade,          # 可选：每人所属班级/组别
    method="MAP",
    missing="skip",              # 缺失作答：skip / wrong / ignore
    label="高一(3)班 诊断",
)
# 路径 B：从仿真 records 出发（带真值，可算真实掌握率）
rep = build_group_report(records=sim_result.records, attr_names=..., q_matrix=q,
                         include_truth=True)
```

```python
rep.attr_mastery_rate       # 各属性掌握率（MAP 判定）
rep.attr_mean_prob          # 各属性掌握概率均值
rep.attr_sd_prob            # 概率标准差 → 群体同质性
rep.mean_mastered           # 人均掌握属性数
rep.n_full_mastery          # 全掌握人数
rep.n_zero_mastery          # 零掌握人数
rep.full_mastery_rate

rep.weakest_attrs(3)        # 最薄弱的 3 个属性 ← 教学重点
rep.strongest_attrs(3)      # 掌握最好的 3 个属性
rep.most_common_patterns(5) # 最高频知识状态

rep.group_attr_mastery()    # 逐组掌握率对比（需 group_labels）
rep.compare_examinee(12)    # 第 12 人 vs 群体常模的百分位
rep.attr_dataframe()        # 属性表（按掌握率升序，最薄弱在最上）
rep.pattern_dataframe(10)
rep.group_dataframe()
rep.to_text() / rep.to_dict()
```

### 报告图（`cdcat.figures`）

```python
from cdcat.figures import (
    plot_attribute_mastery,   # 单图：掌握率横向条形，薄弱在上，颜色分档
    plot_group_report,        # 四联图：掌握率 / 确定性分档 / 典型模式 / 雷达图
    plot_convergence,         # 三连图：边际收敛 / 用题量分布 / 曝光分布
)
fig = plot_group_report(rep, threshold=0.85)
fig.savefig("group.png", dpi=150, bbox_inches="tight")
```

四联图的第二张刻意做成**确定性分档堆叠条**而非箱线图：
每段表示该属性上 `P<0.2`（确定未掌握）/ `0.2~0.8`（模棱两可）/ `P>0.8`（确定掌握）的人数占比。
**橙色段越长，说明这个属性越需要继续加题才能判断清楚** —— 这是普通箱线图看不见的信息。

> **界面**：「群体报告」页可直接生成合成数据或上传作答矩阵，勾选属性层级与分组表后一键出报告，
> 内含掌握率条、确定性分档、典型模式、雷达图、分组对比与个体 vs 群体常模。

命令行等价用法：

```bash
cdcat report -q q.csv -r X.csv --groups grade.csv --missing skip \
             --label "高一(3)班" --top 3 -o out/report --figures
# 生成 group_report.txt / group_report.json / group_report.png
#      attribute_mastery.csv / group_comparison.csv
```

> `--groups` 既支持两列 `被试ID,组名`（按 ID 匹配，**与行序无关**），
> 也支持单列组名（顺序须与作答矩阵一致）。若某被试在分组表里缺失，会明确报错而不是静默填默认值。

---

## 曝光控制

高区分度题目会被反复选中，导致题库过载与题目泄露。四种策略均已实现：

```python
from cdcat import (
    MaxExposureLimit, RandomesqueExposure,
    StratifiedExposure, SympsonHetter,
)

# 硬上限：单题曝光率不超过 20%
ctrl = MaxExposureLimit(q.n_items, max_rate=0.2, warmup=18)

# Sympson-Hetter：更精细的概率化控制，支持自动标定
ctrl = SympsonHetter(q.n_items, target_rate=0.2)
ctrl.calibrate(sim, n_examinees=200, n_replications=2,
               n_iterations=6, seed=0)

sim = CDCATSimulator(..., exposure_control=ctrl)
```

**实测效果**（40 题题库，300 被试 × 5 次重复）：

| 策略 | 最高曝光率 | 曝光 SD | PMR 变化 | 额外题量 |
|---|---|---|---|---|
| 无控制 | 100% | 0.237 | — | — |
| 随机化选题 | 87% | 0.181 | -0.008 | +0.6 |
| 曝光硬上限 0.2 | 80% | 0.137 | -0.012 | +3.1 |
| Sympson-Hetter | **53%** | **0.089** | -0.021 | +3.4 |

---

## 使用方法

### Python API

```python
import numpy as np
from cdcat import QMatrix, build_test_items, CDCATSimulator, Evaluator

# 1. 构建题库
q = QMatrix.from_random(60, 5, seed=42,
                        attr_names=["通分", "约分", "分数加减", "分数乘除", "应用题"])
items = build_test_items(60, 5, seed=42)

# 2. 创建仿真器
sim = CDCATSimulator(
    q,
    model="DINA",
    model_params={"slip": items["slip"], "guess": items["guess"]},
    selector="PWKL",
    estimator="MAP",
    max_items=30,
)

# 3. 单个被试施测
rng = np.random.default_rng(0)
record = sim.administer(np.array([1, 0, 1, 1, 0]), rng=rng)
print(record.n_answered, record.pattern_correct)

# 4. 蒙特卡洛仿真
result = sim.run(n_examinees=300, n_replications=10, seed=0)
print(result.summary())

# 5. 评估
report = Evaluator().evaluate(result, label="PWKL")
print(report.summary())

# 6. 多准则对比（固定 20 题，保证公平）；把 MIV 一起放进去就是无参数对照
results = sim.compare_criteria(
    ["KL", "PWKL", "MPWKL", "GDI", "SHE", "CDI", "MIV"],
    n_examinees=200, n_replications=5, seed=0, fixed_length=20,
)
print(Evaluator().compare_text(results))
```

### 从作答数据标定参数（单模型）

```python
from cdcat import EMCalibrator

# responses: (n_examinees, n_items) 的 0/1 矩阵
# 默认在 EM 的 M 步一并估计属性模式先验 π(α)（经验贝叶斯）
cal = EMCalibrator(q, model_type="DINA", max_iter=300, tol=1e-7)
result = cal.fit(responses, verbose=True)
print(result.summary())        # 含收敛信息、参数构成、π(α) 统计、AIC、BIC

# 若不希望估计先验（退回均匀先验）：
cal0 = EMCalibrator(q, "DINA", estimate_prior=False)

# 直接得到模型实例
model, result = cal.calibrate_to_model(responses)

# 估计各考生属性模式
patterns = cal.examinee_patterns(responses, result.slip, result.guess)

# 题目区分度 = 1 - slip - guess
disc = result.slip * 0 + cal.item_discrimination(result.slip, result.guess)
```

六模型通过同一接口切换，别名不区分大小写：

```python
EMCalibrator(q, model_type="G-DINA")   # 或 "GDINA" / "g_dina"
EMCalibrator(q, model_type="RRUM")     # 结果含 .pi_star 与 .r_star（列表，每题一个向量）
EMCalibrator(q, model_type="ACDM")     # 结果含 .delta0 与 .delta（加性模型，Least Squares M 步）
EMCalibrator(q, model_type="LLM")      # 结果含 .llm_beta0 与 .llm_beta（logit 主效应，IRLS M 步）
EMCalibrator(q, "HO-DINA")             # ValueError：未实现的模型会明确报错
```

> 无论哪种模型，`result.slip` / `result.guess` 都可用：它们是由概率矩阵反推的
> **等价 DINA 参数**，供题目区分度、仿真器、模型比较表这些只认这两个量的
> 下游代码使用；模型自己的原始参数在同名的专属字段里。

---

### 多模型 CDM 分析流水线（推荐入口）

一条调用完成：**多模型标定 → 拟合指标 → 模型择优 → LRT → 被试诊断 → 参数恢复 → Excel 导出**。

```python
import numpy as np
from cdcat import QMatrix, build_test_items, DINA, run_cdm_analysis

q = QMatrix.from_random(60, 5, seed=42)

# --- 生成带真值的模拟数据 ---
items = build_test_items(60, 5, min_attrs_per_item=1, max_attrs_per_item=2, seed=42)
truth = DINA(q, slip=items["slip"], guess=items["guess"])
rng = np.random.default_rng(1)
alphas = q.attribute_patterns()[rng.integers(0, q.n_patterns, 1500)]
X = np.array([truth.simulate(a, rng=rng) for a in alphas], dtype=float)

# --- 六模型标定与比较（默认估计属性先验 π(α)，并跑绝对拟合诊断）---
analysis = run_cdm_analysis(
    q, X,
    models=("DINA", "DINO", "G-DINA", "RRUM"),
    true_slip=items["slip"], true_guess=items["guess"],
    true_patterns=alphas,          # 提供后自动补 PMR / AMR 到拟合表
    estimate_prior=None,           # None=自动；False=固定均匀先验
    absolute_fit=True,             # 题目级 χ²/G² + Yen Q3
    max_iter=300, tol=1e-7,
)

print(analysis.fit_table())        # 含 aic/bic/aicc/cfi/tli/rmsea/pmr/amr 与参数构成的 DataFrame
print(analysis.comparison.summary())   # AIC / BIC / AICc 各自最优模型
print(analysis.best_model)         # 按 BIC 择优 -> 'DINA'

print(analysis.lrt)                # {'lr_statistic': 181.44, 'df': 164, 'p_value': 0.167, ...}
print(analysis.absolute_fit.summary())    # 绝对拟合 + 局部依赖摘要
print(analysis.diagnosis_table())  # 被试诊断明细（每行一名考生）
print(analysis.param_table())      # 题目参数表（含 Q 向量、slip、guess、区分度）
analysis.to_excel("cdm_report.xlsx")   # 多 sheet：拟合 / 参数 / 诊断 / 绝对拟合 / Q3 / 属性先验
```

#### 单独使用各组件

```python
from cdcat import (
    EMCalibrator, compute_fit_indices, saturated_log_likelihood,
    diagnose_patterns, compare_models, likelihood_ratio_test, parameter_recovery,
)

res = EMCalibrator(q, model_type="G-DINA", max_iter=300).fit(X)

# 拟合指标
fit = compute_fit_indices(X, res.log_likelihood, res.to_model(q), model_name="G-DINA")
print(fit.aic, fit.bic, fit.aicc, fit.cfi, fit.rmsea)

# 被试诊断：MAP / EAP 双路 + 完整后验
diag = diagnose_patterns(
    res.to_model(q), X, method="MAP", attr_names=["通分", "约分", "分数加减"]
)
diag.map_patterns        # (n, K) MAP 知识状态
diag.eap_marginal        # (n, K) 属性边际掌握概率
diag.attr_reliability    # (n, K) 属性级分类信度 = 2|P(α_k=1|X) - 0.5|
diag.map_confidence      # (n,)   MAP 模式后验概率
diag.mastery_proportion()          # 各属性掌握比例
diag.low_reliability_attrs(0.6)    # 每个被试中信度不足的属性索引
diag.summary()                     # 文本汇总

# 模型比较与 LRT
cmp = compare_models({"DINA": fit_dina, "G-DINA": fit_gdina})
print(cmp.best_bic)
lrt = likelihood_ratio_test(
    ll_restricted=fit_dina.log_likelihood, n_params_restricted=fit_dina.n_params,
    ll_full=fit_gdina.log_likelihood,     n_params_full=fit_gdina.n_params,
)

# 参数恢复
print(parameter_recovery(res.slip, items["slip"]))   # r / mae / rmse / bias
```

#### 结果可视化

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cdcat.calibration_plots import calibration_report_figure

fig = calibration_report_figure(analysis)          # 6 面板综合报告
fig.savefig("cdm_report.png", dpi=150, bbox_inches="tight")
```

单图函数（matplotlib，适合报告）：`plot_ll_convergence`、`plot_parameter_recovery`、
`plot_item_discrimination`、`plot_fit_comparison`、`plot_mastery_proportion`、
`plot_attr_reliability`、`plot_pattern_heatmap`、`plot_item_fit`、`plot_q3_heatmap`、
`plot_attr_prior`。

交互式图（Plotly，适合界面）：`demo.cdm_charts` 下的 `fig_ll_convergence`、
`fig_param_recovery`、`fig_item_discrimination`、`fig_fit_comparison`、
`fig_mastery_proportion`、`fig_attr_reliability`、`fig_pattern_distribution`、
`fig_item_fit`、`fig_q3_heatmap`、`fig_attr_prior`。

---

### 自定义终止规则

```python
from cdcat import CompositeRule, PatternConfidence, StabilityRule, FixedLength

rule = CompositeRule(
    [
        PatternConfidence(threshold=0.85),   # 模式后验 + 属性边际双达标
        StabilityRule(n_stable=5),           # 或连续 5 题估计不变
    ],
    how="any",       # 任一满足即停止
    min_items=5,     # 最少 5 题
    max_items=40,    # 最多 40 题
)

sim = CDCATSimulator(q, model="DINA", model_params=..., termination=rule)
```

### 使用自定义 Q 矩阵

```python
q = QMatrix(
    matrix=[[1, 0, 0], [1, 1, 0], [0, 1, 1]],
    attr_names=["属性A", "属性B", "属性C"],
    item_names=["题1", "题2", "题3"],
)
print(q.diagnose().summary())   # 上线前务必先诊断
```

---

## 后验预测检验 PPMC

绝对拟合（χ²/G²）问的是"每个题目的作答曲线对不对"，PPMC 问的是更狠的问题：
**用这套模型重新生成一遍数据，能不能长出和真实数据一样的二阶结构？**

四个统计量都刻意避开了"边际答对率"：

| 统计量 | 衡量的东西 | 失配意味着 |
|---|---|---|
| `pair_srmsr` | 题目对标准化残差均方根 | 题目间的相关结构没被模型抓住 |
| `q3_max` | 最大 \|Yen Q3\| | 存在模型没写进 Q 矩阵的共享属性 |
| `joint_max` | 题目对联合答对率的最大偏差 | 局部依赖 / 题干互相提示 |
| `score_sd` | 总分标准差 | 总分分布过于集中或发散 |

> **为什么盯二阶结构？** DINA/DINO 每题两个自由参数，EM 收敛时似然方程保证
> $\sum_i \bar p_{ij} \equiv \bar x_j$ —— **边际答对率被精确匹配**，拿它做统计量
> 观测侧恒为 0，剩下的只有数值噪声。真正有信息量的是题目对之间的关联。

### 两个实现上的坑

1. **两侧口径必须对称**。观测侧若用 MAP 硬模式的条件概率，MAP 模式本身就是
   挑出来最贴合该人作答的那个，天然过拟合 → 观测统计量被压小 → PPP 系统性趋近 1
   → 正确模型也会被判失配。观测侧与复制侧统一改用**后验期望答对概率**。
2. **属性由数据估计时，复制数据必须重走完整估计流程**（含题目参数重估）。
   否则"真实参数生成的复制数据"和"估计参数算出的观测值"不是同一个口径。

此外，缺失作答必须用**成对完整信息**（pairwise-complete）统计：只在该题对被两题
都作答的被试上算联合率，否则填补值相乘会制造出虚假的题目对关联，把 SRMSR 抬高到误判水平。

### 实测

| 场景 | `pair_srmsr` | `q3_max` | `joint_max` | 结论 |
|---|---|---|---|---|
| DINA 真值 → DINA 标定 | PPP 0.44 | 0.49 | 0.41 | 未检出失配 ✓ |
| DINO 真值 → 误用 DINA | **0.000** | **0.000** | **0.000** | 三个统计量同时报警 ✓ |
| 人为注入局部依赖（题 3/4 抄 90%） | — | **0.000** | — | 定位到被注入的那一对 ✓ |

Type I 误差标定：12 组重复 × 4 个统计量 = 48 次检验，假阳性 4 次（**8.3%**，
名义 5%，在抽样误差内）；PPP 中位数 0.41–0.49；对 `refit_max_iter ∈ {8, 20, 45}`
完全不敏感。

```python
from cdcat import posterior_predictive_check

pp = posterior_predictive_check(model, responses, n_replicates=500, seed=2024)
print(pp.to_dataframe())      # 统计量 / 观测值 / 复制均值 / 复制 SD / PPP / 判读
print(pp.verdict())
```

复制侧重估默认开启（`refit=True`），单次约 11.5 ms，300 份复制约 3.5 秒。

---

## 属性层级结构比较

属性之间常有先备关系（不会约分就不该会分数加减）。承认层级能把合法模式从
$2^K$ 压到少数几个，**直接抬高判准率上限**；但层级写错会引入系统偏差。
`compare_hierarchies` 用 BIC 在四种候选里选：

| 候选 | 含义 | 合法模式（K=4） |
|---|---|---|
| `none` | 属性相互独立 | 16 |
| `linear` | 逐级先备 A1→A2→A3→A4 | 5 |
| `convergent` | 末端汇聚 | 9 |
| `divergent` | 多点发散 | 9 |

### 为什么以 BIC 为主判据

受限模型位于**参数空间边界**（非法模式的先验被钉在 0），χ² 的常规自由度近似偏保守，
LR 检验只能当参考；BIC 直接惩罚参数个数，在边界情形下更稳。

### 一个必须显式处理的实现细节

`_update_prior` 里的 `prior_smoothing=1e-6` 会把 0 先验抬成 1e-6，随后每轮指数式回涨，
**几十轮后层级约束形同虚设**（四个候选的 logL 会一模一样）。因此比较时传入
`lock_prior_zeros=True`，用掩码把非法模式的先验永久锁死在 0。

### 实测（四个真值各跑一遍）

| 真值结构 | 选中 | ΔBIC | BIC 权重 |
|---|---|---|---|
| 线型 | `linear` ✓ | 26.7 | 1.00 |
| 收敛型 | `convergent` ✓ | 46.2 | — |
| 发散型 | `divergent` ✓ | 43.6 | — |
| 无层级 | `none` ✓ | —（判为"不支持任何层级约束"） | — |

```python
from cdcat import compare_hierarchies

rep = compare_hierarchies(q_matrix, responses, model_type="DINA")
print(rep.to_dataframe())
print(rep.verdict())          # 一句话结论
```

---

## 题库装配：从大题库到短测验

给定 Q 矩阵与题目参数，挑 $k$ 道题，使**最坏情况下的模式可分性**最大：

$$
\max_{S,\ |S|=k}\ \min_{s \ne s'}\ D(s, s'),\qquad
D(s,s')=\sum_{j\in S}\Big[p_{sj}\ln\frac{p_{sj}}{p_{s'j}}+(1-p_{sj})\ln\frac{1-p_{sj}}{1-p_{s'j}}\Big]
$$

$D$ 是模式 $s$ 与 $s'$ 之间的 KL 距离；取所有模式对的**最小值**，等于"保护最难分开的
那一对"，直接对应 PMR 的短板。KLD 不对称，故用 Jeffreys 对称化
$D=(D_{s\to s'}+D_{s'\to s})/2$。

### 贪心在 max-min 下的退化

题量少时，大量候选的目标值并列 0；若与"属性覆盖"混在同一个排序键里，贪心会退化成
"按题号顺序取题"。解决：

1. 覆盖阶段**单独成阶段**，先把 `min_per_attr` 满足；
2. 目标阶段只用 `np.lexsort((-means, -mins))`，并列时用平均 KL 破平；
3. 覆盖不足时用交换修复，最后跑成对交换局部搜索。

约束不可行时**直接报错**而不是静默放宽：

```python
if n_items > max_per_attr * k:
    raise ValueError("约束不可行：…")
```

### 实测

80 题题库选 16 题：期望 PMR 从随机抽样的均值 **0.630**（最好 0.780）提到 **0.837**（+33.0%）；
K=5 / J=120 时，贪心 0.894 → 局部搜索后 **0.911**。

```python
from cdcat import assemble_test_bank

res = assemble_test_bank(q_matrix, model, n_items=16, min_per_attr=2, max_per_attr=6)
print(res.selected_names, res.expected_pmr, res.attr_coverage)
```

---

## 新题在线标定

新题刚入库、还没做过大样本标定，但它已经被考过了。利用**已有模型给出的属性后验**
$\hat\alpha_i$，用后验加权的 MMLE 反推新题参数：

$$
n_{1j}=\sum_i\sum_s w_{is}\eta_{sj},\quad
r_{1j}=\sum_i\sum_s w_{is}\eta_{sj}x_{ij},\quad
\hat\sigma_j=1-\frac{r_{1j}}{n_{1j}},\quad
\hat g_j=\frac{r_{0j}}{n_{0j}}
$$

$\eta_{sj}$ 是模式 $s$ 下"该题所需的属性全部掌握"的指示；$w_{is}$ 是被试 $i$ 属于模式 $s$
的后验概率。Beta(1,4) 先验把极端值往合理区间收，标准误与置信区间取 Beta 后验解析解。

> **关键洞察**：判定可靠性取决于**掌握组 / 未掌握组的有效人数**，而不是总样本量。
> 1000 人里若只有 8 人掌握该题要求的属性，这题照样标不准 —— 报告里因此给出
> `n_masters` / `n_nonmasters` 与 `unstable` 标记。

实测（新题 8 道、600 人）：slip 相关 **0.882** / MAE 0.025，guess 相关 **0.981** / MAE 0.018；
25 人的小样本下 10 道题里有 5 道被标 `unstable` —— 这正是它该做的事。

```python
from cdcat import calibrate_new_items

rep = calibrate_new_items(model, responses, new_items=[40, 41, 42])
for it in rep.items:
    print(it.item_name, it.slip, it.guess, it.status)
```

---

## 单文件 HTML 报告

`build_cdm_report` 把一次完整分析压成一份**自包含** HTML：CSS 内联在 `<style>`，
图以 base64 PNG 内嵌，**不含任何 `http://` / `<script>` / `<link>`**，可离线打开、
可邮件发送、可直接归档。

八个小节：概览 KPI → 模型比较 → 题目参数 → 拟合指标 → 被试诊断 → 综合图 →
绝对拟合与局部依赖 → 口径说明。表格带条件着色（区分度 < 0.3 标红、拟合 p < .05 标红），
粘性表头 + 打印样式 + 窄屏适配。

```python
from cdcat import run_cdm_analysis, build_cdm_report

analysis = run_cdm_analysis(q_matrix, responses, models=["DINA", "G-DINA"])
build_cdm_report(analysis, responses, title="高一(3)班 · 诊断报告").save("report.html")
```

实测：48 题 / 500 人的报告 **316 KB**（含 2 张内联图），`http://`、`https://`、
`<script`、`<link` 计数全为 0。

---

## 一页式质量看板与明暗主题

「质量看板」页一次「体检」把两类问题分开呈现：

- **自适应施测线** —— 用 CAT 引擎按当前题库实测一遍：PMR / AMR（带 95% CI）、
  平均用题量、题库利用率。回答"这套题库 + 这套选题算法到底有多准"。
- **全题库标定线** —— 用完整作答矩阵做 MMLE/EM：题目参数、绝对拟合、Q3 局部依赖、
  PPMC（可选）。回答"题目与模型本身站不站得住"。

两条线共用同一真值模型与随机种子，数字之间可以互相印证。界面元素：

| 元素 | 说明 |
|---|---|
| **四张半环仪表** | PMR / AMR / 题目拟合合格率 / 平均分类信度，带目标刻度线（`go.Indicator` 的 `threshold`） |
| **知识地图** | 属性层级 DAG，节点内是掌握率、节点下方是属性名，颜色从玫红到青绿 |
| **逐题体检表** | slip / guess / 区分度 / 拟合 p / \|Q3\|max 六列条件着色 + 综合判定徽章 |
| **结论条** | 按指标自动判定"体检通过 / 需要关注 / 建议整改"，并列出具体问题 |
| **一键报告** | 生成并下载自包含 HTML |

### 明暗主题怎么切

```bash
streamlit run demo/app.py                    # 浅色（默认）
streamlit run demo/app.py --theme.base dark  # 深色
```

应用的起手式是 `_detect_dark()`：

1. 先读 `st.get_option("theme.base")` —— 这是**应用侧**的权威值，命令行
   `--theme.base dark` 与 `config.toml` 都反映在这里（传参时返回 `"dark"`，
   不传时返回 `None`，即默认浅色）；
2. 取不到再退回 `st.context.theme`。

> 这里有个坑值得记一笔：`st.context.theme` 读的是**浏览器**上报的主题，与应用
> 配置无关。实测无论是否传 `--theme.base dark`，它都返回 `{'type': 'light'}`。
> 早先版本只看它，结果就是"页面还是白底、下拉框却变黑了"的割裂。改成先读
> `get_option` 才对齐。

读到 `dark` 就注入**深色令牌**并把 Plotly 模板切到 `cdc_dark`。两套令牌是
**同一组 CSS 变量名**，只换取值，所以样式表里没有一处 `if dark` 分支；
图表同理，20 张图共用一套画法。

> 为什么不做页内开关？Streamlit 1.66 没有提供运行时设置主题的 API，
> 主菜单里也已经没有 Settings → Theme 这一项。而只改 CSS 会让 `st.dataframe`
> （canvas 渲染）停留在浅色，反而更割裂。启动参数配合自动检测，
> 是这套技术栈下**唯一能保证全站一致**的做法。

表格的条件着色没用 `Styler`，而是纯 HTML + **半透明叠加色**：
`rgba(13,148,136,.30)` 叠在浅色卡片上是淡青，叠在深色卡片上是墨绿，
一套代码两种主题都可读。

---

## 实测性能

以下为内置演示题库（60 题 / 5 属性 / 32 种模式，slip 与 guess 均匀分布于 0.05–0.25）的实测结果。
末尾各节（属性先验、绝对拟合、并行仿真、属性层级、多级评分）为说明各自机制使用了专门设计的较小题库，
配置已在小节标题或正文中标明；同口径的实测数据也放在对应的方法章节内。

### 选题准则对比（固定 20 题，200 被试 × 3 次重复）

| 准则 | 参数依赖 | PMR | AMR | 题库利用率 | 曝光率标准差 |
|---|---|---|---|---|---|
| KL | slip/guess | 0.7567 | 0.9370 | 1.000 | 0.2399 |
| **PWKL** | slip/guess | **0.9650** | **0.9923** | 1.000 | 0.3081 |
| MPWKL | slip/guess | 0.9550 | 0.9893 | 1.000 | 0.3032 |
| GDI | slip/guess | 0.9533 | 0.9893 | 1.000 | 0.2957 |
| CDI | slip/guess | 0.9533 | 0.9893 | 1.000 | 0.3008 |
| SHE | slip/guess | 0.9517 | 0.9880 | 1.000 | 0.3017 |
| MIV | **无（只读 Q 矩阵）** | 0.9483 | 0.9877 | 1.000 | 0.2881 |
| AMI | **无** | 0.9483 | 0.9887 | 1.000 | 0.3183 |
| NPSHE | **无** | 0.9383 | 0.9850 | 1.000 | 0.2893 |
| ERHD | **无** | 0.9283 | 0.9843 | **0.950** | **0.3660** |

> PWKL 在精度上最优；KL 因为缺少"相对于当前估计"的参照分布，在本题库下明显落后。
>
> **四个非参数准则的 PMR 落在 0.928~0.948**，比 PWKL 低 1.7~3.7 个百分点。
> 注意它们之间的排序与"直觉"不一致：直接优化属性误判个数的 **ERHD 反而最差**
> （0.9283），而它的题库利用率只有 0.950、曝光标准差最高（0.3660）——ERHD 会把
> 题量压在少数几道"能立刻缩小模式集"的题上，重复出同一批题，边际收益很快耗尽。
> 这提醒一件事：**准则的目标函数与最终指标同口径，不等于它一定选得更好**。
>
> AMI 与 MIV 的 PMR 恰好相同（都是 0.9483），但两者选的题并不一样——AMI 的曝光
> 标准差更大（0.3183 对 0.2881）。PMR 相同只是 600 次施测下的巧合，别据此说两者等价。
>
> 这张表能支撑的结论比看上去少：每个准则只有 200×3 = 600 次施测，除 KL 外的九个
> 准则**置信区间两两重叠**（ERHD [0.908, 0.949] 对 PWKL [0.950, 0.980] 也只差
> 一个千分位）。所以严格讲不是"PWKL 显著优于非参数准则"，而是**在 600 次施测的
> 分辨力下，看不出估计题目参数带来了可检出的精度增益**。按正态近似，要区分这
> 2~3 个百分点的差距需要把施测量翻两倍以上。
> 表内数字可用 `python demo/run_demo.py` 的"步骤 4"复现（同一份标定参数与种子，逐位一致）。

### 自适应终止（模式级置信 0.8，300 被试 × 5 次重复）

| 指标 | 数值 |
|---|---|
| 模式判准率 PMR | **0.8293** |
| 属性判准率 AMR | **0.9619** |
| 平均用题量 | **9.32** 题（题库共 60 题） |
| 题库利用率 | 0.9833 |

> 平均只用 **9.3 题**就从 60 题题库中完成了诊断，AMR 达 96.2%。

### 判准率随题量收敛

| 题量 | 1 | 3 | 5 | 8 | 10 | 15 | 20 | 25 |
|---|---|---|---|---|---|---|---|---|
| PMR | 0.080 | 0.160 | 0.533 | 0.767 | 0.847 | 0.947 | 0.940 | 0.960 |
| AMR | 0.585 | 0.743 | 0.876 | 0.945 | 0.960 | 0.987 | 0.985 | 0.992 |

> 约 10 题即可达到 85% 模式判准、96% 属性判准。

### EM 参数标定恢复精度（1500 名考生，DINA，60 题 / 5 属性）

| 参数 | 相关系数 | 平均绝对误差 |
|---|---|---|
| slip | 0.9449 | 0.0139 |
| guess | 0.9748 | 0.0102 |

> 8 次迭代即收敛，参数恢复达论文级精度。

### 经验贝叶斯属性先验 π(α) 的收益（30 题 / 4 属性 / 1500 考生，属性分布非均匀）

真实属性分布由 Dirichlet 采样得到（最大模式占比 0.342，高度倾斜）：

| 先验设定 | 迭代 | 自由参数 | slip MAE | guess MAE | PMR | AMR |
|---|---|---|---|---|---|---|
| 固定均匀先验 | 9 | 60（60 + 0） | 0.0155 | 0.0111 | 0.9180 | 0.9775 |
| **估计先验 π(α)** | 50 | 75（60 + 15） | **0.0147** | **0.0071** | **0.9860** | **0.9948** |

> 估计出的 π(α) 与真实分布相关系数 **0.9993**（L1 距离 0.0411）。
> 代价是迭代次数从 9 增至 50、参数多 15 个；换来 **PMR 0.918 → 0.986（+6.8 个百分点）**、
> guess 恢复误差降低 36%。当总体属性分布明显不均匀时，收益远大于代价；
> 若总体本就均匀，两种设定结果几乎一致。

### 绝对拟合与 Q3 对模型误设的判别力（真实模型 = DINO，2000 考生，30 题）

| 拟合模型 | 不合格题目 | 最大残差均值 | Q3>0.2 对数 | 最大 Q3 |
|---|---|---|---|---|
| **DINO**（正确） | **3 / 30（10%）** | **0.1225** | **0** | 0.0664 |
| DINA（误设） | **21 / 30（70%）** | **0.2667** | **32** | 0.4966 |

> 换一个错误模型，不合格率从 10% 涨到 70%，Q3 超标对从 0 涨到 32。
> 绝对拟合因此可以作为建模前的"体检"，在 AIC/BIC 差异不明显时给出更直接的模型-数据失配证据。

### n_jobs 并行仿真：加速比与一致性（300 考生 × n 次重复，40 题题库，16 核）

| 重复次数 | 串行耗时 | 并行耗时（`n_jobs=8`） | 加速比 | 与串行逐条一致 |
|---|---|---|---|---|
| 8 | 2.11 s | 0.52 s | 4.07× | ✅ |
| 16 | 4.14 s | 0.99 s | 4.17× | ✅ |
| 32 | 8.10 s | 1.94 s | 4.17× | ✅ |
| 64 | 16.12 s | 3.95 s | 4.08× | ✅ |

`n_jobs` 生效性（重复 32 次）：

| `n_jobs` | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 耗时 | 8.13 s | 4.31 s | 2.78 s | **1.99 s** |
| 加速比 | 1.00× | 1.89× | 2.93× | **4.09×** |

> 串行与并行**走同一条代码路径**（模块级 `_run_replication`），
> 每个重复由独立种子驱动并 `deepcopy` 隔离曝光计数等状态，
> 因此并行结果与串行**逐条完全一致**（已在上表多次验证）。重复数越多、加速越稳定。


### 多模型拟合比较（真实模型 = DINA，60 题 / 5 属性 / 1500 名考生）

| 模型 | 参数数 | −2LL | AIC | BIC | AICc | PMR | AMR |
|---|---|---|---|---|---|---|---|
| **DINA** | 151 | 82911.71 | **83213.71** | **84016.01** | **83247.76** | **0.9607** | **0.9917** |
| DINO | 151 | 99338.48 | 99640.48 | 100442.78 | 99674.54 | 0.2847 | 0.8413 |
| G-DINA | 217 | 82833.84 | 83267.84 | 84420.81 | 83341.64 | 0.9607 | 0.9917 |
| RRUM | 184 | 85600.32 | 85968.32 | 86945.96 | 86020.10 | 0.9187 | 0.9820 |
| ACDM | 151 | 97432.92 | 97734.92 | 98537.21 | 97768.97 | 0.6940 | 0.9276 |
| LLM | 151 | 86830.04 | 87132.04 | 87934.33 | 87166.09 | 0.8873 | 0.9751 |

> 参数数已计入**经验贝叶斯先验**的 31 个自由参数（$2^5-1$）：DINA 的 120 个题目参数
> + 31 个先验参数 = 151。旧版把先验当固定值、只数题目参数，AIC/BIC 会系统性偏乐观。

> **模型选择完全正确**：三个信息准则一致选出 DINA。G-DINA 的 −2LL 略优（82833.84
> vs 82911.71），但多出的 66 个参数不足以抵消惩罚；LRT 给出 $LR=77.87$、$df=66$、
> $p=0.151$（不显著），同样支持"DINA 已足够"的结论。DINO 作为误设模型，
> PMR 从 0.961 跌到 0.285 —— 符合预期。

**新加的两个加性模型在这份数据上明显吃亏**，这是模型错配而非标定失败：

| 诊断量 | ACDM | LLM |
|---|---|---|
| 拟合 P 与 DINA 真值 P 的平均绝对差 | 0.1090 | 0.0536 |
| 与"把 DINA 概率直接投影过来"的初值相比，logL 提升 | +490.1 | +671.2 |
| ΔBIC（相对 DINA） | +14521 | +3918 |

合取结构要求"全部掌握才跳变"，加性模型只能给出平滑的递增，所以无论怎么估都
拟合不上：即使把 DINA 的 P 直接投影到 ACDM 结构，两者本身就差约 0.10。
关键在于 EM 的最终解比这个投影初值还好（logL 高 490 / 671），说明**优化已经
到位，差的是模型结构本身**——这也正是"模型比较"应该暴露的东西。
要判断"模型本身好不好"而不是"这份数据适合谁"，得看自生成数据：在 ACDM 自己
生成的数据上，P 矩阵恢复 MAE 仅 0.013、PMR 0.90（见 `tests/test_new_models.py`
与 `test_cdm_analysis.py`）。

> 顺带一个方法论提醒：`run_cdm_analysis` 只对**已知嵌套**的组合给出 LRT 的 p 值——
> 五个受限模型（DINA / DINO / RRUM / ACDM / LLM）都嵌套于饱和的 G-DINA
> （每题 $2^{r_j}$ 个自由概率，能表示任何题目反应函数），
> 而它们彼此之间（如 ACDM 与 DINA）既非嵌套也无参数包含关系，
> 会附注"非嵌套，请以 AIC/BIC 为准"。强行套 χ² 会给出看似严谨、实则无意义的 p 值。

### 各模型的参数恢复精度

| 模型 | 参数 | 相关系数 | MAE | RMSE |
|---|---|---|---|---|
| DINA | slip | 0.9461 | 0.0144 | 0.0190 |
| DINA | guess | 0.9784 | 0.0093 | 0.0116 |
| G-DINA | slip | 0.9461 | 0.0144 | 0.0190 |
| G-DINA | guess | 0.9785 | 0.0094 | 0.0116 |
| RRUM | slip | 0.9305 | 0.0166 | 0.0221 |
| RRUM | guess | 0.9094 | 0.0251 | 0.0312 |
| ACDM | slip | 0.6466 | 0.0718 | 0.0938 |
| ACDM | guess | 0.3390 | 0.0720 | 0.1002 |
| LLM | slip | 0.7173 | 0.0533 | 0.0717 |
| LLM | guess | 0.9431 | 0.0168 | 0.0219 |
| DINO | slip | 0.0725 | 0.1829 | 0.2462 |

> G-DINA 在 DINA 数据上退化为 DINA（`from_dina` 给出良好初值），恢复精度与 DINA 持平；
> RRUM 因模型误设而略低；DINO 的 slip 几乎无法恢复 —— 正是模型选择应当排除它的原因。
>
> ACDM / LLM 那两行的数字要谨慎读：它们本身没有 slip/guess，表里是**等价 DINA 参数**
> （由全掌握组与非全掌握组的概率均值反推），与真值不同口径；系数低主要反映模型误设，
> 而不是估计不准。判断估计准不准应看概率矩阵：同一份数据上 ACDM 的 P 恢复 MAE 是
> 0.013（自生成数据）／0.032（相对投影初值）。

### 拟合指标数值自检

`likelihood_ratio_test` 的 χ² 上尾概率与 `scipy.stats.chi2` 在
$k\in\{1,2,3,4,5,7,10,20,50\}$、上尾 $\in\{0.001,\dots,0.99\}$ 的全部 63 组组合上
最大绝对误差 $2.2\times10^{-15}$，且不依赖 scipy。

### 曝光控制效果（40 题题库，300 被试 × 5 次重复）

| 策略 | 最高曝光率 | 曝光率 SD | PMR 变化 | 平均题量变化 |
|---|---|---|---|---|
| 无控制 | 100% | 0.237 | 基准 | 基准 |
| 随机化选题（top 5） | 87% | 0.181 | -0.008 | +0.6 |
| 曝光硬上限 0.2 | 80% | 0.137 | -0.012 | +3.1 |
| Sympson-Hetter | **53%** | **0.089** | -0.021 | +3.4 |

> Sympson-Hetter 将最高曝光率从 100% 压到 53%、曝光标准差降低 62%，
> 代价仅为约 3.4 道额外题目 —— 在题库安全性与诊断效率间取得最佳平衡。

### 结构可辨识性验证

用一个刻意设计的缺陷题库（40 题 / 4 属性，`QMatrix.from_random(40, 4, seed=1)`）验证：

| 条件 | PMR |
|---|---|
| 完美确定性作答（slip=guess=0） | 0.9400 |
| 结构分析预测的理论上限 | 0.9375 |
| 误差 | **0.0025** |

> 在 `slip=guess=0` 下 PMR 仍达不到 1.0 —— 这不是算法缺陷，
> 而是该 Q 矩阵中模式 `0000` 与 `0010` 产生**完全相同的理想反应向量**，
> 数学上无法区分。结构分析预测的上限与实际观测仅差 0.0025，验证了分析的准确性。

---

## API 速查

### 核心类

```python
QMatrix(matrix, attr_names=None, item_names=None)
  .from_random(n_items, n_attrs, seed=...)
  .from_csv(path)
  .ideal_response(mode='conjunctive')   # -> (2^K, J)
  .validate()                            # -> 问题列表
  .diagnose()                            # -> QMatrixDiagnostics
  .required_attrs(item_idx)
  .attribute_exposure()

DINA(q, slip=None, guess=None)
DINO(q, slip=None, guess=None)
GDINA(q, beta=None, gamma=None)
  .from_dina(q, slip, guess)             # 精确等价转换
RRUM(q, pi_star=None, r_star=None)       # r_star 为列表，每题一个向量

EMCalibrator(q, model_type='DINA', max_iter=500, tol=1e-6,
             prior=None, estimate_prior=None, prior_smoothing=1e-6)
  #  model_type: 'DINA' | 'DINO' | 'G-DINA' | 'RRUM'（别名大小写/下划线不敏感）
  #  estimate_prior: None=自动（未传 prior 则估计）、False=固定均匀、True=强制估计
  #  prior: 自定义属性模式先验 (2^K,)，传入即固定
  .fit(responses)                        # -> CalibrationResult
  .calibrate_to_model(responses)         # -> (model, CalibrationResult)
  .examinee_patterns(responses, slip, guess)
  .item_discrimination(slip, guess)
  .n_params_per_model .n_prior_params .n_free_params   # 参数计数（property）

CalibrationResult
  .slip .guess .beta .gamma .pi_star .r_star
  .log_likelihood .log_likelihood_history .deviance
  .n_params .n_item_params .n_prior_params .attr_prior .prior_estimated
  .n_iterations .converged .model_name
  .aic .bic .aicc
  .to_model(q)                           # -> 已标定的响应模型实例
  .summary()                             # 含参数构成与 π(α) 统计

calibrate(q, responses, model_type='DINA', **kw)   # 便捷函数 -> (model, result)

# ---------- 绝对拟合 / 局部依赖 ----------
analyse_absolute_fit(model, responses, alpha_hat=None, prior=None,
                     n_bins=5, min_expected=1.0, item_names=None)
  # -> AbsoluteFitResult (.item_fit .local_dependence .observations .items .summary())

item_fit(model, responses, alpha_hat=None, prior=None,
         n_bins=5, min_expected=1.0, item_names=None)   # -> ItemFitResult
ItemFitResult
  .chi_square .g_square .df .p_value .p_value_g .n_cells
  .max_abs_residual .rmse_residual .item_names .n_bins
  .to_dataframe() .poorly_fitting(alpha=0.05) .worst_items(top_n=5) .summary()

local_dependence(model, responses, alpha_hat=None, prior=None,
                 threshold=0.2, item_names=None)        # -> LocalDependenceResult
LocalDependenceResult
  .q3 .threshold .mean_q3 .max_q3 .n_flagged .pairs .item_names
  .top_pairs(n=10) .flagged_pairs(top_n=None) .to_dataframe() .summary()

# ---------- 拟合指标 / 诊断 / 模型比较 ----------
compute_fit_indices(responses, log_likelihood, model, model_name='',
                    n_extra_params=0)    # n_extra_params: 题目参数之外的自由参数（如 2^K-1 先验）
saturated_log_likelihood(responses)

FitIndices   # .log_likelihood .n_params .n_observations .n_examinees .n_items
             # .aic .bic .aicc .deviance .saturated_ll .saturated_deviance
             # .cfi .tli .rmsea .model_name .pmr .amr     (pmr/amr 可选)
  .to_dict() .summary()

diagnose_patterns(model, responses, prior=None, method='MAP', attr_names=None)
  # -> PatternDiagnosis
PatternDiagnosis
  .posterior .map_patterns .eap_marginal .eap_patterns
  .map_confidence .attr_reliability .pattern_indices .attr_names
  .n_examinees .n_attrs
  .mastered_counts() .mastery_proportion() .low_reliability_attrs(threshold=0.6)
  .mean_confidence() .pattern_distribution() .summary() .to_dataframe()

compare_models({'DINA': fit1, 'G-DINA': fit2})     # -> ModelComparison
ModelComparison  # .rows .best_aic .best_bic .best_aicc .summary() .to_dataframe()

likelihood_ratio_test(ll_restricted, n_params_restricted,
                      ll_full, n_params_full)
  # -> {'lr_statistic','df','p_value','significant'}，χ² 自实现，无需 scipy

parameter_recovery(estimated, true)     # -> {'correlation','mae','rmse','bias'}

# ---------- 端到端流水线 ----------
run_cdm_analysis(q_matrix, responses, models=('DINA',), prior=None,
                 estimate_prior=None, absolute_fit=True, n_bins=5, q3_threshold=0.2,
                 max_iter=500, tol=1e-6, attr_names=None,
                 true_slip=None, true_guess=None, true_patterns=None,
                 verbose=False)         # -> CDMAnalysis
CDMAnalysis
  .q_matrix .models .fits .comparison .diagnosis .best_model .lrt .recovery
  .absolute_fit                         # -> AbsoluteFitResult（最优模型）
  .model(name)                          # -> 已标定的响应模型实例
  .fit_table()                          # 拟合指标 DataFrame（含 n_item_params/n_prior_params/pmr/amr）
  .param_table()                        # 题目参数表
  .diagnosis_table(model_name=None)     # 被试诊断明细
  .summary()                            # 含绝对拟合与局部依赖摘要
  .to_excel(path)                       # sheet：拟合/参数/诊断/绝对拟合/Q3/属性先验

AttributeEstimator(model, prior=None, method='MAP')
  .estimate(responses, item_indices)     # -> EstimatorResult
  .posterior(responses, item_indices)
  .marginal_probabilities(posterior)
  .estimate_sequential(responses, item_indices)

# ---------- PPMC / 层级比较 / 装配 / 在线标定 / 报告 ----------
posterior_predictive_check(model, responses, *, prior=None, n_replicates=500,
                           seed=2024, alpha=0.05, refit=True,
                           max_items_pair=150)      # -> PPMCResult
PPMCResult
  .observed .replicated .ppp .stat_names .misfit_stats .rmsea_ppp
  .alpha_source .used_refit
  .to_dataframe() .to_dict() .verdict()

compare_hierarchies(q_matrix, responses, model_type='DINA', *,
                    structures=('none','linear','convergent','divergent'))
  # -> HierarchyComparisonReport
HierarchyComparisonReport
  .candidates                            # list[HierarchyCandidate]
  .ranking .best .best_label
  .get(label) .verdict() .to_dataframe() .to_dict()

assemble_test_bank(q_matrix, model=None, n_items=None, *,
                   must_include=None, min_per_attr=1, max_per_attr=None,
                   prior=None, local_search=True)   # -> AssemblyResult
AssemblyResult
  .selected .selected_names .objective .score .expected_pmr .expected_amr
  .attr_coverage .n_swaps .must_include
  .coverage_ok() .to_dataframe() .to_dict() .summary()

calibrate_new_items(model, responses, new_items, *, old_items=None, prior=None,
                    min_effective_n=10.0, ci_level=0.95)  # -> OnlineCalibrationReport
OnlineCalibrationReport
  .items .usable_items .flagged
  .get(idx) .verdict() .to_dataframe() .to_dict() .summary()
NewItemResult
  .slip .guess .slip_se .guess_se .slip_ci .guess_ci
  .n_masters .n_nonmasters .discrimination .status .usable

build_cdm_report(analysis, responses=None, *, title='认知诊断分析报告',
                 subtitle=None, author=None, include_figures=True)  # -> HTMLReport
HTMLReport
  .heading() .paragraph() .kpis() .callout() .table() .figure() .raw()
  .render()                              # -> str
  .save(path)                            # 写盘并返回路径

# ---------- 可视化 ----------
from cdcat.calibration_plots import (          # matplotlib，静态报告
    calibration_report_figure,                 # 6 面板标定报告
    absolute_fit_report_figure,                # 3 面板绝对拟合报告
    plot_ll_convergence, plot_parameter_recovery, plot_item_discrimination,
    plot_fit_comparison, plot_mastery_proportion, plot_attr_reliability,
    plot_pattern_heatmap,
    plot_item_fit, plot_q3_heatmap, plot_attr_prior,     # 绝对拟合 / Q3 / 属性先验
)
from demo.cdm_charts import (                  # plotly，交互式
    fig_ll_convergence, fig_param_recovery, fig_item_discrimination,
    fig_fit_comparison, fig_mastery_proportion, fig_attr_reliability,
    fig_pattern_distribution,
    fig_item_fit, fig_q3_heatmap, fig_attr_prior,
)

CDCATSimulator(q, model='DINA', model_params={...}, selector='PWKL', ...)
  .administer(true_pattern, rng=...)           # 单被试
  .simulate_fixed_length(true_pattern, n_items)# 固定题量
  .run(n_examinees, n_replications, seed=...,
       n_jobs=1)                               # 蒙特卡洛；n_jobs>1 走多进程
  .compare_criteria(['PWKL','GDI'], fixed_length=20)
  .convergence_curve(n_examinees, max_items)

Evaluator(confidence=0.95)
  .evaluate(result, label)
  .compare(results)                      # -> DataFrame
  .exposure_chi_square(result)
  .efficiency_index(result)
  .full_report(results)
```

### 选题准则

```python
from cdcat import KL, PWKL, MPWKL, GDI, SHE, CDI, MIV, NPSHE, ERHD, AMI, create_selector

sel = create_selector('PWKL', model)
sel.compute(posterior, available_items)   # 所有候选题的准则值
sel.select(posterior, exclude=answered)   # 选出最优题
sel.rank(posterior, top_n=10)             # 前 10 名

# 四个非参数准则：不读 slip/guess，标定不可用时也能选题
sel = create_selector('MIV', model)       # 理想反应方差
sel = create_selector('NPSHE', model)     # 模式级熵缩减（SHE 的无参数版）
sel = create_selector('ERHD', model)      # 期望汉明距离缩减
sel = create_selector('AMI', model)       # 属性边际熵缩减

# 理想反应默认跟随模型（DINO 用析取），也可显式指定
sel = ERHD(model, ideal_mode='disjunctive')
```

### 非参数分类

```python
from cdcat import classify_nonparametric, NONPARAMETRIC_METHODS
from cdcat.estimation.nonparametric import distance_matrix, item_weights

res = classify_nonparametric(
    q, responses,                # (n, J)，允许 NaN 表示未作答
    method='gdd',                # 'gdd'（方差加权）/ 'hamming'（等权）
    hierarchy=None,              # 给 AttributeHierarchy 则只在合法模式中搜索
    temperature=1.0,             # 伪后验温度，单位为"等效题数"
    attr_names=q.attr_names,
)
res.patterns                     # (n, K) 判别出的属性模式
res.confidence                   # (n,) 判别把握（伪后验最大值）
res.distance_gap()               # (n,) 最优与次优的距离差，接近 0 说明不可信
res.accuracy(true_patterns)      # {'pmr': ..., 'amr': ..., 'per_attr': [...]}
res.to_pattern_diagnosis()       # 转成 PatternDiagnosis，复用全部诊断图表

distance_matrix(q, responses, method='gdd')   # (n, 2^K) 加权距离
item_weights(responses, 'gdd')                # (J,) 题目权重，均值归一
```

### 终止规则

```python
from cdcat import (
    PosteriorThreshold, FixedLength, StabilityRule,
    AttributeConfidence, PatternConfidence, CompositeRule,
)

PosteriorThreshold(threshold=0.8, mode='all_attrs')
FixedLength(n_items=20)
StabilityRule(n_stable=5)
AttributeConfidence(max_entropy=0.3)
PatternConfidence(threshold=0.8)          # 推荐默认
CompositeRule([...], how='any', min_items=5, max_items=40)
```

### 曝光控制

```python
from cdcat import (
    NoExposureControl, MaxExposureLimit, RandomesqueExposure,
    StratifiedExposure, SympsonHetter, create_exposure_control,
)

MaxExposureLimit(n_items, max_rate=0.2, warmup=18)
RandomesqueExposure(n_items, top_n=5)
StratifiedExposure(n_items, penalty=0.5, warmup=18)
SympsonHetter(n_items, target_rate=0.2).calibrate(sim, ...)
```

### 题库诊断与分析

```python
from cdcat import analyse_identifiability, analyse_items, create_model
from cdcat import build_examinee_report

# 结构可辨识性
rep = analyse_identifiability(q, ideal_mode='conjunctive')
rep.expected_pmr_ceiling        # PMR 理论上限
rep.is_complete                 # 是否严格完备
rep.equivalence_classes         # 不可区分的模式等价组
rep.uncovered_attrs             # 无题考查的属性
rep.interchangeable_attr_pairs  # 可互换属性对
rep.summary()                   # 文本报告

# 题目分析
ia = analyse_items(model, n_simulate=4000, seed=2024)
ia.discrimination               # 每题区分度
ia.information                  # 每题信息量
ia.structural_duplicates        # 结构冗余题组（主判据）
ia.strongest_items / ia.weakest_items

# 个人报告
report = build_examinee_report(rec, q.attr_names, q.item_names, q)
report.to_text() / report.to_dict()
report.mastered / report.accuracy / report.correct_count   # property
report.low_confidence_attrs(threshold=0.6)

# 结果 vs 上限
result.pmr_vs_ceiling()
# {'pmr', 'ceiling', 'gap', 'efficiency', 'n_unidentifiable', 'verdict'}
```

### Q 矩阵

```python
q = QMatrix.from_random(J, K, seed=..., attr_names=[...])
q = QMatrix.from_csv(path) / QMatrix(matrix, attr_names=...)

q.matrix                  # (J, K) 0/1 矩阵
q.attribute_patterns()    # (2^K, K) 全部掌握模式
q.ideal_response()        # (2^K, J) 理想反应矩阵
q.required_attrs(j)       # 题 j 考查的属性**索引**（注意：返回索引）
q.attr_names              # 属性名列表
q.diagnose()              # QMatrixDiagnostics
q.analyse_identifiability()
q.validate()
```

### 数据 I/O（`cdcat.io`，纯 Python，不依赖 streamlit）

```python
from cdcat.io import (
    read_q_matrix, write_q_matrix,
    read_responses, write_responses,
    read_params, write_params,
    read_bank, write_bank,
    write_table, read_delimited,
)

q = read_q_matrix("q.csv")                        # 自动识别题号列 / 属性名列
X, ids = read_responses("X.csv")                  # 缺失值 -> np.nan（不插补）
name, params = read_params("params.json")
bank = read_bank("bank.json")                     # 与 Web 界面 JSON 格式互通
```

> 分隔符自动嗅探（`,` `;` `\t` `|`），表头自动判定。
> 缺失标记集合：`""` / `na` / `n/a` / `nan` / `null` / `none` / `.` / `-` / `?`
> （大小写不敏感）。Q 矩阵若含缺失或非 0/1 值会**直接报错**，不会静默通过。

### 属性层级 / Q 矩阵修正

```python
from cdcat import (
    AttributeHierarchy, check_hierarchy_consistency,
    validate_q_matrix,
)

AttributeHierarchy.none(k) / .linear(k, names) / .convergent(k) / .divergent(k)
AttributeHierarchy.from_edges(k, [(a, b), ...]) / .from_adjacency(matrix)
AttributeHierarchy.from_dict(d)
  .n_attrs .n_legal_patterns .edges .structure .is_empty
  .reachability() .all_prerequisites(k) .direct_prerequisites(k)
  .direct_dependents(k) .levels .legal_mask() .legal_indices()
  .legal_patterns() .is_legal(pattern) .project(pattern)
  .reduced_q_matrix(q) .to_dict() .summary()

check_hierarchy_consistency(q_matrix, hierarchy)   # -> HierarchyConsistencyReport

validate_q_matrix(q_matrix, responses, model, pvaf_threshold=0.95,
                  max_attrs_per_item=None, alpha_hat=None, prior=None,
                  missing="skip")                  # -> QValidationReport
  #  model: 已标定模型（用于推断被试属性模式）；也可直接用 alpha_hat 传入
  #  max_attrs_per_item: 候选属性数上限，默认 min(K, 4)
QValidationReport
  .threshold .attr_names .item_names .n_items
  .item_pvaf .item_best_pvaf .item_best_attrs .item_current_attrs
  .suggestions .n_suspect .n_additions .n_removals .mean_pvaf
  .n_candidates_scanned .max_attrs_per_item
  .to_dict() .to_dataframe() .summary(top_n=10) .apply(q, min_improvement=0.0)
QSuggestion
  .item_idx .item_name .current_attrs .suggested_attrs .attr_names
  .pvaf_current .pvaf_suggested
  .added .removed .improvement                 # property
  .current_attrs_str() .suggested_attrs_str() .describe() .to_dict()

# 辨识性分析支持层级
analyse_identifiability(q, ideal_mode='conjunctive', hierarchy=h)
  # 新增：.hierarchy_structure .n_legal_patterns
  #       .expected_pmr_ceiling_no_hierarchy
```

### 群体报告

```python
from cdcat import build_group_report, GroupReport
from cdcat.figures import plot_attribute_mastery, plot_group_report, plot_convergence

rep = build_group_report(
    records=None,            # 与 responses 二选一
    attr_names=[...], q_matrix=q,
    item_names=None, model=None,
    responses=None, examinee_ids=None, group_labels=None,
    prior=None, method="MAP",          # MAP | EAP
    label="群体诊断", missing="skip",   # skip | wrong | ignore
    include_truth=False,
)
GroupReport
  .attr_mastery_rate .attr_mean_prob .attr_sd_prob .mastery_matrix
  .probabilities .pattern_idx .pattern_freq .n_answered
  .group_labels .true_patterns
  .mean_mastered .n_full_mastery .n_zero_mastery .full_mastery_rate
  .weakest_attrs(n=3) .strongest_attrs(n=3) .most_common_patterns(n=5)
  .group_attr_mastery() .compare_examinee(i)
  .attr_dataframe() .pattern_dataframe(n) .group_dataframe()
  .to_dict() .to_text()
```

### 多级评分（`cdcat.polytomous`）

```python
from cdcat import (
    PolytomousQMatrix, SequentialGDINA,
    calibrate_polytomous, build_polytomous_bank,
    expand_to_steps, collapse_step_responses,
    PolyKL, PolyPWKL, PolyGDI, create_poly_selector,
)

PolytomousQMatrix.sequential(matrix, n_categories)      # 属性即步骤
PolytomousQMatrix.from_steps(step_q, item_index)        # 显式步骤表
PolytomousQMatrix.from_binary_qmatrix(q, n_categories)  # 由二分 Q 扩展
  .steps .item_index .n_items .n_steps .n_categories .n_attrs .max_categories
  .step_slice(j) .union_q_matrix()

model, res = calibrate_polytomous(pq, responses, max_iter=500, tol=1e-6,
                                  estimate_prior=True, init_model=None)
res.n_steps .n_categories .aic .bic .aicc          # CalibrationResult 扩展字段

SequentialGDINA(pq, ...)
  .step_probs()               # (2^K, S)
  .category_probs()           # (2^K, J, Cmax)
  .log_likelihood(responses)  # 多级似然
  .simulate(patterns, rng)    # 多级分数
  .item_information()

expand_to_steps(responses, pq) / collapse_step_responses(y, pq)   # 严格互逆

PolyKL(model) / PolyPWKL(model) / PolyGDI(model)
create_poly_selector("PWKL" | "KL" | "GDI", model)
  .compute(posterior, available_items) .select(posterior, exclude=[]) .rank(...)
```

### CLI

```python
from cdcat.cli import main
main(["calibrate", "--responses", "X.csv", "--q", "q.csv", "--out", "out/"])
```

```bash
cdcat presets | list | demo | calibrate | simulate | diagnose | validate-q | report
```

---

## 参考文献

1. Xu, X., Chang, H., & Douglas, J. (2003). A simulation study to compare CAT strategies for cognitive diagnosis. *NCME*.
2. Cheng, Y. (2009). When cognitive diagnosis meets computerized adaptive testing: CD-CAT. *Psychometrika*, 74(4), 619–632.
3. Kaplan, M., de la Torre, J., & Barrada, J. R. (2015). New item selection methods for cognitive diagnosis computerized adaptive testing. *Applied Psychological Measurement*, 39(3), 167–188.
4. de la Torre, J. (2011). The generalized DINA model framework. *Psychometrika*, 76(2), 179–199.
5. Junker, B. W., & Sijtsma, K. (2001). Cognitive assessment models with few assumptions. *Applied Psychological Measurement*, 25(3), 258–272.
6. Ma, W., & de la Torre, J. (2020). GDINA: An R package for cognitive diagnosis modeling. *Journal of Statistical Software*, 93(14).
7. Henson, R. A., Templin, J. L., & Willse, J. T. (2009). Defining a family of cognitive diagnosis models using log-linear models. *Psychometrika*, 74(2), 191–210.
8. Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters: Application of an EM algorithm. *Psychometrika*, 46(4), 443–459.
9. Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society: Series B*, 39(1), 1–38.
10. Yen, W. M. (1984). Effects of local item dependence on the fit and equating performance of the three-parameter logistic model. *Applied Psychological Measurement*, 8(2), 125–145.
11. Chen, J., de la Torre, J., & Zhang, Z. (2013). Relative and absolute fit evaluation in cognitive diagnosis modeling. *Journal of Educational Measurement*, 50(2), 123–140.
12. **de la Torre, J., & Chiu, C.-Y. (2016).** A general method of empirical Q-matrix validation. *Psychometrika*, 81(2), 253–273. —— 本系统 Qval（PVAF）方法的来源。
13. **Chiu, C.-Y. (2013).** Statistical refinement of the Q-matrix in cognitive diagnosis. *Applied Psychological Measurement*, 37(8), 598–618. —— PVAF 指标的定义。
14. **Ma, W., & de la Torre, J. (2016).** A sequential cognitive diagnosis model for polytomous responses. *British Journal of Mathematical and Statistical Psychology*, 69(3), 253–275. —— 顺序 G-DINA 与条件式步骤参数化。
15. **Leighton, J. P., Gierl, M. J., & Hunka, S. M. (2004).** The attribute hierarchy method for cognitive assessment: A variation on Tatsuoka's rule-space approach. *Journal of Educational Measurement*, 41(3), 205–237. —— 属性层级与合法模式。
16. **Templin, J., & Henson, R. A. (2006).** Measurement of psychological disorders using cognitive diagnosis models. *Psychological Methods*, 11(3), 287–305. —— 属性层级下 PMR 上限的讨论。
17. Leighton, J. P., & Gierl, M. J. (2007). *Cognitive diagnostic assessment for education: Theory and applications*. Cambridge University Press.

---

## 许可证

本项目仅供教学与研究使用。
