

<div align=center>
<h1 align="center">
EoH: Evolution of Heuristics 
</h1>
<h5 align="center">
进化计算 + 大语言模型 自动算法设计平台
</h5>

[English Version 英文版本](./README.md)

[![Github][Github-image]][Github-url]
[![License][License-image]][License-url]
[![Releases][Releases-image]][Releases-url]
[![Wiki][Wiki-image]][Wiki-url]


[Github-image]: https://img.shields.io/badge/github-12100E.svg?style=flat-square
[License-image]: https://img.shields.io/badge/License-MIT-orange?style=flat-square
[Releases-image]: https://img.shields.io/badge/Release-Version_1.0-blue?style=flat-square
[Installation-image]: https://img.shields.io/badge/Web_Demo-Version_1.0-blue?style=flat-square
[Wiki-image]: https://img.shields.io/badge/Docs-参考文档-black?style=flat-square


[Github-url]: https://github.com/FeiLiu36/EOH
[License-url]: https://github.com/FeiLiu36/EOH/blob/main/LICENSE
[Releases-url]: https://github.com/FeiLiu36/EOH/releases
[Wiki-url]: https://github.com/FeiLiu36/EOH/tree/main/docs



</div>
<br>

轻量级、易用的 EoH 框架，用于大语言模型驱动的自动算法/启发式设计。

> [!Note]
> **使用旧版 API？** 如果您的代码使用 `Paras` / `eoh.EVOL`，对应的是旧版 v0.1 接口。
> 请从 [**v0.1 发布页**](https://github.com/FeiLiu36/EoH/releases/tag/v0.1) 下载或安装旧版本：
> ```bash
> pip install git+https://github.com/FeiLiu36/EoH.git@v0.1#subdirectory=eoh
> ```
> 当前 `main` 分支使用下文所述的新版 `LLMConfig` / `EoH` / `BaseProblem` API。

<img src="./docs/figures/eoh.JPG" alt="eoh" width="600" height="280">

---
## 新闻 🔥

+ 2026.02 🎉🎉 **新冠军！** 我们的 LLM4AD 系统赢得了 [CVRPLib BKS 竞赛](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/bks_challenge/score/)，在大规模 CVRP 基准上创立了 **51 个新的最优已知解**。
+ 2026.01 🎉🎉 我们的综述论文 ["A Systematic Survey on Large Language Models for Algorithm Design"](https://arxiv.org/pdf/2410.14716) 已被 [**ACM Computing Surveys**](https://dl.acm.org/journal/csur) 录用！综述相关代码库见[此处](https://github.com/FeiLiu36/LLM4AlgorithmDesign)。
+ 2025.6 🎉🎉 **EoH** 近期在圆填充问题上创造了**世界纪录**，26 个圆的最优值达到 2.63594！[结果详见](https://github.com/Optima-CityU/llm4ad/tree/main/example/circle_packing)
+ 2024.5.2，[EoH (Evolution of Heuristics: Towards Efficient Automatic Algorithm Design using Large Language Model)](https://arxiv.org/abs/2401.02051) 已被 **ICML 2024（口头报告，Top 1.5%）** 录用！🎉

---

## 简介 📖

启发式算法在解决复杂的搜索与优化问题中不可或缺。然而，人工设计启发式算法既繁琐，又高度依赖人类直觉与经验。

EOH 引入了一种新范式，利用大语言模型（LLM）与进化计算（EC）之间的协同作用进行自动启发式设计（AHD）。在进化框架内对思路与代码进行协同演化，可在降低计算开销的同时，获得优异的 AHD 性能。

<img src="./docs/figures/framework.jpg" alt="eoh" width="500" height="auto" div align=center>

EOH 可在数分钟至数小时内自动设计出极具竞争力的算法/启发式方法。以在线装箱问题为例，EoH 自动设计出的新启发式算法优于人工设计算法，并以显著更少的 LLM 调用次数超越了 FunSearch。

下图展示了 EOH 在在线装箱问题上的进化过程。图中标注了进化过程中对最优结果贡献最大的关键**思路**及对应**代码片段**，同时标记了带来改进的提示策略，最终展示最优种群中的启发式方法，并与人工设计算法和 FunSearch 进行比较。

<img src="./docs/figures/evolution.jpg" alt="eoh" width="1000" height="auto">

如果您发现 EoH 对您的研究或应用项目有所帮助：

```bibtex
@inproceedings{fei2024eoh,
    title={Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model},
    author={Fei Liu, Xialiang Tong, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu, Qingfu Zhang},
    booktitle={International Conference on Machine Learning (ICML)},
    year={2024},
    url={https://arxiv.org/abs/2401.02051}
}
```

如果您对 LLM4Opt 或 EoH 感兴趣，欢迎：

1) 通过邮件 fliu36-c@my.cityu.edu.hk 与我们联系。
2) 访问[大模型与优化参考文献和研究论文收藏](https://github.com/FeiLiu36/LLM4Opt)。
3) 加入我们的讨论组（即将推出）。

如果您在使用代码时遇到任何困难，请通过上述方式与我们联系或提交 [issue](https://github.com/FeiLiu36/EoH/issues)。



## 系统要求

- python >= 3.10
- numpy
- joblib



## EoH 示例用法 💻

#### 第 1 步：安装 EoH

建议在 Python >= 3.10 的 [conda](https://conda.io/projects/conda/en/latest/index.html) 环境中安装和运行 EoH。

```bash
cd eoh
pip install .
```

#### 第 2 步：配置 LLM 并运行示例

**<span style="color: red;">启动前请先配置远程 LLM 的端点和密钥，或配置本地 LLM！</span>**

```python
from eoh import EoH, LLMConfig
from prob import MyProblem  # 您的问题类（参见下方"在您的应用中使用 EoH"）

llm = LLMConfig(
    api_endpoint="api.deepseek.com",  # 例如 "api.openai.com" 或 "api.deepseek.com"
    api_key="your-api-key",
    model="deepseek-chat",            # 例如 "gpt-4o"、"deepseek-chat"
    timeout=150,
)

task = MyProblem(timeout=40, n_processes=4)

eoh = EoH(
    llm=llm,
    problem=task,
    pop_size=5,       # 每代种群大小
    n_pop=20,         # 进化代数
    operators=['e1', 'e2', 'm1', 'm2'],
    output_dir="./results",
)
eoh.run()
```

###### 示例 1：旅行商问题构造算法

```bash
cd examples/tsp_construct
python runEoH.py
```
**评估**
```bash
cd examples/tsp_construct/evaluation
# 将您的启发式方法复制到 heuristic.py（函数名/输入/输出须与评估模块对齐）
python runEval.py
```

###### 示例 2：在线装箱问题
（**<span style="color: red;">在您的个人计算机上 30 分钟内生成新的最优启发式方法！</span>** i7-10700 2.9GHz，32 GB）

```bash
cd examples/bp_online
python runEoH.py
```
**评估**
```bash
cd examples/bp_online/evaluation
# 将您的启发式方法复制到 heuristic.py（函数名/输入/输出须与评估模块对齐）
python runEval.py
```



## 在您的应用中使用 EoH

通过继承 `BaseProblem` 定义您的问题，需要提供以下三项：
- `template_program`：LLM 将进化的目标函数（或类）的 Python 代码骨架。
- `task_description`：一句话描述 LLM 需要优化的目标。
- `evaluate_program`：接收生成的可调用对象并返回浮点适应度值（越小越好）的方法，失败时返回 `None`。

```python
import numpy as np
from eoh import EoH, LLMConfig, BaseProblem


class MyProblem(BaseProblem):
    template_program = '''
def heuristic(x: np.ndarray) -> float:
    """Compute a score for input x."""
    return float(x.mean())
'''
    task_description = "Design a heuristic function that minimises the mean of the input."

    def evaluate_program(self, program_str: str, callable_func) -> float | None:
        # 替换为您的实际评估逻辑
        score = callable_func(np.random.rand(10))
        return score  # 越小越好


llm = LLMConfig(
    api_endpoint="api.deepseek.com",
    api_key="your-api-key",
    model="deepseek-chat",
)

task = MyProblem(timeout=30, n_processes=4)

eoh = EoH(llm=llm, problem=task, pop_size=5, n_pop=20, output_dir="./results")
eoh.run()
```

EoH 支持三种模板类型，自动识别：
- **`function`**（函数）— 单个函数，最常用。
- **`multi_function`**（多函数）— 多个协作函数，最后一个为入口点。
- **`class`**（类）— 含指定方法的类，类名为入口点。

类模板和多函数模板示例分别见 [`examples/tsp_construct_class`](https://github.com/FeiLiu36/EOH/tree/main/examples/tsp_construct_class) 和 [`examples/tsp_construct_multifunction`](https://github.com/FeiLiu36/EOH/tree/main/examples/tsp_construct_multifunction)。



## 三种层级的启发式设计

EoH 支持三种层级的启发式设计，表达能力逐级增强。以下以 TSP 构造启发式任务为例分别说明。

---

#### 第一层级 — 单函数（[`tsp_construct`](https://github.com/FeiLiu36/EOH/tree/main/examples/tsp_construct)）

最简单也最常用的形式。LLM 进化一个独立函数，该函数本身即为入口点。

```python
template_program = '''
def select_next_node(current_node: int, destination_node: int,
                     unvisited_nodes: np.ndarray,
                     distance_matrix: np.ndarray) -> int:
    """在 TSP 贪婪构造中选择下一个访问节点。"""
    return unvisited_nodes[np.argmin(distance_matrix[current_node][unvisited_nodes])]
'''
```

---

#### 第二层级 — 多函数（[`tsp_construct_multifunction`](https://github.com/FeiLiu36/EOH/tree/main/examples/tsp_construct_multifunction)）

LLM 进化多个协作函数。**最后**一个顶层函数为入口点；定义在其上方的辅助函数由入口函数内部调用。这使 LLM 能够将启发式分解为可复用的子组件。

```python
template_program = '''
def compute_node_scores(current_node: int, unvisited_nodes: np.ndarray,
                        distance_matrix: np.ndarray,
                        destination_node: int) -> np.ndarray:
    """为每个候选未访问节点计算优先级分数。"""
    return -distance_matrix[current_node][unvisited_nodes]


def select_next_node(current_node: int, destination_node: int,
                     unvisited_nodes: np.ndarray,
                     distance_matrix: np.ndarray) -> int:
    """利用 compute_node_scores 选择下一个节点。"""
    scores = compute_node_scores(current_node, unvisited_nodes,
                                 distance_matrix, destination_node)
    return unvisited_nodes[np.argmax(scores)]
'''
```

---

#### 第三层级 — 类（[`tsp_construct_class`](https://github.com/FeiLiu36/EOH/tree/main/examples/tsp_construct_class)）

LLM 进化一个完整的类。**类名**为入口点——EoH 实例化该类并调用指定方法。这是表达能力最强的层级，支持有状态的启发式、内部数据结构及面向对象设计。

```python
template_program = '''
class TSPConstructor:
    """旅行商问题的构造启发式。"""

    def select_next_node(self, current_node: int, destination_node: int,
                         unvisited_nodes: np.ndarray,
                         distance_matrix: np.ndarray) -> int:
        """选择下一个访问节点。"""
        return unvisited_nodes[np.argmin(distance_matrix[current_node][unvisited_nodes])]
'''
```

---

| | 单函数 | 多函数 | 类 |
|---|:---:|:---:|:---:|
| **入口点** | 函数本身 | 最后一个顶层函数 | 类（实例化后调用） |
| **辅助组件** | — | 上方定义的辅助函数 | 方法与属性 |
| **有状态设计** | — | — | 支持 |
| **典型用途** | 大多数启发式 | 分解式评分/选择 | 有状态或复杂启发式 |
| **示例** | `tsp_construct` | `tsp_construct_multifunction` | `tsp_construct_class` |



## 示例列表

下表列出了 `examples/` 目录中的全部 33 个示例任务。

| TSP 构造启发式 | BBOB 元启发式 | Atari Breakout |
|:---:|:---:|:---:|
| <img src="./docs/figures/tsp_construct.gif" width="280"> | <img src="./docs/figures/bbob_metaheuristic.gif" width="280"> | <img src="./docs/figures/gameplay_baseline_ball-tracking.gif" width="280"> |
| `tsp_construct` | `bbob_metaheuristic` | `ale_breakout` |

| 名称 | 描述 | 链接 | 备注 |
|------|------|------|------|
| `aco_pheromone` | 为 TSP 上的蚁群优化设计信息素更新规则。 | [代码](https://github.com/FeiLiu36/EOH/tree/main/examples/aco_pheromone) | |
| `admissible_set` | 设计用于贪婪构造最大基数对称可容许集的优先级函数。 | [代码](https://github.com/FeiLiu36/EOH/tree/main/examples/admissible_set) | |
| `ale_breakout` | 为 Atari Breakout 智能体进化动作选择启发式。 | [代码](https://github.com/FeiLiu36/EOH/tree/main/examples/ale_breakout) | 需要 ALE |
| `ale_pong` | 为 Atari Pong 智能体进化动作选择启发式。 | [代码](https://github.com/FeiLiu36/EOH/tree/main/examples/ale_pong) | 需要 ALE |
| `bbob_metaheuristic` | 为连续黑箱优化设计完整的单目标元启发式算法。 | [代码](https://github.com/FeiLiu36/EOH/tree/main/examples/bbob_metaheuristic) | 类模板 |
| `bo_acquisition` | 为贝叶斯优化设计采集函数以指导候选点选择。 | [代码](https://github.com/FeiLiu36/EOH/tree/main/examples/bo_acquisition) | [PPSN 2024 最佳论文提名](https://arxiv.org/abs/2404.16906) |
| `bp_online` | 为在线装箱设计箱子评分函数以最小化使用箱数。 | [代码](https://github.com/FeiLiu36/EOH/tree/main/examples/bp_online) | EoH 论文基准 |
| `cmaes_cov_update` | 为 CMA-ES 设计协方差矩阵更新规则（10 维基准）。 | [代码](https://github.com/FeiLiu36/EOH/tree/main/examples/cmaes_cov_update) | |
| `cvrp_construct` | 为有容量约束的车辆路径问题设计贪婪构造启发式。 | [代码](https://github.com/FeiLiu36/EOH/tree/main/examples/cvrp_construct) | |
| `de_crossover_100d` | 为 100 维差分进化设计自适应交叉算子。 | [代码](https://github.com/FeiLiu36/EOH/tree/main/examples/de_crossover_100d) | 高维 |
| `de_mutation` | 为 10 维差分进化设计新型变异算子。 | [代码](https://github.com/FeiLiu36/EOH/tree/main/examples/de_mutation) | |
| `deap_eaSimple_selection` | 为基于 DEAP 的遗传算法设计亲本选择算子（10 维基准）。 | [代码](https://github.com/FeiLiu36/EOH/tree/main/examples/deap_eaSimple_selection) | 需要 DEAP |
| `es_step_size` | 为 (1+λ)-进化策略设计步长自适应规则。 | [代码](https://github.com/FeiLiu36/EOH/tree/main/examples/es_step_size) | |
| `evo_dynamic` | 为周期性环境变化下的动态优化设计种群响应策略。 | [代码](https://github.com/FeiLiu36/EOH/tree/main/examples/evo_dynamic) | 动态环境 |
| `fssp_gls` | 为流水车间调度设计引导式局部搜索扰动策略以最小化最大完工时间。 | [代码](https://github.com/FeiLiu36/EOH/tree/main/examples/fssp_gls) | |
| `gnn_aggregation` | 为 GNN 节点分类设计邻域聚合函数。 | [代码](https://github.com/FeiLiu36/EOH/tree/main/examples/gnn_aggregation) | 需要 PyTorch |
| `large_scale_es` | 为可分离 CMA-ES 设计 100 维对角方差自适应规则。 | [代码](https://github.com/FeiLiu36/EOH/tree/main/examples/large_scale_es) | 高维 |
| `mobbob_metaheuristic` | 为 2 目标 BBOB 设计多目标元启发式以最大化超体积。 | [代码](https://github.com/FeiLiu36/EOH/tree/main/examples/mobbob_metaheuristic) | 多目标；类模板 |
| `moead_decomposition` | 为 MOEA/D 设计分解函数将多目标问题转化为标量子问题。 | [代码](https://github.com/FeiLiu36/EOH/tree/main/examples/moead_decomposition) | 多目标 |
| `nsga2_crowding` | 为 NSGA-II 设计拥挤距离度量以最大化 ZDT 问题的超体积。 | [代码](https://github.com/FeiLiu36/EOH/tree/main/examples/nsga2_crowding) | 多目标 |
| `nsga2_pymoo` | 通过 pymoo 为 NSGA-II 在 ZDT 问题上设计交叉算子。 | [代码](https://github.com/FeiLiu36/EOH/tree/main/examples/nsga2_pymoo) | 多目标；需要 pymoo |
| `nurse_rostering` | 为护士排班设计班次分配优先级评分函数以平衡工作量与偏好。 | [代码](https://github.com/FeiLiu36/EOH/tree/main/examples/nurse_rostering) | |
| `one_plus_one` | 为 nevergrad 的 (1+1)-ES 设计变异噪声生成器（10 维基准）。 | [代码](https://github.com/FeiLiu36/EOH/tree/main/examples/one_plus_one) | 需要 nevergrad |
| `portfolio_construct` | 为贪婪投资组合构造设计资产评分函数以最大化夏普比率。 | [代码](https://github.com/FeiLiu36/EOH/tree/main/examples/portfolio_construct) | |
| `pso_velocity` | 为粒子群优化设计速度更新规则（10 维基准）。 | [代码](https://github.com/FeiLiu36/EOH/tree/main/examples/pso_velocity) | |
| `sa_acceptance` | 为模拟退火设计接受概率函数（10 维基准）。 | [代码](https://github.com/FeiLiu36/EOH/tree/main/examples/sa_acceptance) | |
| `tabu_tsp` | 为 TSP 禁忌搜索设计基于 2-opt 移动的评分函数。 | [代码](https://github.com/FeiLiu36/EOH/tree/main/examples/tabu_tsp) | |
| `tpe_bandwidth` | 为 Optuna 的树形 Parzen 估计器设计观测加权函数。 | [代码](https://github.com/FeiLiu36/EOH/tree/main/examples/tpe_bandwidth) | 需要 Optuna |
| `tsp_construct` | 为贪婪 TSP 路径构造设计下一节点选择启发式。 | [代码](https://github.com/FeiLiu36/EOH/tree/main/examples/tsp_construct) | EoH 论文基准 |
| `tsp_construct_class` | 将 TSP 构造启发式设计为含 `select_next_node` 方法的类。 | [代码](https://github.com/FeiLiu36/EOH/tree/main/examples/tsp_construct_class) | 类模板示例 |
| `tsp_construct_multifunction` | 为 TSP 构造设计两个协作函数（节点评分 + 选择）。 | [代码](https://github.com/FeiLiu36/EOH/tree/main/examples/tsp_construct_multifunction) | 多函数模板示例 |
| `tsp_gls` | 为 TSP 引导式局部搜索设计边距离更新策略。 | [代码](https://github.com/FeiLiu36/EOH/tree/main/examples/tsp_gls) | |
| `tsp_rnr` | 为毁坏-重建 TSP 算法设计毁坏算子（节点选择）。 | [代码](https://github.com/FeiLiu36/EOH/tree/main/examples/tsp_rnr) | |



## 大语言模型配置

#### 1) 远程 LLM API（推荐）

在 `LLMConfig` 中设置 `api_endpoint`、`api_key` 和 `model`，支持所有 OpenAI 兼容接口：

```python
llm = LLMConfig(
    api_endpoint="api.openai.com",   # 或 "api.deepseek.com" 等
    api_key="your-api-key",
    model="gpt-4o",
    timeout=150,
)
```

支持的服务商包括：
- [OpenAI API](https://platform.openai.com/)
- [DeepSeek API](https://platform.deepseek.com/)
- 任何其他 OpenAI 兼容接口

#### 2) 本地 LLM

设置 `use_local=True` 并将 `local_url` 指向您的推理服务器：

```python
llm = LLMConfig(
    use_local=True,
    local_url="http://127.0.0.1:11012/completions",
    timeout=180,
)
```

本地服务器需按 `api_local_llm.py` 所期望的格式接受 POST 请求。任何托管 Hugging Face 模型（例如通过简单的 Flask/FastAPI 包装）并返回 `{"content": ["<生成文本>"]}` 的服务器均可使用。

#### 3) 自定义实现

在 `eoh/src/eoh/llm/` 中继承 LLM 接口，即可接入任何其他 LLM 服务。



## 常见问题解答

有关安装、LLM 配置、问题定义、进化算子、结果输出及故障排查等常见问题，请参阅 **[FAQ（常见问题）](./FAQ_CN.md)**。

---

## LLM4Opt 相关工作
欢迎访问[大模型与优化参考文献和研究论文收藏](https://github.com/FeiLiu36/LLM4Opt)


## 贡献者
<img src="https://github.com/FeiLiu36.png" width="60" div align=center> [Fei Liu](https://github.com/FeiLiu36)
<img src="https://github.com/RayZhhh.png" width="60" div align=center> [Rui Zhang](https://github.com/RayZhhh) 
<img src="https://github.com/yzy1996.png" width="60" div align=center> [Zhiyuan Yang](https://github.com/yzy1996) 
<img src="https://github.com/pgg3.png" width="60" div align=center> [Ping Guo](https://github.com/pgg3)  
<img src="https://github.com/ShunyuYao6.png" width="60" div align=center> [Shunyu Yao](https://github.com/ShunyuYao6)
