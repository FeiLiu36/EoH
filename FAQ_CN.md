# EoH 常见问题解答

## 目录

- [EoH 常见问题解答](#eoh-常见问题解答)
  - [目录](#目录)
  - [1. EoH 是什么，能解决哪些问题？](#1-eoh-是什么能解决哪些问题)
  - [2. 如何安装 EoH？](#2-如何安装-eoh)
  - [3. 支持哪些大语言模型？](#3-支持哪些大语言模型)
  - [4. 如何配置大语言模型（API 密钥、端点、模型名称）？](#4-如何配置大语言模型api-密钥端点模型名称)
  - [5. API 调用频繁超时，如何解决？](#5-api-调用频繁超时如何解决)
  - [6. 如何定义自己的优化问题？](#6-如何定义自己的优化问题)
  - [7. 支持哪些模板类型？](#7-支持哪些模板类型)
  - [8. 初始模板函数应该写什么？](#8-初始模板函数应该写什么)
  - [9. EoH 是最大化还是最小化适应度值？](#9-eoh-是最大化还是最小化适应度值)
  - [10. 进化算子 e1、e2、m1、m2 分别是什么？](#10-进化算子-e1e2m1m2-分别是什么)
  - [11. 如何运行 EoH，有哪些关键配置参数？](#11-如何运行-eoh有哪些关键配置参数)
  - [12. 结果保存在哪里？](#12-结果保存在哪里)
  - [13. 如何恢复中断的运行？](#13-如何恢复中断的运行)
  - [14. 如何用手工设计的算法作为初始种子？](#14-如何用手工设计的算法作为初始种子)
  - [15. 如何通过并行加速评估？](#15-如何通过并行加速评估)
  - [16. 为什么 EoH 没有产生有效结果或始终返回 None？](#16-为什么-eoh-没有产生有效结果或始终返回-none)
  - [17. 模板中的注释对 EoH 性能有何影响？](#17-模板中的注释对-eoh-性能有何影响)
  - [18. 基于大语言模型的启发式设计相比传统方法有哪些优势？](#18-基于大语言模型的启发式设计相比传统方法有哪些优势)
  - [19. 有哪些已知的局限性或失效场景？](#19-有哪些已知的局限性或失效场景)

---

## 1. EoH 是什么，能解决哪些问题？

**EoH（Evolution of Heuristics，启发式进化）** 是一个将进化计算（EC）与大语言模型（LLM）相结合的框架，可自动为搜索和优化问题设计算法与启发式方法，无需人工专家介入。

EoH 同时对启发式的推理过程（"思路"）和代码实现进行协同进化，以大语言模型作为智能的变异/交叉算子，驱动跨代演化。

EoH 已被应用于 33 种以上的问题类型，包括：

- **组合优化**：TSP、CVRP、装箱问题、护士排班、圆填充
- **元启发式组件设计**：PSO 速度更新、DE 变异策略、SA 接受准则、CMA-ES 更新
- **构造算法**：路径规划与调度的贪婪启发式
- **机器学习组件**：GNN 聚合函数、贝叶斯优化采集函数
- **动态/在线问题**：适应环境变化的策略

EoH 已在 **ICML 2024（口头报告，Top 1.5%）** 发表，在圆填充问题上创造了**世界纪录**，并在 **CVRPLib BKS 竞赛**中以 51 个新最优已知解夺冠。

---

## 2. 如何安装 EoH？

```bash
git clone https://github.com/FeiLiu36/EoH.git
cd eoh
pip install .
```

**依赖要求：** Python >= 3.10、`numpy`、`joblib`。

可通过以下方式验证安装：

```python
import eoh
print(eoh.__version__)
```

---

## 3. 支持哪些大语言模型？

EoH 支持所有通过 **OpenAI 兼容 API** 访问的大语言模型，以及本地推理服务器。

| 服务商 | 说明 |
|---|---|
| OpenAI（GPT-4o、GPT-4 等） | 设置 `api_endpoint="api.openai.com"` |
| DeepSeek（deepseek-chat 等） | 设置 `api_endpoint="api.deepseek.com"` |
| 任何 OpenAI 兼容接口 | 开箱即用 |
| 本地 LLM（HuggingFace 模型） | 设置 `use_local=True`，并将 `local_url` 指向推理服务器 |

建议使用代码生成能力较强的模型（如 GPT-4o、DeepSeek-V3 等）。弱模型可能产生更多语法无效的程序。

---

## 4. 如何配置大语言模型（API 密钥、端点、模型名称）？

使用 `LLMConfig` 配置大语言模型后端：

```python
from eoh import LLMConfig

llm = LLMConfig(
    api_endpoint="api.deepseek.com",   # 仅填主机名，不含 https://
    api_key="your-api-key-here",
    model="deepseek-chat",
    timeout=150,                        # 单次 LLM 调用的最大等待秒数
)
```

**本地 LLM 配置：**

```python
llm = LLMConfig(
    use_local=True,
    local_url="http://localhost:8080",  # 推理服务器地址
    model="your-model-name",
    timeout=180,
)
```

常见错误：
- `api_endpoint` 中**不要**填写 `https://`，只填主机名即可。
- 不同服务商的 API 密钥格式不同，DeepSeek 密钥与 OpenAI 密钥不通用。
- 如果使用代理或第三方端点，请严格核对 URL，即使多余的斜杠也可能导致请求失败。

---

## 5. API 调用频繁超时，如何解决？

需要注意两个独立的超时参数：

| 参数 | 控制内容 | 设置位置 |
|---|---|---|
| `LLMConfig(timeout=...)` | 等待单次 LLM 响应的最大秒数 | `LLMConfig` |
| `BaseProblem(timeout=...)` | 单个生成程序允许运行的最大秒数 | `BaseProblem` 子类 |

**对于响应较慢的模型**（如带链式推理的 DeepSeek-R1），默认的 `timeout=180` 往往不够。建议设为 `timeout=300` 或更高。

**对于评估较慢的问题**（复杂问题实例），同样需要相应增大问题级别的超时时间。

网络不稳定也是 API 调用失败的常见原因。如出现间歇性错误，请检查网络连接，并考虑在基础设施层面添加重试机制。

---

## 6. 如何定义自己的优化问题？

继承 `BaseProblem` 并实现两个必需属性：

```python
from eoh import BaseProblem
import numpy as np

class MyProblem(BaseProblem):
    # LLM 将进化的代码骨架
    template_program = '''
def heuristic(items: list, capacity: float) -> list:
    """选择物品，在不超过容量的前提下最大化总价值。"""
    return sorted(items, key=lambda x: x[1] / x[0], reverse=True)
'''
    # 目标的自然语言描述
    task_description = (
        "设计一个启发式算法，在不超过重量容量的前提下，"
        "为背包选择物品以最大化总价值。"
    )

    def evaluate_program(self, program_str: str, callable_func) -> float | None:
        """返回适应度分数（越小越好），失败时返回 None。"""
        try:
            items = [(2, 3), (3, 4), (4, 5), (5, 8), (9, 10)]
            selected = callable_func(items, capacity=10.0)
            return -sum(v for _, v in selected)  # 取反：价值越高 → 适应度越小（越好）
        except Exception:
            return None
```

关键规则：
- `evaluate_program` 必须返回 **float**（越小越好）或 **None**（程序无效时）。
- `task_description` 应简洁清晰——它会直接注入到 LLM 提示词中。
- 实例化时使用 `MyProblem(timeout=30, n_processes=4)`。

---

## 7. 支持哪些模板类型？

EoH 支持三种模板风格：

**1. 单函数**（最常用）
LLM 进化一个函数。适用于大多数启发式设计任务。

**2. 多函数**
多个协作函数；最后定义的函数是 `evaluate_program` 调用的入口点。适用于希望 LLM 同时设计辅助函数和主函数的场景。

**3. 类**
带有指定入口方法的类模板。适用于有状态的启发式，或面向对象结构更自然的问题。

模板类型根据 `template_program` 的结构自动识别，无需手动指定。

---

## 8. 初始模板函数应该写什么？

**尽量保持简单。** 最小化的、正确的基线比复杂的实现更好。推荐选择：

- 平凡但正确的实现（如随机选择、首次适应、最近邻）
- 该领域的简单经典算法（如背包问题的价值/重量比贪婪法）

应避免：
- 过于复杂的实现，会限制搜索空间
- 含有大量编码了过多领域知识的注释（见第 17 题）

初始模板确定了 LLM 必须遵守的函数签名和文档字符串——请确保这两者清晰且具有信息量。

如果您希望 EoH 从已有的优质算法出发，请使用种子机制（见第 14 题），而非将其编码到模板中。

---

## 9. EoH 是最大化还是最小化适应度值？

**EoH 最小化** `evaluate_program` 返回的值，返回值越小越好。

若要**最大化**某个目标（如总价值），只需对返回值取反：

```python
def evaluate_program(self, program_str, callable_func):
    value = run_evaluation(callable_func)
    return -value  # 取反：价值越高 → 适应度越小（越好）
```

---

## 10. 进化算子 e1、e2、m1、m2 分别是什么？

EoH 使用四个将 LLM 生成与进化搜索相结合的算子：

| 算子 | 类型 | 描述 |
|---|---|---|
| `e1` | 交叉 | 结合两个亲本程序的**代码** |
| `e2` | 交叉 | 结合两个亲本的**思路**（推理过程）与代码 |
| `m1` | 变异 | 修改单个程序的**代码** |
| `m2` | 变异 | 修改单个程序的**思路**，再重新生成代码 |

默认情况下，四个算子等权使用：`operators=['e1', 'e2', 'm1', 'm2']`。

可以限定使用算子的子集，或指定自定义权重：

```python
eoh = EoH(
    ...,
    operators=['e1', 'm1'],
    operator_weights=[0.7, 0.3],
)
```

---

## 11. 如何运行 EoH，有哪些关键配置参数？

```python
from eoh import EoH, LLMConfig

llm = LLMConfig(api_endpoint="api.deepseek.com", api_key="...", model="deepseek-chat")
problem = MyProblem(timeout=40, n_processes=4)

eoh = EoH(
    llm=llm,
    problem=problem,
    pop_size=5,          # 每代保留的种群大小
    n_pop=20,            # 进化代数
    operators=['e1', 'e2', 'm1', 'm2'],
    output_dir="./results",
    debug=False,
)
eoh.run()
```

**关键参数说明：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `pop_size` | 5 | 每代保留的程序数量 |
| `n_pop` | 20 | 运行的进化代数 |
| `operators` | 全部四个 | 使用的进化算子 |
| `operator_weights` | 均匀分布 | 各算子的采样权重 |
| `n_parents` | 2 | 交叉时使用的亲本数量 |
| `output_dir` | `"./"` | 日志和结果的保存目录 |
| `debug` | `False` | 开启详细日志输出 |
| `use_seed` | `False` | 从种子文件加载初始种群 |
| `use_continue` | `False` | 从上次运行继续 |

---

## 12. 结果保存在哪里？

结果写入 `output_dir`（默认 `./`），目录结构如下：

```
results/
  run_log.txt                          # 进化进度日志，包含每代的适应度信息
  samples/
    samples_0~N.json                   # 批量保存的评估程序（代码 + 适应度 + 思路）
    ...
    samples_best.json                  # 所有代中发现的最优程序
  pops/
    population_generation_1.json       # 第 1 代结束后的完整种群快照
    population_generation_2.json
    ...
  pops_best/
    population_generation_1.json       # 每代的最优个体快照
    population_generation_2.json
    ...
```

最优解始终保存在 `samples/samples_best.json`。若该文件缺失或为空，请查看 `run_log.txt` 排查错误——通常是 LLM 在返回有效程序前就已超时。

---

## 13. 如何恢复中断的运行？

在创建 `EoH` 实例时设置 `use_continue=True`，并将 `output_dir` 指向与上次运行相同的目录：

```python
eoh = EoH(
    llm=llm,
    problem=problem,
    pop_size=5,
    n_pop=20,
    output_dir="./results",   # 与上次相同
    use_continue=True,
)
eoh.run()
```

EoH 将从 `pops/` 中加载最后保存的种群，并从中断处继续运行。

---

## 14. 如何用手工设计的算法作为初始种子？

您可以提供一组已知算法作为初始种群，让 EoH 从强基线出发，而非从零开始进化。准备一个与种群快照（`pops/population_generation_N.json`）格式相同的种子 JSON 文件，然后：

```python
eoh = EoH(
    ...,
    use_seed=True,
    seed_path="./my_seed_algorithms.json",
)
eoh.run()
```

种子文件中每个条目至少需要包含 `"code"` 和 `"fitness"` 字段。具体格式可参考 `examples/` 目录中的示例。

---

## 15. 如何通过并行加速评估？

在实例化问题时设置 `n_processes`：

```python
problem = MyProblem(timeout=40, n_processes=-1)  # -1 使用全部可用 CPU
```

每个工作进程独立评估一个生成的程序。并行评估是安全的，因为每个程序在独立子进程中运行，并由 `joblib` 强制执行超时限制。

注意：如果您的评估函数内部本身也使用了多进程，请谨慎嵌套，避免启动过多进程。

---

## 16. 为什么 EoH 没有产生有效结果或始终返回 None？

常见原因：

- **LLM 超时**：模型响应时间超过 `LLMConfig(timeout=...)` 的限制。请增大超时值（见第 5 题）。
- **评估超时**：生成的程序运行时间超过 `BaseProblem(timeout=...)` 的限制。请增大问题超时或简化评估逻辑。
- **生成代码存在语法错误**：弱模型产生无效 Python 的概率更高。请换用更强的模型，或在 `evaluate_program` 中添加 `try/except` 在异常时返回 `None`。
- **返回类型错误**：`evaluate_program` 必须返回 `float` 或 `None`。返回非数值类型（如列表）会导致静默失败。
- **模板不匹配**：若 `callable_func` 的签名与 `evaluate_program` 中的调用方式不符，每个程序都会抛出 `TypeError`。请仔细核对 `template_program` 中的参数名称。

在 `EoH(...)` 中启用 `debug=True`，可查看完整的 LLM 响应内容和评估失败的异常堆栈。

---

## 17. 模板中的注释对 EoH 性能有何影响？

`template_program` 中的注释会被包含在 LLM 提示词中，并影响模型的生成内容。基本原则是：**在说清楚关键信息的前提下，越简洁越好。**

- **简短的文档字符串**（说明函数用途和参数类型）有助于模型理解，应始终保留。
- **解释隐式约束或不变量的行内注释**，可帮助 LLM 避免无效设计。
- **大量编码了领域知识或特定算法思路的注释**，可能无意中限制搜索空间，降低多样性。

建议先从最简模板开始，只有在 LLM 持续误解任务时才逐步添加注释。

---

## 18. 基于大语言模型的启发式设计相比传统方法有哪些优势？

- **高度自动化**：除定义问题接口外，无需人工特征工程或领域专属调参。
- **无需训练数据**：大语言模型从预训练中带来先验知识，无需收集或标注样本。
- **可解释性强**：生成的启发式是人类可读的 Python 代码，而非"黑箱"神经网络策略。
- **通用性强**：同一框架可跨越不同问题类型使用，无需修改算法本身。
- **速度快**：通常在几分钟到几小时内，即可在普通笔记本电脑上找到有竞争力的启发式方法。

---

## 19. 有哪些已知的局限性或失效场景？

- **问题描述复杂**：从复杂的现实问题中提炼出简洁、无歧义的任务描述具有挑战性。`task_description` 写得不好是结果质量低下的常见原因。
- **模型能力上限**：进化出的启发式质量受限于大语言模型的代码生成能力。过于弱的模型可能始终无法产生有效程序。
- **评估成为瓶颈**：若单次评估耗时数分钟，总运行时间会迅速增长。请合理使用 `n_processes` 并严格控制 `timeout`。
- **复现性**：大语言模型输出具有随机性，相同设置的两次运行会产生不同的启发式方法。发表研究时建议运行多个随机种子并取最优结果。
- **模板设计敏感性**：初始模板和函数签名的选择会影响 LLM 的探索空间。设计不当的模板（如错误的返回类型、含糊的参数名）会显著降低性能。
