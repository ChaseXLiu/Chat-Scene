# Methodology

## 3.1 Preliminary

### Reinforcement Learning with Verifiable Rewards

Reinforcement Learning with Verifiable Rewards (RLVR) 是一种训练方法，用于增强语言模型在具有**可验证结果**的任务（如数学和代码）中的表现。与依赖人工偏好数据训练的奖励模型的 RLHF 不同，RLVR 使用直接的**验证函数**来判断正确性，从而简化奖励机制。

给定输入问题 $q$，策略模型 $\pi_\theta$ 生成响应 $o$ 并获得可验证奖励，优化目标为：

$$
\max_{\pi_\theta} \ \mathbb{E}_{o \sim \pi_\theta(q)} \big[ R_{\text{RLVR}}(q, o) \big]
=[R(q, o) - \beta \, \text{KL}\big[\pi_\theta(o|q) \parallel \pi_{\text{ref}}(o|q)\big]]
\tag{1}
$$

其中，$\pi_{\text{ref}}$ 为参考模型，$\beta$ 控制 KL 正则项，$R$ 为可验证奖励函数。奖励函数定义为：

$$
R(q, o) =
\begin{cases}
1, & \text{if } o = \text{ground truth} \\
0, & \text{otherwise}
\end{cases}
\tag{2}
$$

---

### DeepSeek R1-Zero and GRPO

DeepSeek R1-Zero 引入 **Group Relative Policy Optimization (GRPO)**，无需监督微调即可训练。不同于 PPO 依赖 critic 模型，GRPO 通过比较候选响应组的相对质量来优化。

对于给定问题 $q$，生成 $G$ 个候选响应 $\{o_1, o_2, ..., o_G\}$，并计算对应奖励 $\{r_1, r_2, ..., r_G\}$。定义归一化的优势函数：

$$
A_i = \frac{r_i - \text{mean}(\{r_1, \dots, r_G\})}{\text{std}(\{r_1, \dots, r_G\})}
\tag{4}
$$

其中，$A_i$ 表示第 $i$ 个答案的相对质量，GRPO 引导模型偏向组内更优的答案。

---

## 3.2 Visual-RFT

输入包含图像和文本问题，策略模型 $\pi_\theta$ 输出**推理过程**与候选响应。每个响应通过可验证奖励函数计算奖励，再结合组内归一化进行更新。训练过程中还加入 KL 散度约束，以保证策略模型与参考模型保持稳定。

Visual-RFT 的关键在于 **视觉任务的可验证奖励设计** 和 **数据准备流程**。

---

### 3.2.1 Verifiable Reward in Visual Perception

为视觉感知任务设计基于规则的可验证奖励函数：

#### IoU Reward in Detection Tasks

检测任务的输出包含边界框 $b_i$ 和置信度 $c_i$。我们提出的奖励函数 $R_d$ 包含三部分：

$$
R_d = R_{\text{IoU}} + R_{\text{conf}} + R_{\text{format}}
\tag{5}
$$

- **IoU 奖励**  
  $$ 
  R_{\text{IoU}} = \frac{1}{n} \sum_{i=1}^n \text{IoU}_i 
  \tag{6}
  $$

- **置信度奖励**  
  对于每个预测框：
  $$
  r_{c_i} =
  \begin{cases}
  c_i, & \text{if } \text{IoU}_i \neq 0 \\
  1 - c_i, & \text{if } \text{IoU}_i = 0
  \end{cases}
  \tag{7}
  $$

  整体置信度奖励为：
  $$
  R_{\text{conf}} = \frac{1}{n} \sum_{i=1}^n r_{c_i}
  \tag{8}
  $$

- **格式奖励**  
  $R_{\text{format}}$ 用于确保模型输出符合 `<think>` 与 `<answer>` 的格式要求。

---

#### CLS Reward in Classification Tasks

分类任务的奖励函数由准确率奖励与格式奖励组成：

$$
R_{\text{cls}} = R_{\text{acc}} + R_{\text{format}}
\tag{9}
$$

其中，$R_{\text{acc}}$ 根据预测类别是否与真实标签一致来判定（正确为 1，错误为 0）。

---

### 3.2.2 Data Preparation

为了在不同视觉任务上训练 Visual-RFT，需要构建多模态训练数据集。  
关键步骤：

- **提示设计（Prompt Design）**  
  - 检测任务：要求输出推理过程 `<think>` 与边界框预测 `<answer>`。  
  - 分类任务：要求输出推理过程 `<think>` 与类别名称 `<answer>`。  

- **格式奖励的作用**  
  强制模型在输出时包含推理步骤与最终答案，以促进自我学习与推理能力提升。

---

# 小结

Methodology 部分主要包括：
1. **预备知识**：介绍 RLVR 和 GRPO；
2. **Visual-RFT 框架**：结合推理过程、可验证奖励和 KL 约束；
3. **可验证奖励设计**：针对检测与分类任务提出 IoU/CLS 奖励；
4. **数据准备**：通过特定 Prompt 设计和格式奖励，提升模型的推理与感知能力。
