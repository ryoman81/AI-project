# Phase 2：模型理解与定制化（应用与实验）

## 🎯 目标
掌握主流大模型的使用、微调、向量检索与智能体机制，具备构建复杂 LLM 应用的能力。

---

## 🔹 2.1 Hugging Face Transformers 基础

- [ ] 使用 `transformers.pipeline` 进行快速原型开发
- [ ] 使用 `AutoTokenizer`, `AutoModelFor*` 构建自定义流程
- [ ] 理解模型类型（GPT, T5, BERT, DistilBERT, Falcon, etc.）
- [ ] 熟练查阅与选择模型卡（Model Hub）

---

## 🔹 2.2 Tokenizer 与嵌入基础

- [ ] Tokenizer 类型（BPE、WordPiece、SentencePiece）
- [ ] Padding、Mask、Truncation、特殊 token 机制
- [ ] 文本向量表示基础（word embedding、CLS token、pooling）

---

## 🔹 2.3 微调（Fine-tuning）技术

- [ ] 使用 `datasets` 和 `Trainer` 实现标准微调流程
- [ ] 数据清洗与格式准备（json, csv, arrow）
- [ ] 使用 PEFT 进行轻量微调（LoRA, QLoRA, Adapters）
- [ ] 模型训练技巧（learning rate, weight decay, early stopping）
- [ ] 模型验证与保存（metrics, checkpoint）

---

## 🔹 2.4 向量检索 + RAG 架构

- [ ] 文档预处理（分段、去噪、结构保持）
- [ ] 嵌入模型使用（`sentence-transformers`, `text-embedding-ada-*`, E5）
- [ ] 使用向量数据库（`faiss`, `chromadb`, `weaviate`）
- [ ] 构建完整的 RAG 系统（retriever + reranker + generator）

---

## 🔹 2.5 智能体与工具调用（Function Calling + MCP）

- [ ] OpenAI / Transformers 工具调用机制（function calling, tools API）
- [ ] 结构化上下文注入（符合 Model Context Protocol 原则）
- [ ] 多工具组合调用 + 函数路由（如搜索 + 计算器 + 检索器）
- [ ] 简易智能体框架构建（Agent Loop, Tool Hub, Memory）

---

## 🔹 2.6 Prompt Engineering 技术

- [ ] Prompt 模式：Zero-shot / Few-shot / Chain-of-Thought / Self-Ask
- [ ] Prompt 模板构造（`langchain`, `PromptTemplate`, `jinja2`）
- [ ] Prompt 注入防御、上下文长度管理
- [ ] Prompt Tuning / Prefix Tuning（了解原理 + 应用）


# Phase 3：工程平台与部署实践（MLE能力核心）

## 🎯 目标
掌握模型部署、监控、评估、平台化开发能力。能将 LLM 应用端到端部署与持续优化。

---

## 🔹 3.1 模型服务部署

- [ ] 使用 `FastAPI` / `Flask` 封装推理接口
- [ ] 加载与缓存 tokenizer / model（低延迟优化）
- [ ] Batching, half precision（`torch.bfloat16`, `fp16`）
- [ ] 部署至本地、Docker、Azure App Service

---

## 🔹 3.2 Azure ML 必学能力

- [ ] 注册与使用 Azure ML Workspace
- [ ] 使用 compute instance / cluster 进行训练
- [ ] 自定义训练脚本与 YAML job pipeline（`command job`, `pipeline job`）
- [ ] 使用 `mlflow` 跟踪模型与实验（integrated）
- [ ] 模型注册、版本控制、部署与 endpoint 暴露
- [ ] 监控模型运行状态与资源用量

---

## 🔹 3.3 Azure AI Foundry 必学能力

- [ ] 使用 Azure Studio 或 CLI 创建 Foundry 项目
- [ ] 构建完整的 promptflow（节点式 prompt 编排）
- [ ] 集成模型（OpenAI, HF, Azure endpoints）
- [ ] 微调 pipeline（data preprocess + fine-tune + eval）
- [ ] 使用 PromptFlow Eval 设计自定义指标
- [ ] 发布与共享项目（多人协作、分支管理）

---

## 🔹 3.4 模型评估与 CI/CD

- [ ] 指标：Accuracy, F1, BLEU, ROUGE, embedding similarity
- [ ] 手动评估 + 自动化评估（PromptFlow Eval、HF Evaluate）
- [ ] 使用 `mlflow`, `Azure ML`, `PromptFlow` 跟踪和可视化指标
- [ ] 构建模型自动部署流程（CI/CD：GitHub Actions + Azure）

---

## 🔹 3.5 数据与训练流水线管理

- [ ] 数据版本控制（`dvc`, `mlflow artifacts`, Azure data assets）
- [ ] 数据预处理 pipeline（结构化文本 + 标注数据）
- [ ] 异常检测与数据清洗（nulls, outliers, duplicates）
- [ ] 多任务数据分流（用于不同模型）

---

## 🔹 3.6 模型优化与资源管理

- [ ] 多 GPU / 多实例训练（了解 `deepspeed`, `accelerate`）
- [ ] 显存优化：gradient checkpointing, quantization
- [ ] 推理优化：ONNX, TensorRT（了解）
- [ ] 模型压缩技术（Pruning, Distillation - 理解框架）

---

## 🔹 3.7 安全性与工业规范（Optional Advanced）

- [ ] Prompt 注入与反制机制（Guardrails, TypeScript wrappers）
- [ ] 模型输出审查（toxicity, jailbreak）
- [ ] 权限控制与用户数据合规（GDPR, 企业内部策略）

