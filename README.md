# LlamaIndex-Project

基于 [LlamaIndex](https://docs.llamaindex.ai/) 的 RAG（检索增强生成）学习与实践项目，按天（day01 ~ day04）组织，从提示词模板、文档/节点解析、摄取管道，逐步深入到属性图索引、向量存储与自定义查询/检索/聊天引擎。

- **LLM**：阿里云百炼（DashScope，默认 `qwen-plus`）、DeepSeek
- **Embedding**：本地 HuggingFace 模型（BAAI/bge 系列，CPU 运行，不依赖在线 API）
- **向量库**：Chroma（本地持久化）、内置 `SimpleVectorStore`

---

## 环境要求

- Python 3.12（项目内置 `.venv`）
- macOS / Linux / Windows 均可（Embedding 走 CPU）
- 已安装依赖见 `requirements.txt`

安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> `day04/09-混合检索.py` 还用到了 `llama-index-retrievers-bm25`，若运行报缺少该模块，执行：
> `pip install llama-index-retrievers-bm25`

---

## 配置

在项目根目录创建 `.env` 文件：

```dotenv
# 百炼（DashScope）
DASHSCOPE_API_KEY=sk-xxxxxx
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL_NAME=qwen-plus

# DeepSeek
DEEPSEEK_API_KEY=sk-xxxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL_NAME=deepseek-chat
```

`util/load_model.py` 中 Embedding 使用**本地绝对路径**，换机器时需改成自己的模型目录：

```python
# util/load_model.py
model_name = r"/Users/zhangjinshan/LLM/local_model/BAAI/bge-large-zh-v1___5"
embed_model = HuggingFaceEmbedding(model_name=model_name, device="cpu", max_length=512)
```

> **重要**：`get_llm()` / `get_embed()` 会在内部写入 `Settings.llm` / `Settings.embed_model`，
> 必须在构建索引**之前**调用，否则 LlamaIndex 会回落到 OpenAI 默认实现并报 `No API key`。

---

## 运行方式

脚本内使用的是**相对路径**（如 `./data/小说.txt`），因此需要把**工作目录**设为对应的 day 目录，
同时保证**项目根目录**在 `PYTHONPATH` 中（脚本依赖 `from util.load_model import ...`）。

```bash
# 在项目根目录执行
cd day01 && PYTHONPATH=.. python "定义节点.py"
```

PyCharm 用户：Run Configuration 中
`Working directory` 设为 `.../LlamaIndex-Project/day0X`，并勾选 `Add content roots to PYTHONPATH`。

---

## 目录结构

```
LlamaIndex-Project/
├── util/
│   └── load_model.py          # 统一加载 LLM / Embedding（DashScope、DeepSeek、HuggingFace）
├── docs/                      # 参考资料
├── day01/                     # 提示词模板、文档与节点、Agent 入门
├── day02/                     # 节点解析器 & 摄取管道
├── day03/                     # 属性图索引（Property Graph）与元数据提取
├── day04/                     # 存储层、查询引擎、聊天引擎、检索器
├── requirements.txt
└── SSH配置指南.md
```

各 day 目录下的 `data/` 存放示例语料，`chroma_db/`、`storage/`、`documents/`、`vector_store_index/`
等目录为运行后自动生成的持久化产物。

---

## 分天内容

### day01 — 提示词模板与基础对象

| 文件 | 说明 |
| --- | --- |
| `定义文档.py` | `Document` 手动构造元数据、`SimpleDirectoryReader` + `file_metadata` 自动注入元数据 |
| `定义节点.py` | 用 `SentenceSplitter` 把 Document 切成 Node |
| `定义节点-手动.py` | 手动建立 `NEXT` / `PREVIOUS` 双向节点关系 |
| `ChatPromptTemplate提示词模板.py` | 聊天消息模板，`format` / `format_messages` |
| `RichPromptTemplate提示词模板.py` | Jinja 风格（`{{ var }}`）富文本模板 |
| `部分格式化.py` | `partial_format` 预填部分变量 |
| `函数映射.py` | `function_mappings` 在渲染时做敏感信息脱敏 |
| `动态小样本.py` | 从索引中检索示例，动态填充 few-shot |
| `元数据提取.py` | `TitleExtractor` + `QuestionsAnsweredExtractor`（自定义中文提示词） |
| `starter_agent.py` | `FunctionAgent` 同时挂载计算工具与 RAG 工具 |

### day02 — 节点解析器与摄取管道

| 文件 | 说明 |
| --- | --- |
| `句子分割器.py` | `SentenceSplitter` 基础切分 |
| `句子窗口节点解析器.py` | `SentenceWindowNodeParser`，保留上下文窗口 |
| `语义分割节点解析器.py` | `SemanticSplitterNodeParser`，按语义相似度切分 |
| `层次节点解析器.py` | `HierarchicalNodeParser`，多层粒度节点 |
| `文件节点解析器.py` | `SimpleFileNodeParser`，按文件切分 |
| `json节点解析器.py` / `json读取器.py` | JSON 数据读取与节点解析 |
| `html节点解析器.py` | HTML 结构解析 |
| `摄取管道-基础使用.py` | `IngestionPipeline` 组合切分 + 向量化 + 标题提取 |
| `摄取管道-向量存储.py` | 管道接 Chroma，并演示 `persist` / `load` 缓存去重 |

### day03 — 属性图索引（Property Graph）

| 文件 | 说明 |
| --- | --- |
| `01-存储向量索引.py` | Chroma + `IngestionPipeline` 构建并持久化向量索引 |
| `02-提取器.py` | `PropertyGraphIndex` 默认提取器快速上手 |
| `03-SimpleLLMPathExtractor提取器.py` | 简单三元组提取 |
| `03-自定义属性图提取器.py` | 自定义 `extract_prompt` + `parse_fn` 解析 `实体|关系|实体` |
| `04-ImplicitPathExtractor提取器.py` | 隐式路径提取 |
| `05-SchemaLLMPathExtractor提取器.py` | 限定实体/关系 Schema + `LLMSynonymRetriever`、`VectorContextRetriever` |
| `06-元数据提取.py` | `TitleExtractor`、`QuestionsAnsweredExtractor`、`SummaryExtractor`、`KeywordExtractor` 组合 |

### day04 — 存储、查询引擎与检索

| 文件 | 说明 |
| --- | --- |
| `01-路由索引.py` | `RouterQueryEngine` + `LLMSingleSelector` 多知识库路由 |
| `02-向量存储.py` | `SimpleVectorStore` 与 Chroma 两种向量存储、持久化与重载 |
| `03-文档存储.py` | `SimpleDocumentStore` 文档存储与本地持久化 |
| `04-索引存储.py` | 索引持久化（`persist`）与 `load_index_from_storage` 重载 |
| `05-键值存储.py` | `SimpleKVStore` 存取与持久化 |
| `06-配置查询引擎.py` | 通过参数配置查询引擎（`similarity_top_k`、`response_mode` 等） |
| `06-自定义查询引擎.py` | 继承 `CustomQueryEngine` 自定义检索 + 生成流程 |
| `07-配置聊天引擎.py` | `SimpleChatEngine` / `CondenseQuestionChatEngine`，含流式输出 |
| `08-自定义检索器.py` | 自定义 `BaseRetriever`，向量 + 关键词 AND/OR 融合 |
| `09-混合检索.py` | `QueryFusionRetriever` 稠密向量 + BM25 混合检索 |

---

## 注意事项

1. **模型设置顺序**：先 `get_llm()` / `get_embed()`，再建索引。
2. **DashScope 流式与工具调用**：`FunctionAgent` 需设置 `streaming=False`，否则工具调用参数解析会报错。
3. **重复向量化**：`chroma_db/`、`storage/` 等目录已存在数据时重复运行会写入重复向量，
   调试时可先删除这些目录；`IngestionPipeline` 的 `persist`/`load` 可基于缓存跳过重复计算。
4. **中文 Embedding**：默认使用本地 `bge` 中文模型，首次运行需下载/加载模型，CPU 下较慢属正常。
5. **PDF 读取**：`day04/data` 含 PDF 语料，LlamaIndex 读取 PDF 走 `pypdf` 后端即可，无需额外解析器。
