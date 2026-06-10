# Basic RAG — 本地文档智能问答系统

基于 RAG（Retrieval-Augmented Generation）架构的本地 PDF 文档问答系统。用户上传 PDF 后，系统完成文档解析、文本切分、向量化入库、相似度检索，并调用本地大语言模型生成基于文档内容的回答。

适用场景：企业内部知识库检索、合同/制度文档问答、本地隐私敏感场景的文档智能处理。

本项目覆盖 RAG 完整链路：文档加载 → 文本切分 → Embedding → ChromaDB 持久化 → 检索 → Prompt 构造 → LLM 生成。

---

## Demo

> 将运行效果截图放入 `docs/images/demo.png`，以下为占位说明。

![demo](docs/images/demo.png)

*上传 PDF 文档后，在聊天界面输入问题即可获得基于文档内容的回答。*

---

## Architecture

> 将系统架构图放入 `docs/images/architecture.png`，以下为占位说明。

![architecture](docs/images/architecture.png)

**组件关系：**

- **PDF 文档** — 用户上传的原始 PDF 文件
- **文档解析** — `pypdf` 提取文本，自动跳过空白页，识别扫描版
- **Chunk 切分** — `RecursiveCharacterTextSplitter` 按语义边界递归切块
- **Embedding** — Ollama 运行 `mxbai-embed-large` 将文本块向量化
- **ChromaDB** — 向量持久化存储，支持应用重启后复用知识库
- **Retriever** — Top-K 相似度检索，返回与查询最相关的文档片段
- **Prompt** — 将检索结果与用户问题组装为 RAG 提示词模板
- **Ollama LLM** — 本地运行 `qwen3:8b` 生成基于上下文的回答

---

## Workflow

> 将工作流程图放入 `docs/images/workflow.png`，以下为占位说明。

![workflow](docs/images/workflow.png)

```
用户问题 → Embedding 向量化 → ChromaDB 向量检索 → Top-K Chunk
    → Prompt 组装（上下文 + 问题） → Ollama LLM 生成答案 → 返回用户
```

**文档入库流程：**

```
PDF 上传 → 文本提取 → 递归切块 → Ollama Embedding → ChromaDB 持久化
```

---

## Tech Stack

| 分类 | 技术 |
|------|------|
| Web UI | Streamlit |
| RAG 编排 | LangChain Core（LCEL） |
| 本地模型 | Ollama — `mxbai-embed-large`（Embedding）/ `qwen3:8b`（Chat） |
| 向量数据库 | ChromaDB + LangChain Chroma |
| 文档解析 | pypdf |
| 环境管理 | uv + python-dotenv |
| 测试 | pytest + unittest.mock |

---

## Features

- 支持单/多 PDF 上传与批量入库，提供入库进度反馈
- 自动过滤空白页，扫描版 PDF 给出 OCR 提示
- ChromaDB 持久化向量知识库，应用重启数据不丢失
- 基于相似度检索返回 Top-K 相关文档片段
- 本地 Ollama 运行 Embedding 与 Chat 模型，无需云端 API Key
- 上传文件处理完成后自动清理临时文件
- 针对模型未下载、Ollama 未启动等场景给出可操作的中文错误提示

---

## Highlights

- **端到端本地 RAG** — 文档、向量库、模型调用均在本机完成，无需联网，适合隐私敏感场景
- **清晰模块边界** — 加载、切分、检索、提示词、模型链路和向量库各自独立，职责明确
- **可维护配置** — 路径、模型、切分参数统一由 `config.py` 管理，修改方便
- **持久化知识库** — ChromaDB 本地落地，重启即用
- **工程化异常处理** — 模型缺失、服务不可用、文档格式不支持等场景均有明确提示
- **可测试性** — 模型和向量库调用支持依赖注入，测试无需启动 Ollama
- **安全临时文件管理** — 上传文件名路径收敛，`finally` 块保证始终清理
- **可扩展** — 模块化设计，天然支持演进为 Corrective RAG / Agentic RAG / Multi-Agent

---

## Project Structure

```
.
├── app.py                    # Streamlit 入口，页面布局与业务流程编排
├── config.py                 # 全局配置：路径、模型、切分、检索、批处理参数
├── rag/                      # RAG 核心逻辑
│   ├── chain.py              #   检索 + Prompt + LLM + 输出解析（LCEL 链）
│   ├── loader.py             #   PDF 文本提取，OCR 场景识别
│   ├── retriever.py          #   基于 Chroma 的相似度检索
│   └── splitter.py           #   递归文本切块
├── vector_store/             # 向量数据库封装
│   └── chroma_store.py       #   Embedding 模型管理、Chroma 初始化与批量写入
├── prompts/                  # 提示词模板
│   └── rag_prompt.py         #   RAG 提示词
├── tests/                    # 单元测试与集成测试
├── docs/                     # 示例文档（sample.pdf 用于项目演示）
│   └── images/               # 架构图与演示截图（按需创建）
├── pyproject.toml            # 项目依赖与配置（uv 管理）
└── README.md
```

以下为运行时自动创建的本地数据目录，不会提交到 GitHub：

```
chroma_db/    ChromaDB 持久化向量库
temp/         PDF 上传临时中转
uploads/      用户上传目录（预留）
```

---

## Quick Start

### 环境要求

- Python 3.10+
- [Ollama](https://ollama.com)
- 推荐 16 GB+ 内存

### 1. 安装并启动 Ollama

```bash
ollama serve
```

首次运行前下载模型：

```bash
ollama pull mxbai-embed-large
ollama pull qwen3:8b
```

验证服务：

```bash
curl http://localhost:11434/api/tags
```

### 2. 安装依赖

```bash
cd basic-rag
uv sync
```

### 3. 启动应用

```bash
uv run streamlit run app.py
```

浏览器访问 <http://localhost:8501>。

### 4. 使用

1. 侧边栏上传 PDF（已内置 `docs/sample.pdf` 可供测试）
2. 点击「上传并处理」，等待入库完成
3. 主页面输入与文档相关的问题
4. 点击「提交查询」，查看回答

> 首次查询需要加载模型，耗时略长；Chat 模型默认缓存 30 分钟。

---

## Testing

```bash
uv run python -m pytest -q
```

覆盖范围：PDF 文本提取与 OCR、文档切分参数、Chroma 写入与进度回调、Retriever 检索、RAG 链路输出、Streamlit 交互与临时文件清理。

---

## Configuration

所有参数集中在 `config.py`：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `EMBEDDING_MODEL` | Embedding 模型 | `mxbai-embed-large` |
| `CHAT_MODEL` | 对话模型 | `qwen3:8b` |
| `CHUNK_SIZE` | 文本切块大小 | `500` |
| `CHUNK_OVERLAP` | 切块重叠长度 | `50` |
| `RETRIEVER_K` | 检索返回片段数 | `5` |
| `DB_WRITE_BATCH_SIZE` | 向量库批量写入数 | `10` |
| `CHROMA_PERSIST_DIR` | Chroma 持久化目录 | `chroma_db/` |

---

## Limitations

- 仅支持带文本层的 PDF，扫描版需先通过外部 OCR 处理
- 当前检索策略为固定 Top-K 相似度检索，不支持混合检索
- 单用户本地应用，不含认证、权限和多租户隔离
- 不含对话历史，每次问答独立处理

---

## Roadmap

- [ ] OCR 管线集成，支持扫描件和图片型 PDF
- [ ] 引用来源、页码展示和答案可追溯
- [ ] 混合检索（BM25 + 向量）+ 重排序
- [ ] 文档去重、删除和知识库管理
- [ ] 多轮对话历史与检索策略联动
- [ ] Docker 部署与 CI/CD
- [ ] 演进为 Corrective RAG / Agentic RAG
