# 🎬 电影知识图谱智能问答系统 (KGQA)

基于 Neo4j + BERT + FastAPI + Streamlit 的电影领域知识图谱问答系统，实现自然语言问句到图谱查询再到答案生成的完整流程。支持意图识别、实体链接、Cypher 参数化查询、多轮对话指代消解、图谱可视化等能力。

## 📋 项目结构

```
kgqa_movie/
├── movies_data.csv           # 电影数据集（60条，含7个字段）
├── data_tools/               # 数据采集、预处理、图谱导入脚本
│   ├── data_preprocess.py    # 数据预处理（去重、去空、格式标准化）
│   ├── neo4j_import.py       # Neo4j图谱批量导入（Movie/Person/Genre节点及关系）
│   └── douban_crawler.py     # 豆瓣公开电影数据采集脚本
├── tests/                    # 自动化测试与连通性检查脚本
│   ├── test_config.py        # LLM配置读取测试
│   ├── test_douban_crawler.py # 豆瓣采集解析测试
│   └── test_llm_api.py       # 大模型API连通性测试脚本
├── intent_model_train.py     # BERT意图分类模型训练脚本
├── intent_model.pth          # 已训练的BERT意图模型权重
├── core_modules.py           # 六大核心模块封装（意图+实体+查询+答案+对话）
├── main_api.py               # FastAPI后端接口（8个RESTful端点）
├── app.py                    # Streamlit前端界面（含ECharts图谱可视化）
├── config.py                 # 环境变量和本地 .env 配置读取
├── requirements.txt          # Python依赖包（精确版本号）
├── DEPLOYMENT.md             # 跨平台部署指南（Arch Linux / Windows 10）
└── README.md                 # 项目说明文档
```

## 🏗️ 系统架构

```
┌───────────────────────────────────────────────────────────────┐
│                    用户交互层 (Streamlit)                     │
│              多轮对话 / 推理可视化 / 知识图谱子图浏览         │
├───────────────────────────────────────────────────────────────┤
│                    后端服务层 (FastAPI)                       │
├───────────────────────────────────────────────────────────────┤
│  意图识别  →  实体链接  →  Cypher生成  →  答案生成            │
│  (BERT)      (别名+精确+模糊)  (参数化模板)  (规则/DeepSeek)  │
│                                    ↕                          │
│                            多轮对话管理 (指代消解 + 实体记忆) │
├───────────────────────────────────────────────────────────────┤
│                    数据存储层 (Neo4j 5.x)                     │
├───────────────────────────────────────────────────────────────┤
│                 大模型能力层 (DeepSeek / ChatGLM)             │
└───────────────────────────────────────────────────────────────┘
```

## 🚀 快速开始

### 1. 环境准备

**Python 版本要求：3.8 ~ 3.10**

本项目已使用 `uv` 在仓库根目录创建虚拟环境 `.venv`。后续不要重复创建普通 `venv`，直接激活并安装/刷新依赖即可：

```bash
# 激活 uv 管理的虚拟环境（Linux/macOS）
source .venv/bin/activate

# 安装或刷新依赖
uv pip install -r requirements.txt
```

Windows PowerShell 可使用 `.venv\Scripts\Activate.ps1` 激活环境。也可以不激活环境，直接在命令前添加 `uv run`。

### 2. 安装并启动 Neo4j

> 项目使用 Neo4j 5.x，连接协议为 `neo4j://`。

**方式 A：Docker 部署（推荐）**

```bash
docker run -d \
    --name neo4j-kgqa \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/20040121 \
    -e NEO4J_dbms_memory_heap_max__size=1G \
    -v $(pwd)/neo4j_data:/data \
    -v $(pwd):/var/lib/neo4j/import \
    neo4j:5
```

**方式 B：Neo4j Desktop**

1. 下载 Neo4j Desktop 5.x：https://neo4j.com/download
2. 创建项目 `KGQA_Movie`，添加本地数据库 `movie_kg`
3. 设置密码为 `20040121`（与 `core_modules.py` 配置一致）
4. 启动数据库，访问 http://localhost:7474 验证

### 3. 导入数据到 Neo4j

```bash
# 预处理数据（去重、去空、格式标准化）
uv run python data_tools/data_preprocess.py

# 构建知识图谱（创建节点和关系）
uv run python data_tools/neo4j_import.py
```

> 使用 Docker 方式时项目目录已挂载到容器的 `/var/lib/neo4j/import/`，无需手动复制文件。

### 4. 训练意图识别模型（可选）

```bash
uv run python intent_model_train.py
```

> 模型将保存为 `intent_model.pth`。若跳过此步骤，`core_modules.py` 将以规则匹配方式运行（关键词 + 长匹配权重），仍可正常工作但准确率略低于 BERT 模型。

### 5. 启动后端服务

```bash
uv run uvicorn main_api:app --reload --host 0.0.0.0 --port 8000
```

访问 http://localhost:8000/docs 查看 Swagger API 文档。

### 6. 启动前端界面

```bash
# 新开一个终端
uv run streamlit run app.py --server.port 8501
```

访问 http://localhost:8501 使用问答系统。

> 详见 [`DEPLOYMENT.md`](./DEPLOYMENT.md) 获取 Arch Linux + Hyprland / Windows 10 的完整部署指南，含 Docker 配置、Hyprland 工作区布局、systemd 服务化等。

## ✅ 自动化测试

```bash
uv run python -m unittest discover -s tests -p 'test*.py'
```

大模型 API 连通性检查：

```bash
uv run python tests/test_llm_api.py
```

## 💬 问答测试用例

| 问题 | 预期意图 | 预期答案 |
|------|---------|---------|
| 流浪地球的导演是谁？ | query_director | 郭帆 |
| 周星驰主演过哪些电影？ | query_actor | 功夫、大话西游、少林足球... |
| 我不是药神的评分是多少？ | query_rating | 9.0分 |
| 泰坦尼克号是哪一年上映的？ | query_year | 1997年 |
| 科幻电影有哪些？ | query_genre | 流浪地球、阿凡达、盗梦空间... |

## 🔧 核心模块说明

系统包含 **六大核心模块**，封装在 `core_modules.py` 中：

### 1. 意图识别模块 (IntentRecognizer)
- **模型**：BERT-base-chinese + Softmax分类（5意图）
- **兜底策略**：BERT置信度 <0.55 时自动回退到关键词规则匹配（按长短词权重计分）
- **标签**：query_director / query_actor / query_rating / query_year / query_genre
- **输入**：用户自然语言问句
- **输出**：意图标签 + 置信度

### 2. 实体链接模块 (EntityLinker)
- **策略**：别名映射 → 精确最长匹配 → difflib 轻量初筛 → BERT向量精排 → 意图联合推理
- **优化**：启动时批量预计算所有实体BERT向量，O(N)次编码降为O(1)
- **实体类型**：Movie（电影）、Person（人物）、Genre（类型）
- **输入**：用户问句 + 意图线索
- **输出**：主要实体 + 实体类型 + 全部实体

### 3. Cypher生成模块 (CypherGenerator)
- **模板法**：意图 + 实体类型 → 预定义Cypher模板，**参数化查询防止注入**
- **高级模式**：支持多跳查询（如"与某演员合作过的导演"）
- **支持查询**：导演、演员、评分、年份、类型、高分推荐
- **输入**：意图标签 + 实体名称 + 实体类型
- **输出**：Cypher语句 + 参数字典

### 4. Neo4j交互模块 (Neo4jSession)
- **功能**：连接管理、参数化查询执行、结果格式化、路径查询、统计信息、邻居子图
- **协议**：Neo4j 5.x `neo4j://` 协议
- **输入**：Cypher语句 + 参数
- **输出**：结构化查询结果 / 路径数据 / 子图谱数据

### 5. 答案生成模块 (AnswerGenerator)
- **规则模式**：基于意图模板快速生成答案（无需API、零延迟的保底策略）
- **大模型模式**：调用 DeepSeek API 生成自然语言答案
- **输入**：原始问题 + 意图 + 实体 + 图谱结果
- **输出**：自然语言答案 + 来源标记（rule/llm）

### 6. 多轮对话管理 (ConversationManager)
- **指代消解**：自动识别"它"、"这部电影"等代词，替换为当前实体
- **实体记忆**：维护最近实体，支持上下文继承
- **推荐问题**：基于当前实体自动生成追问建议
- **历史管理**：维护最近5轮对话上下文

## 📡 API接口文档

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 服务状态检查 |
| `/health` | GET | 健康检查（含Neo4j连接状态和图谱统计） |
| `/kgqa` | GET/POST | 主问答接口（完整Pipeline） |
| `/query` | POST | 直接执行Cypher参数化查询（高级接口） |
| `/entity/search` | GET | 实体搜索（电影/人物模糊搜索） |
| `/graph/path` | GET | 实体间最短路径查询 |
| `/graph/stats` | GET | 图谱统计信息（节点/关系数量） |
| `/graph/neighbors` | GET | 实体邻居子图查询（用于前端图谱可视化） |
| `/chat/history` | GET | 获取多轮对话历史 |
| `/chat/history` | DELETE | 清空对话历史 |

## 🔌 大模型API配置

本项目默认使用 **DeepSeek** 大模型生成自然语言答案（未配置时自动回退到规则模板）。

通过环境变量或本地 `.env` 文件配置 API 密钥，避免把真实密钥写入代码：

```bash
export LLM_API_KEY="your-deepseek-api-key"
export LLM_URL="https://api.deepseek.com/chat/completions"
export LLM_MODEL="deepseek-chat"
```

也可以创建本地 `.env` 文件（已被 `.gitignore` 忽略）：

```bash
cp .env.example .env
# 然后只在本地 .env 中填写真实 LLM_API_KEY
```

配置完成后可运行连通性测试：

```bash
uv run python tests/test_llm_api.py
```

> 兼容其他 OpenAI 兼容 API（如 ChatGLM、通义千问等），只需替换 `LLM_URL` 和 `LLM_MODEL`。

## 🕷️ 豆瓣数据采集

可使用 `data_tools/douban_crawler.py` 采集豆瓣公开电影页面中的中国电影、导演、演员、类型、年份、评分等信息，用于后续扩充图数据库。

```bash
uv run python data_tools/douban_crawler.py \
  --movie-limit 100 \
  --actor-limit 100 \
  --output-dir data/douban
```

默认输出：

- `data/douban/movies.json`：电影完整信息
- `data/douban/actors.json`：演员信息
- `data/douban/movies_data_douban.csv`：兼容当前 `movies_data.csv` 字段的图谱导入数据
- `data/douban/crawl_report.json`：采集统计

如果默认公开列表页无法收集到足够多的中国电影，可准备一个种子文件，每行一个豆瓣电影详情页 URL：

```bash
uv run python data_tools/douban_crawler.py --seed-url-file douban_seed_urls.txt
```

脚本会限速、缓存 HTML，并遵守豆瓣 robots 规则；不会访问 `/search`、`/subject_search`、`/celebrities/search`、`/j/` 等搜索/API 路径，也不会处理登录、验证码或反爬绕过。

## 🌟 前端功能

### 多轮对话
系统自动维护对话上下文，支持指代消解（如 "它的评分是多少？" 中的 "它" 会自动替换为上一轮实体），并基于当前实体推荐追问。

### 推理过程可视化
每次回答均可展开查看完整推理链路：意图识别（含置信度进度条）→ 实体链接 → Cypher 查询语句 → 图谱原始结果 → 答案来源（大模型/规则模板）。

### 知识图谱可视化
基于 ECharts 的力导向图，支持输入实体名称探索其邻居子图（电影、人物、类型节点及关系），支持拖拽、缩放、悬停查看详情。

### 意图标签着色
五种意图类型以不同颜色标签展示（导演蓝、演员紫、评分橙、年份绿、类型粉），清晰区分查询意图。

### 多跳推理
示例：查找与某演员合作过的所有导演
```cypher
MATCH (a:Person {name: $entity})-[:ACTED_IN]->(m:Movie)<-[:DIRECTED]-(d:Person)
RETURN DISTINCT d.name AS director, m.title AS movie
```

## 🐛 常见问题

| 问题 | 解决方案 |
|------|---------|
| Neo4j连接失败 | 检查服务是否启动、端口7687、账号密码是否为 `neo4j/20040121`、协议是否为 `neo4j://` |
| 实体识别错误 | 扩充实体词典、降低BERT相似度阈值、添加别名映射 |
| Cypher语法错误 | 检查生成的Cypher语句，确认使用参数化查询 |
| 大模型API调用失败 | 检查API Key、额度、请求格式、网络连接（未配置时将自动回退到规则模板） |
| 前后端不通 | 确认跨域配置、端口是否开放、API地址是否一致 |
| BERT模型下载慢 | 设置 `HF_ENDPOINT=https://hf-mirror.com` 使用镜像 |
| pandas/numpy版本冲突 | 使用 `uv pip install -r requirements.txt` 按精确版本号安装 |

## 📝 评分标准参考

| 项目 | 分值 | 要求 |
|------|------|------|
| 环境部署 | 10分 | 无报错运行 |
| 图谱构建 | 20分 | 节点、关系完整 |
| 模块开发 | 30分 | 五大模块功能正常 |
| 系统整合 | 20分 | 前后端可交互、稳定 |
| 实训报告 | 20分 | 文档完整规范 |
| 扩展功能 | +10分 | 多轮对话、可视化等 |

## 📚 技术栈

- **图数据库**：Neo4j 5.x (`neo4j://` 协议)
- **深度学习**：PyTorch 1.13、Transformers 4.30 (BERT-base-chinese)
- **后端框架**：FastAPI 0.103、Uvicorn 0.23
- **前端框架**：Streamlit 1.24（含 ECharts 图谱可视化）
- **大模型**：DeepSeek（兼容 OpenAI Chat Completions API）
- **数据科学**：pandas 2.0、scikit-learn 1.3

## 📄 License

本实训项目仅供学习参考使用。

## 👨‍💻 作者
freebird

知识图谱应用开发实训项目

---

💡 **提示**：首次运行前，请确保已完成 Neo4j 安装、数据导入、依赖安装等环境准备工作。
