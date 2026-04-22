# 🎬 电影知识图谱智能问答系统 (KGQA)

基于 Neo4j + BERT + FastAPI + Streamlit 的电影领域知识图谱问答系统，实现自然语言问句到图谱查询再到答案生成的完整流程。

## 📋 项目结构

```
kgqa_movie/
├── movies_data.csv           # 电影数据集（60条）
├── data_preprocess.py        # 数据预处理脚本
├── neo4j_import.py           # Neo4j图谱导入脚本
├── intent_model_train.py     # 意图识别模型训练
├── core_modules.py           # 五大核心模块封装
├── main_api.py               # FastAPI后端接口
├── app.py                    # Streamlit前端界面
├── test_llm_api.py           # 大模型API测试
├── requirements.txt          # Python依赖包
└── README.md                 # 项目说明文档
```

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    用户交互层 (Streamlit)                      │
├─────────────────────────────────────────────────────────────┤
│                    后端服务层 (FastAPI)                        │
├─────────────────────────────────────────────────────────────┤
│  意图识别  →  实体链接  →  Cypher生成  →  答案生成              │
│  (BERT)      (词典+BERT)   (模板/Seq2Seq)  (规则/大模型)        │
├─────────────────────────────────────────────────────────────┤
│                    数据存储层 (Neo4j)                          │
├─────────────────────────────────────────────────────────────┤
│                 大模型能力层 (ChatGLM/通义千问)                 │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv kgqa_env

# 激活虚拟环境（Windows）
kgqa_env\Scripts\activate
# 激活虚拟环境（macOS/Linux）
source kgqa_env/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 安装并启动 Neo4j

1. 下载 Neo4j Desktop 5.x：https://neo4j.com/download
2. 创建项目 `KGQA_Movie`，添加本地数据库 `movie_kg`
3. 设置密码为 `123456`（与代码配置一致）
4. 启动数据库，访问 http://localhost:7474 验证

### 3. 导入数据到 Neo4j

```bash
# 预处理数据
python data_preprocess.py

# 将 movies_data.csv 复制到 Neo4j 的 import 文件夹
# 然后运行导入脚本
python neo4j_import.py
```

### 4. 训练意图识别模型（可选）

```bash
python intent_model_train.py
```

> 若跳过此步骤，core_modules.py 中的模型将以随机权重运行，意图识别效果会下降。

### 5. 启动后端服务

```bash
uvicorn main_api:app --reload
```

访问 http://localhost:8000/docs 查看 API 文档。

### 6. 启动前端界面

```bash
# 新开一个终端，激活虚拟环境
streamlit run app.py
```

访问 http://localhost:8501 使用问答系统。

## 💬 测试用例

| 问题 | 预期意图 | 预期答案 |
|------|---------|---------|
| 流浪地球的导演是谁？ | query_director | 郭帆 |
| 周星驰主演过哪些电影？ | query_actor | 功夫、大话西游、少林足球... |
| 我不是药神的评分是多少？ | query_rating | 9.0分 |
| 泰坦尼克号是哪一年上映的？ | query_year | 1997年 |
| 科幻电影有哪些？ | query_genre | 流浪地球、阿凡达、盗梦空间... |

## 🔧 核心模块说明

### 1. 意图识别模块 (IntentRecognizer)
- **模型**：BERT-base-chinese + Softmax分类
- **标签**：query_director / query_actor / query_rating / query_year / query_genre
- **输入**：用户自然语言问句
- **输出**：意图标签 + 置信度

### 2. 实体链接模块 (EntityLinker)
- **策略**：词典精确匹配 + BERT语义相似度消歧
- **实体类型**：Movie（电影）、Person（人物）、Genre（类型）
- **输入**：用户问句
- **输出**：主要实体 + 实体类型 + 全部实体

### 3. Cypher生成模块 (CypherGenerator)
- **模板法**：意图 + 实体类型 → 预定义Cypher模板
- **支持查询**：导演、演员、评分、年份、类型
- **输入**：意图标签 + 实体名称 + 实体类型
- **输出**：Cypher查询语句

### 4. Neo4j交互模块 (Neo4jSession)
- **功能**：连接管理、查询执行、结果格式化、路径查询、统计信息
- **输入**：Cypher语句
- **输出**：结构化查询结果

### 5. 答案生成模块 (AnswerGenerator)
- **规则模式**：基于模板快速生成答案（保底策略）
- **大模型模式**：调用 ChatGLM/通义千问 生成自然语言答案
- **输入**：原始问题 + 意图 + 实体 + 图谱结果
- **输出**：自然语言答案

## 📡 API接口文档

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 服务状态 |
| `/health` | GET | 健康检查 |
| `/kgqa` | GET/POST | 主问答接口 |
| `/query` | POST | 直接执行Cypher |
| `/entity/search` | GET | 实体搜索 |
| `/graph/path` | GET | 实体间路径查询 |
| `/graph/stats` | GET | 图谱统计 |
| `/chat/history` | GET | 对话历史 |
| `/chat/history` | DELETE | 清空对话 |

## 🔌 大模型API配置

在 `core_modules.py` 中配置你的 API 密钥：

```python
# ChatGLM
LLM_API_KEY = "your-api-key-here"
LLM_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

# 或通过环境变量设置
export LLM_API_KEY="your-api-key"
export LLM_URL="https://open.bigmodel.cn/api/paas/v4/chat/completions"
```

## 🌟 扩展功能

### 多轮对话
系统自动维护对话上下文，支持指代消解（如 "它的评分是多少？" 中的 "它"）。

### 推理可视化
通过 `/graph/path` 接口获取实体间的路径关系，用于前端图谱展示。

### 多跳推理
示例：查找与某演员合作过的所有导演
```cypher
MATCH (a:Person)-[:ACTED_IN]->(m:Movie)<-[:DIRECTED]-(d:Person)
WHERE a.name = "吴京"
RETURN DISTINCT d.name
```

## 🐛 常见问题

| 问题 | 解决方案 |
|------|---------|
| Neo4j连接失败 | 检查服务是否启动、端口7687、账号密码是否正确 |
| 实体识别错误 | 扩充实体词典、降低BERT相似度阈值 |
| Cypher语法错误 | 检查生成的Cypher语句，使用参数化查询 |
| API调用失败 | 检查API Key、额度、请求格式、网络连接 |
| 前后端不通 | 确认跨域配置、端口是否开放 |

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

- **图数据库**：Neo4j
- **深度学习**：PyTorch、Transformers (BERT)
- **后端框架**：FastAPI、Uvicorn
- **前端框架**：Streamlit
- **大模型**：ChatGLM / 通义千问 / 文心一言

## 📄 License

本实训项目仅供学习参考使用。

## 👨‍💻 作者

知识图谱应用开发实训项目

---

💡 **提示**：首次运行前，请确保已完成 Neo4j 安装、数据导入、依赖安装等环境准备工作。
