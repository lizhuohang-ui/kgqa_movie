# ==============================
# KGQA 五大核心模块统一封装
# 1.意图识别 2.实体链接 3.Cypher生成 4.Neo4j交互 5.答案生成
# ==============================

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from neo4j import GraphDatabase
import requests
import json
import re
import os
from config import get_llm_config, has_llm_api_key

# ======================
# 全局配置
# ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "bert-base-chinese"

# 初始化tokenizer（全局复用）
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# Neo4j连接信息（5.x 推荐 neo4j:// 协议）
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PWD = "20040121"

# 大模型API配置（从环境变量或本地 .env 读取，避免硬编码密钥）
LLM_CONFIG = get_llm_config()
LLM_API_KEY = LLM_CONFIG.api_key
LLM_URL = LLM_CONFIG.url
LLM_MODEL = LLM_CONFIG.model

# ======================
# 3.1 意图识别模块
# ======================
intent_map = {
    0: "query_director",   # 查询导演
    1: "query_actor",      # 查询演员
    2: "query_rating",     # 查询评分
    3: "query_year",       # 查询年份
    4: "query_genre"       # 查询类型
}

class IntentModel(nn.Module):
    """基于BERT的意图分类模型"""
    def __init__(self, num_classes=5):
        super(IntentModel, self).__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = self.dropout(outputs.pooler_output)
        return self.classifier(x)


class IntentRecognizer:
    """
    意图识别器（BERT模型 + 规则兜底）
    策略：优先使用BERT模型预测，低置信度或模型未加载时回退到关键词规则匹配
    """
    def __init__(self, model_path="intent_model.pth"):
        self.model = IntentModel().to(DEVICE)
        self.model_loaded = False
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            self.model_loaded = True
            print(f"✅ 意图模型已加载: {model_path}")
        else:
            print(f"⚠️ 模型文件不存在: {model_path}，将使用规则匹配兜底")
        self.model.eval()

        # 意图关键词规则（兜底策略）—— 按匹配权重排序
        self.intent_keywords = {
            "query_director": [
                "导演", "执导", "拍的", "谁拍", "谁导", "导演是谁", "谁导演",
                "filmmaker", " filmmaker", "幕后", "主创", "总导演", "执导者",
                "导戏", "导的", "执导的", "拍这部电影"
            ],
            "query_actor": [
                "演员", "主演", "出演", "演过", "谁演", "演的", "表演",
                "演员表", "阵容", "参演", "扮演", "角色", "领衔", "联袂",
                "卡司", "cast", "出演者", "饰演", "男一号", "女一号",
                "男主角", "女主角", "扮演者", "表演者", "客串", "友情出演"
            ],
            "query_rating": [
                "评分", "多少分", "打分", "评价", "口碑", "得分", "几分",
                "评分多少", "分多少", "rating", "豆瓣", "imdb", "星",
                "烂番茄", "猫眼", "淘票票", "好评率", "推荐度", "值得看",
                "观感", "影迷评价", "网友评分", "星评", "几颗星", "评价分数"
            ],
            "query_year": [
                "年份", "哪年", "什么时候", "上映", "年代", "出品", "发行",
                "几几年", "时间", "日期", "首映", "何时", "档期", "公映",
                "院线", "上线", "播出", "首播", "重映", "拍摄", "开机",
                "杀青", "后期", "立项", "备案", "问世", "诞生", "出品时间"
            ],
            "query_genre": [
                "类型", "什么类型", "题材", "风格", "哪种", "分类", "种类",
                "片种", "genre", "类目", "流派", "划分", "归属", "标签",
                "体裁", "片类", "影片类型", "电影种类", "什么片", "算哪种",
                "哪类影片", "电影题材"
            ]
        }

    def _rule_recognize(self, question):
        """基于关键词的规则意图识别（兜底策略）"""
        question_lower = question.lower()
        scores = {}
        for intent, keywords in self.intent_keywords.items():
            # 长关键词权重2，单字关键词权重1
            score = sum(2 if len(kw) >= 2 and kw in question_lower else (1 if kw in question_lower else 0)
                        for kw in keywords)
            scores[intent] = score

        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]

        # 如果所有意图都没匹配到，默认返回 query_director（兜底但置信度很低）
        if best_score == 0:
            return "query_director", 0.25

        # 规则匹配的置信度与分数挂钩，但不超过0.82
        confidence = min(0.45 + 0.06 * best_score, 0.82)
        return best_intent, round(confidence, 4)

    def recognize(self, question):
        """
        识别用户问句的意图
        策略：优先BERT模型，低置信度或模型未加载时回退到规则匹配
        """
        # 1. 尝试BERT模型预测
        if self.model_loaded:
            inputs = tokenizer(
                question,
                padding="max_length",
                max_length=64,
                truncation=True,
                return_tensors="pt"
            )
            input_ids = inputs["input_ids"].to(DEVICE)
            attention_mask = inputs["attention_mask"].to(DEVICE)

            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
                probs = torch.softmax(outputs, dim=1)
                idx = torch.argmax(outputs, dim=1).item()
                confidence = probs[0][idx].item()

            # 置信度足够高（>=0.55），直接返回模型预测
            if confidence >= 0.55:
                return intent_map[idx], round(confidence, 4)

        # 2. 规则兜底
        intent, confidence = self._rule_recognize(question)
        return intent, confidence


def intent_recognition(question):
    """
    意图识别函数（便捷调用）
    自动加载模型并进行意图分类
    """
    recognizer = IntentRecognizer()
    return recognizer.recognize(question)


# ======================
# 3.2 实体链接模块
# ======================
class EntityLinker:
    """
    实体链接模块（改进版）
    策略：别名映射 → 精确最长匹配 → difflib轻量初筛 → BERT向量精排 → 意图联合推理
    """

    def __init__(self, neo4j_uri=NEO4J_URI, neo4j_user=NEO4J_USER, neo4j_pwd=NEO4J_PWD):
        # 连接Neo4j获取实体词典
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pwd))
        self.movie_list = self._load_movies()
        self.person_list = self._load_persons()
        self.genre_list = self._load_genres()

        # 别名映射（昵称、简称 → 标准名）
        self.alias_map = {
            "星爷": "周星驰",
            "周星星": "周星驰",
            "国师": "张艺谋",
            "老谋子": "张艺谋",
            "冯导": "冯小刚",
            "姜文": "姜文",
            "卡梅隆": "卡梅隆",
            "诺兰": "诺兰",
            "宫崎骏": "宫崎骏",
            "新海诚": "新海诚",
            "奉俊昊": "奉俊昊",
            "延相昊": "延相昊",
        }

        # 按长度降序排序，确保最长匹配优先（避免子串误匹配）
        self.movie_list_sorted = sorted(set(self.movie_list), key=len, reverse=True)
        self.person_list_sorted = sorted(set(self.person_list), key=len, reverse=True)
        self.genre_list_sorted = sorted(set(self.genre_list), key=len, reverse=True)

        # 加载BERT用于语义相似度计算
        self.bert = BertModel.from_pretrained(MODEL_NAME).to(DEVICE)
        self.bert.eval()

        # 预计算所有实体的BERT向量（避免每次重复编码）
        self._embeddings = {}
        self._precompute_embeddings()

        print(f"📚 实体词典加载完成:")
        print(f"   电影: {len(self.movie_list_sorted)} 部")
        print(f"   人物: {len(self.person_list_sorted)} 人")
        print(f"   类型: {len(self.genre_list_sorted)} 种")
        print(f"   预计算向量: {len(self._embeddings)} 个")

    def _load_movies(self):
        """从Neo4j加载电影列表"""
        with self.driver.session() as session:
            result = session.run("MATCH (m:Movie) RETURN DISTINCT m.title AS title")
            return [record["title"] for record in result]

    def _load_persons(self):
        """从Neo4j加载人物列表"""
        with self.driver.session() as session:
            result = session.run("MATCH (p:Person) RETURN DISTINCT p.name AS name")
            return [record["name"] for record in result]

    def _load_genres(self):
        """从Neo4j加载类型列表"""
        with self.driver.session() as session:
            result = session.run("MATCH (g:Genre) RETURN DISTINCT g.name AS name")
            return [record["name"] for record in result]

    def _precompute_embeddings(self):
        """批量预计算所有实体的BERT向量，大幅提升模糊匹配速度"""
        all_entities = list(set(self.movie_list + self.person_list + self.genre_list))
        batch_size = 32
        for i in range(0, len(all_entities), batch_size):
            batch = all_entities[i:i+batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                max_length=32,
                truncation=True,
                return_tensors="pt"
            )
            input_ids = inputs["input_ids"].to(DEVICE)
            attention_mask = inputs["attention_mask"].to(DEVICE)

            with torch.no_grad():
                outputs = self.bert(input_ids, attention_mask)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            for j, ent in enumerate(batch):
                self._embeddings[ent] = embeddings[j].reshape(1, -1)

    def _get_embedding(self, text):
        """获取文本的BERT向量表示"""
        inputs = tokenizer(
            text,
            padding="max_length",
            max_length=32,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs["input_ids"].to(DEVICE)
        attention_mask = inputs["attention_mask"].to(DEVICE)

        with torch.no_grad():
            outputs = self.bert(input_ids, attention_mask)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embedding

    def _clean_question(self, question):
        """清洗问句，去除疑问词和无关成分"""
        patterns = [
            r"[的了吗呢啊吧]?[\?？]",
            r"^(请问|我想知道|告诉我|请问一下|我想问|请问您|你知道|请问谁)",
            r"(有谁|有哪些|有什么|是什么|分别是|分别是哪些|分别是啥)",
            r"(分别|各自|各自都)",
        ]
        cleaned = question
        for p in patterns:
            cleaned = re.sub(p, "", cleaned)
        return cleaned.strip()

    def _fuzzy_match(self, text, candidates, threshold=0.70, top_k=5):
        """
        模糊匹配：先用difflib快速初筛，再用预计算BERT向量精排
        比原始O(N)次BERT编码快数十倍
        """
        if not candidates:
            return None, 0.0

        # 候选较多时，先用difflib轻量初筛（纯Python，极快）
        if len(candidates) > 15:
            import difflib
            close_matches = difflib.get_close_matches(text, candidates, n=top_k, cutoff=0.25)
            if close_matches:
                candidates = close_matches
            else:
                # 放宽初筛条件再试一次
                close_matches = difflib.get_close_matches(text, candidates, n=top_k, cutoff=0.15)
                if not close_matches:
                    return None, 0.0
                candidates = close_matches

        # BERT向量精排（使用预计算向量，只需编码问句一次）
        text_emb = self._get_embedding(text)
        best_sim = 0.0
        best_match = None

        for cand in candidates:
            cand_emb = self._embeddings.get(cand)
            if cand_emb is not None:
                sim = cosine_similarity(text_emb, cand_emb)[0][0]
                if sim > best_sim and sim > threshold:
                    best_sim = sim
                    best_match = cand

        return best_match, best_sim

    def extract_entities(self, question):
        """
        从问句中提取实体（支持多实体识别）
        策略：别名替换 → 精确最长匹配 → 轻量模糊匹配
        """
        original_question = question
        entities = {
            "movie": [],
            "person": [],
            "genre": [],
            "all": []
        }

        # 1. 别名替换（将昵称转为标准名）
        for alias, standard in self.alias_map.items():
            if alias in question:
                question = question.replace(alias, standard)

        # 2. 精确匹配（最长匹配优先，避免子串误匹配）
        # 电影：按长度降序，只取第一个匹配（避免一部电影被多次匹配）
        for movie in self.movie_list_sorted:
            if movie in question:
                entities["movie"].append(movie)
                entities["all"].append(("movie", movie))
                break

        # 人物：支持多个人物匹配（但去重）
        matched_persons = set()
        for person in self.person_list_sorted:
            if person in question and person not in matched_persons:
                entities["person"].append(person)
                entities["all"].append(("person", person))
                matched_persons.add(person)

        # 类型：只取第一个
        for genre in self.genre_list_sorted:
            if genre in question:
                entities["genre"].append(genre)
                entities["all"].append(("genre", genre))
                break

        # 3. 模糊匹配（精确匹配失败时使用）
        cleaned_q = self._clean_question(question)

        if not entities["movie"]:
            best_movie, sim = self._fuzzy_match(cleaned_q, self.movie_list_sorted, threshold=0.68)
            if best_movie:
                entities["movie"].append(best_movie)
                entities["all"].append(("movie", best_movie, round(sim, 4)))

        if not entities["person"]:
            best_person, sim = self._fuzzy_match(cleaned_q, self.person_list_sorted, threshold=0.68)
            if best_person:
                entities["person"].append(best_person)
                entities["all"].append(("person", best_person, round(sim, 4)))

        return entities, original_question

    def link(self, question, intent_hint=None):
        """
        实体链接主函数（支持意图联合推理）
        Args:
            question: 用户输入的问句
            intent_hint: 意图类型，用于指导实体优先级
        Returns:
            primary_entity: 主要实体（用于查询）
            entity_type: 实体类型
            all_entities: 所有识别到的实体
        """
        entities, original_question = self.extract_entities(question)

        # 根据意图类型调整实体优先级（意图联合推理）
        priority_map = {
            "query_director": ["person", "movie", "genre"],   # 查导演时，人名优先级高（查某人导演的作品）
            "query_actor":    ["person", "movie", "genre"],   # 查演员时，人名优先级高
            "query_rating":   ["movie", "person", "genre"],   # 查评分时，电影优先级高
            "query_year":     ["movie", "person", "genre"],   # 查年份时，电影优先级高
            "query_genre":    ["genre", "movie", "person"]    # 查类型时，类型优先级高
        }

        priority = priority_map.get(intent_hint, ["movie", "person", "genre"])

        primary_entity = None
        entity_type = None

        for etype in priority:
            if entities[etype]:
                primary_entity = entities[etype][0]  # 取第一个匹配
                entity_type = etype
                break

        return primary_entity, entity_type, entities


# 便捷函数：实体链接
def entity_linking(question, intent_hint=None):
    """便捷调用实体链接"""
    linker = EntityLinker()
    return linker.link(question, intent_hint=intent_hint)


# ======================
# 3.3 Cypher生成模块
# ======================
class CypherGenerator:
    """
    Cypher查询语句生成器
    支持两种模式：
    1. 模板法（稳定、可解释，推荐）
    2. Seq2Seq生成法（灵活，进阶）
    """

    # 意图到Cypher模板的映射（使用参数化查询，防止Cypher注入）
    TEMPLATES = {
        "query_director": {
            "movie": "MATCH (m:Movie {title: $entity})<-[:DIRECTED]-(p:Person) RETURN p.name AS director",
            "person": "MATCH (p:Person {name: $entity})-[:DIRECTED]->(m:Movie) RETURN m.title AS movie"
        },
        "query_actor": {
            "movie": "MATCH (m:Movie {title: $entity})<-[:ACTED_IN]-(p:Person) RETURN p.name AS actor",
            "person": "MATCH (p:Person {name: $entity})-[:ACTED_IN]->(m:Movie) RETURN m.title AS movie"
        },
        "query_rating": {
            "movie": "MATCH (m:Movie {title: $entity}) RETURN m.rating AS rating"
        },
        "query_year": {
            "movie": "MATCH (m:Movie {title: $entity}) RETURN m.year AS year"
        },
        "query_genre": {
            "movie": "MATCH (m:Movie {title: $entity}) RETURN m.genre AS genre",
            "genre": "MATCH (m:Movie)-[:IS_GENRE]->(g:Genre {name: $entity}) RETURN m.title AS movie"
        }
    }

    @staticmethod
    def generate_template(intent, entity, entity_type="movie"):
        """
        基于模板生成Cypher语句（返回参数化查询，避免注入风险）
        Args:
            intent: 意图标签，如 "query_director"
            entity: 实体名称
            entity_type: 实体类型
        Returns:
            cypher: Cypher查询语句
            parameters: 参数字典
        """
        if intent not in CypherGenerator.TEMPLATES:
            # 默认返回电影信息
            return "MATCH (m:Movie {title: $entity}) RETURN m", {"entity": entity}

        templates = CypherGenerator.TEMPLATES[intent]

        if entity_type in templates:
            cypher = templates[entity_type]
        else:
            # 使用默认模板（movie类型）
            cypher = templates.get("movie", "MATCH (m:Movie {title: $entity}) RETURN m")

        return cypher, {"entity": entity}

    @staticmethod
    def generate_advanced(intent, entity, entity_type="movie"):
        """
        高级查询生成（支持多条件、多跳，使用参数化查询）
        """
        # 多跳查询：查找演员和导演的合作关系
        if intent == "query_director" and entity_type == "person":
            cypher = """
                MATCH (a:Person {name: $entity})-[:ACTED_IN]->(m:Movie)<-[:DIRECTED]-(d:Person)
                RETURN DISTINCT d.name AS director, m.title AS movie
            """
            return cypher, {"entity": entity}

        # 高分电影推荐
        if intent == "query_rating" and entity_type == "genre":
            cypher = """
                MATCH (m:Movie)-[:IS_GENRE]->(g:Genre {name: $entity})
                WHERE toFloat(m.rating) >= 8.0
                RETURN m.title AS movie, m.rating AS rating
                ORDER BY toFloat(m.rating) DESC
            """
            return cypher, {"entity": entity}

        # 默认使用模板法
        return CypherGenerator.generate_template(intent, entity, entity_type)


def generate_cypher(intent, entity, entity_type="movie"):
    """便捷调用Cypher生成（返回语句和参数字典）"""
    generator = CypherGenerator()
    return generator.generate_template(intent, entity, entity_type)


# ======================
# 3.4 Neo4j交互模块
# ======================
class Neo4jSession:
    """
    Neo4j数据库交互封装
    功能：连接管理、查询执行、结果格式化、异常处理
    """

    def __init__(self, uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PWD):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        self._connect()

    def _connect(self):
        """建立数据库连接"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # 测试连接
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS num")
                record = result.single()
                if record and record["num"] == 1:
                    print("✅ Neo4j连接成功")
        except Exception as e:
            print(f"❌ Neo4j连接失败: {e}")
            raise

    def query(self, cypher, parameters=None):
        """
        执行Cypher查询
        Args:
            cypher: Cypher语句
            parameters: 参数化查询的参数
        Returns:
            results: 查询结果列表
        """
        if not self.driver:
            self._connect()

        try:
            with self.driver.session() as session:
                result = session.run(cypher, parameters)
                data = [record.data() for record in result]
                return data
        except Exception as e:
            print(f"❌ 查询执行失败: {e}")
            print(f"   Cypher: {cypher}")
            return []

    def query_with_path(self, cypher, parameters=None):
        """
        执行返回路径的查询（用于可视化）
        """
        if not self.driver:
            self._connect()

        try:
            with self.driver.session() as session:
                result = session.run(cypher, parameters)
                paths = []
                for record in result:
                    for key, value in record.items():
                        if hasattr(value, 'nodes'):  # Path对象
                            path_data = {
                                "nodes": [{"id": n.id, "labels": list(n.labels), "properties": dict(n)}
                                    for n in value.nodes],
                                "relationships": [{"id": r.id, "type": r.type,
                                                    "start": r.start_node.id, "end": r.end_node.id}
                                    for r in value.relationships]
                            }
                            paths.append(path_data)
                return paths
        except Exception as e:
            print(f"❌ 路径查询失败: {e}")
            return []

    def get_statistics(self):
        """获取图谱统计信息"""
        stats = {}
        try:
            # 节点统计（Neo4j 5.x 兼容语法）
            node_result = self.query("""
                MATCH (n)
                RETURN labels(n) AS label, count(n) AS count
            """)
            stats["nodes"] = {str(r["label"]): r["count"] for r in node_result}

            # 关系统计
            rel_result = self.query("""
                MATCH ()-[r]->() 
                RETURN type(r) AS type, count(r) AS count
            """)
            stats["relationships"] = {r["type"]: r["count"] for r in rel_result}

            return stats
        except Exception as e:
            print(f"❌ 统计失败: {e}")
            return {}

    def get_neighbors(self, entity, depth=1):
        """
        获取实体的邻居子图（用于可视化）
        Args:
            entity: 实体名称
            depth: 探索深度（默认1跳）
        Returns:
            dict: {nodes: [...], edges: [...]}
        """
        if not self.driver:
            self._connect()

        try:
            cypher = """
                MATCH (center)-[r]-(neighbor)
                WHERE (center:Person AND center.name = $entity)
                   OR (center:Movie AND center.title = $entity)
                   OR (center:Genre AND center.name = $entity)
                RETURN center, neighbor, r
                LIMIT 80
            """
            with self.driver.session() as session:
                result = session.run(cypher, {"entity": entity})

                nodes_map = {}
                edges = []

                for record in result:
                    center = record["center"]
                    neighbor = record["neighbor"]
                    rel = record["r"]

                    # 中心节点
                    center_id = center.get("name") or center.get("title")
                    center_label = list(center.labels)[0] if center.labels else "Unknown"
                    if center_id and center_id not in nodes_map:
                        nodes_map[center_id] = {
                            "id": center_id,
                            "label": center_label,
                            "name": center_id,
                            "properties": dict(center)
                        }

                    # 邻居节点
                    neighbor_id = neighbor.get("name") or neighbor.get("title")
                    neighbor_label = list(neighbor.labels)[0] if neighbor.labels else "Unknown"
                    if neighbor_id and neighbor_id not in nodes_map:
                        nodes_map[neighbor_id] = {
                            "id": neighbor_id,
                            "label": neighbor_label,
                            "name": neighbor_id,
                            "properties": dict(neighbor)
                        }

                    # 关系（去重）
                    if center_id and neighbor_id:
                        edge_id = f"{center_id}-{rel.type}-{neighbor_id}"
                        edges.append({
                            "source": center_id,
                            "target": neighbor_id,
                            "relation": rel.type
                        })

                return {"nodes": list(nodes_map.values()), "edges": edges}
        except Exception as e:
            print(f"❌ 邻居查询失败: {e}")
            return {"nodes": [], "edges": []}

    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()
            print("🔒 Neo4j连接已关闭")


# ======================
# 3.5 答案生成模块
# ======================
class AnswerGenerator:
    """
    答案生成模块
    功能：将结构化查询结果转换为自然语言答案
    策略：
    1. 规则模板（快速、稳定）
    2. 大模型生成（自然、灵活）
    """

    def __init__(self, api_key=None, api_url=None, model=None):
        self._api_key_override = api_key
        self._api_url_override = api_url
        self._model_override = model
        self._refresh_config()

    def _refresh_config(self):
        llm_config = get_llm_config(
            api_key=self._api_key_override,
            api_url=self._api_url_override,
            model=self._model_override,
        )
        self.api_key = llm_config.api_key
        self.api_url = llm_config.url
        self.model = llm_config.model
        self.use_llm = has_llm_api_key(self.api_key)

    def _rule_based(self, question, intent, entity, result):
        """
        基于规则的答案生成（保底策略）
        """
        if not result:
            return f"抱歉，未找到关于「{entity}」的相关信息。"

        values = []
        for record in result:
            values.extend(record.values())

        values = list(set(values))  # 去重
        answer = "，".join(str(v) for v in values if v)

        # 根据意图组织语言
        if intent == "query_director":
            if "actor" in result[0]:
                return f"「{entity}」的主演是：{answer}。"
            return f"「{entity}」的导演是：{answer}。"

        elif intent == "query_actor":
            if "movie" in result[0]:
                return f"「{entity}」主演过的电影有：{answer}。"
            return f"「{entity}」的主演是：{answer}。"

        elif intent == "query_rating":
            return f"「{entity}」的评分是：{answer}分。"

        elif intent == "query_year":
            return f"「{entity}」于{answer}年上映。"

        elif intent == "query_genre":
            if "movie" in result[0]:
                return f"「{entity}」类型的电影有：{answer}。"
            return f"「{entity}」是一部{answer}电影。"

        return f"关于「{entity}」的查询结果：{answer}"

    def _llm_generate(self, question, result, context=""):
        """
        调用大模型生成自然语言答案
        """
        # 构建提示词
        result_str = json.dumps(result, ensure_ascii=False)
        prompt = f"""你是一个专业的电影知识问答助手。请根据以下结构化数据，用自然语言回答用户的问题。

用户问题：{question}
图谱查询结果：{result_str}
{context}

要求：
1. 直接回答问题，不要重复问题
2. 如果结果为空，礼貌地告知未找到相关信息
3. 保持回答简洁、准确、口语化
4. 可以适当补充相关背景知识

请回答："""

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是一个电影知识图谱问答助手，擅长将结构化数据转化为自然语言。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 512
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=data, timeout=15)
            response.raise_for_status()
            result_data = response.json()
            answer = result_data["choices"][0]["message"]["content"]
            return answer.strip()
        except Exception as e:
            print(f"⚠️ 大模型调用失败: {e}，回退到规则模板")
            return None

    def generate(self, question, intent, entity, result, use_llm=True):
        """
        生成答案
        Args:
            question: 原始问题
            intent: 意图标签
            entity: 查询实体
            result: 图谱查询结果
            use_llm: 是否优先使用大模型
        Returns:
            answer: 自然语言答案
            source: 答案来源（rule/llm）
        """
        self._refresh_config()
        print("use_llm:", use_llm, "llm_configured:", self.use_llm)
        # 优先尝试大模型
        if use_llm and self.use_llm:
            llm_answer = self._llm_generate(question, result)
            if llm_answer:
                return llm_answer, "llm"

        # 回退到规则模板
        rule_answer = self._rule_based(question, intent, entity, result)
        return rule_answer, "rule"


# 便捷函数
def generate_answer(question, result, intent="", entity="", use_llm=True):
    """便捷调用答案生成"""
    generator = AnswerGenerator()
    return generator.generate(question, intent, entity, result, use_llm)


# ======================
# 多轮对话上下文管理
# ======================
class ConversationManager:
    """
    多轮对话上下文管理器
    功能：维护对话历史、指代消解、上下文继承
    """

    def __init__(self, max_history=5):
        self.history = []  # 历史记录 [(question, answer, intent, entity), ...]
        self.max_history = max_history
        self.current_entity = None  # 当前关注的实体

    def add_turn(self, question, answer, intent=None, entity=None):
        """添加一轮对话"""
        self.history.append({
            "question": question,
            "answer": answer,
            "intent": intent,
            "entity": entity
        })
        if entity:
            self.current_entity = entity

        # 限制历史长度
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_context(self):
        """获取上下文摘要"""
        if not self.history:
            return ""

        context_parts = []
        for turn in self.history[-3:]:  # 最近3轮
            q = turn["question"]
            a = turn["answer"]
            context_parts.append(f"Q: {q} A: {a}")

        return "\n".join(context_parts)

    def resolve_coreference(self, question):
        """
        指代消解
        将代词（它、这部电影、他等）替换为具体实体
        """
        if not self.current_entity:
            return question

        # 指代词替换
        pronouns = ["它", "这部电影", "这部片子", "该片", "他", "她"]
        for pronoun in pronouns:
            if pronoun in question:
                question = question.replace(pronoun, f"「{self.current_entity}」")

        return question

    def get_suggested_questions(self):
        """生成推荐问题（基于当前实体）"""
        if not self.current_entity:
            return []

        entity = self.current_entity
        suggestions = [
            f"{entity}的评分是多少？",
            f"{entity}是哪一年上映的？",
            f"{entity}是什么类型的电影？",
            f"{entity}的主演都有谁？",
        ]
        return suggestions


# ======================
# 完整问答Pipeline
# ======================
class KGQAPipeline:
    """
    KGQA完整流程封装
    串联五大模块，实现端到端问答
    """

    def __init__(self, use_llm=False):
        print("🚀 初始化KGQA Pipeline...")
        self.intent_recognizer = IntentRecognizer()
        self.entity_linker = EntityLinker()
        self.neo4j = Neo4jSession()
        self.answer_generator = AnswerGenerator()
        self.conversation = ConversationManager()
        self.use_llm = use_llm
        print("✅ Pipeline初始化完成\n")

    def ask(self, question):
        """
        主问答接口
        Args:
            question: 用户自然语言问句
        Returns:
            result: 包含intent, entity, cypher, graph_result, answer的字典
        """
        print(f"❓ 用户问题: {question}")

        # 1. 指代消解
        resolved_question = self.conversation.resolve_coreference(question)
        if resolved_question != question:
            print(f"🔄 指代消解: {resolved_question}")

        # 2. 意图识别
        intent, confidence = self.intent_recognizer.recognize(resolved_question)
        print(f"🎯 意图识别: {intent} (置信度: {confidence:.4f})")

        # 3. 实体链接（传入意图进行联合推理）
        primary_entity, entity_type, all_entities = self.entity_linker.link(resolved_question, intent_hint=intent)
        print(f"🔗 实体链接: {primary_entity} (类型: {entity_type})")

        if not primary_entity:
            answer = "抱歉，我没能理解您提到的电影或人物，请提供更具体的信息。"
            self.conversation.add_turn(question, answer, intent, None)
            return {
                "intent": intent,
                "entity": None,
                "cypher": None,
                "graph_result": [],
                "answer": answer,
                "confidence": confidence
            }

        # 4. Cypher生成（参数化查询）
        cypher, params = CypherGenerator.generate_template(intent, primary_entity, entity_type)
        print(f"📝 Cypher生成: {cypher}")

        # 5. Neo4j查询
        graph_result = self.neo4j.query(cypher, params)
        print(f"📊 图谱结果: {graph_result}")

        # 6. 答案生成
        answer, source = self.answer_generator.generate(
            resolved_question, intent, primary_entity, graph_result, self.use_llm
        )
        print(f"💬 答案生成 ({source}): {answer}\n")

        # 7. 更新对话上下文
        self.conversation.add_turn(question, answer, intent, primary_entity)

        return {
            "intent": intent,
            "entity": primary_entity,
            "entity_type": entity_type,
            "cypher": cypher,
            "graph_result": graph_result,
            "answer": answer,
            "confidence": confidence,
            "source": source,
            "context": self.conversation.get_context()
        }

    def get_recommendations(self):
        """获取推荐问题"""
        return self.conversation.get_suggested_questions()

    def close(self):
        """释放资源"""
        self.neo4j.close()


# ======================
# 测试入口
# ======================
if __name__ == "__main__":
    # 测试各模块
    print("=" * 50)
    print("KGQA 核心模块测试")
    print("=" * 50)

    # 测试意图识别
    print("\n【1. 意图识别测试】")
    recognizer = IntentRecognizer()
    test_questions = [
        "流浪地球的导演是谁？",
        "周星驰演过哪些电影？",
        "这部电影评分多少？",
        "哪一年上映的？",
        "什么类型的电影？"
    ]
    for q in test_questions:
        intent, conf = recognizer.recognize(q)
        print(f"  {q} → {intent} ({conf:.4f})")

    # 测试实体链接
    print("\n【2. 实体链接测试】")
    linker = EntityLinker()
    for q in ["流浪地球的导演是谁？", "张艺谋导演了哪些电影？"]:
        entity, e_type, entities = linker.link(q)
        print(f"  {q} → 实体: {entity}, 类型: {e_type}")

    # 测试Cypher生成
    print("\n【3. Cypher生成测试】")
    for intent in ["query_director", "query_actor", "query_rating", "query_year", "query_genre"]:
        cypher, params = CypherGenerator.generate_template(intent, "流浪地球", "movie")
        print(f"  {intent} → {cypher} | 参数: {params}")

    # 测试Neo4j查询
    print("\n【4. Neo4j交互测试】")
    neo = Neo4jSession()
    stats = neo.get_statistics()
    print(f"  图谱统计: {stats}")
    result = neo.query("MATCH (m:Movie) RETURN m.title AS title LIMIT 3")
    print(f"  查询测试: {result}")
    neo.close()

    # 测试答案生成
    print("\n【5. 答案生成测试】")
    generator = AnswerGenerator()
    test_result = [{"director": "郭帆"}]
    answer, source = generator.generate("流浪地球的导演是谁？", "query_director", "流浪地球", test_result, use_llm=False)
    print(f"  规则生成: {answer}")

    # 完整Pipeline测试
    print("\n【6. 完整Pipeline测试】")
    pipeline = KGQAPipeline(use_llm=True)
    test_qs = [
        "流浪地球的导演是谁？",
        "周星驰主演过哪些电影？",
        "霸王别姬的评分是多少？"
    ]
    for q in test_qs:
        result = pipeline.ask(q)
        print(f"  答案: {result['answer']}")
    pipeline.close()

    print("\n✅ 所有模块测试完成！")
