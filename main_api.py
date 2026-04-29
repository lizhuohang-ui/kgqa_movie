from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
from config import get_llm_status

# 导入五大核心模块
from core_modules import (
    KGQAPipeline, IntentRecognizer, EntityLinker,
    CypherGenerator, Neo4jSession, AnswerGenerator, ConversationManager
)

# ==========================================
# FastAPI后端：电影KGQA系统API
# 功能：封装KGQA五大模块为RESTful接口
# 启动命令：uvicorn main_api:app --reload
# ==========================================

app = FastAPI(
    title="电影知识图谱智能问答系统 API",
    description="基于意图识别 + 实体链接 + Cypher生成 + Neo4j + 大模型的KGQA系统",
    version="1.0.0"
)

# 跨域配置（允许前端访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== 请求/响应模型 =====================

class QuestionRequest(BaseModel):
    question: str
    use_llm: bool = True
    session_id: Optional[str] = "default"


class QAResponse(BaseModel):
    success: bool
    question: str
    intent: Optional[str] = None
    entity: Optional[str] = None
    entity_type: Optional[str] = None
    cypher: Optional[str] = None
    graph_result: Optional[list] = None
    answer: Optional[str] = None
    confidence: Optional[float] = None
    source: Optional[str] = None
    context: Optional[str] = None
    suggested_questions: Optional[List[str]] = None
    error: Optional[str] = None


class Neo4jQueryRequest(BaseModel):
    cypher: str
    parameters: Optional[dict] = None


# ===================== 全局实例 =====================
# 使用单例模式复用核心组件
neo4j_session = Neo4jSession()
intent_recognizer = IntentRecognizer()
entity_linker = EntityLinker()
answer_generator = AnswerGenerator()

# 会话管理（简单内存存储，生产环境建议用Redis）
sessions = {}


def get_or_create_session(session_id: str):
    """获取或创建对话会话"""
    if session_id not in sessions:
        sessions[session_id] = ConversationManager()
    return sessions[session_id]


# ===================== API接口 =====================

@app.get("/")
def root():
    """根路径 - 服务状态检查"""
    return {
        "message": "电影知识图谱智能问答系统 API 运行中",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
def health_check():
    """健康检查接口"""
    try:
        # 测试Neo4j连接
        stats = neo4j_session.get_statistics()
        return {
            "status": "healthy",
            "neo4j_connected": True,
            "graph_stats": stats,
            "llm": get_llm_status()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "neo4j_connected": False,
            "error": str(e),
            "llm": get_llm_status()
        }


@app.post("/kgqa", response_model=QAResponse)
def kgqa(request: QuestionRequest):
    """
    KGQA主接口 - 完整问答流程
    """
    try:
        question = request.question.strip()
        if not question:
            return QAResponse(
                success=False,
                question=question,
                error="问题不能为空"
            )

        # 获取会话上下文
        conversation = get_or_create_session(request.session_id)

        # 指代消解
        resolved_question = conversation.resolve_coreference(question)

        # 1. 意图识别
        intent, confidence = intent_recognizer.recognize(resolved_question)

        # 2. 实体链接（传入意图进行联合推理）
        primary_entity, entity_type, all_entities = entity_linker.link(resolved_question, intent_hint=intent)

        if not primary_entity:
            answer = "抱歉，我没能理解您提到的电影或人物，请提供更具体的信息。"
            conversation.add_turn(question, answer, intent, None)
            return QAResponse(
                success=True,
                question=question,
                intent=intent,
                entity=None,
                answer=answer,
                confidence=confidence,
                source="rule",
                context=conversation.get_context()
            )

        # 3. Cypher生成（参数化查询，防止注入）
        cypher, params = CypherGenerator.generate_template(intent, primary_entity, entity_type)

        # 4. Neo4j查询
        graph_result = neo4j_session.query(cypher, params)

        # 5. 答案生成
        answer, source = answer_generator.generate(
            resolved_question, intent, primary_entity, graph_result, request.use_llm
        )

        # 更新对话上下文
        conversation.add_turn(question, answer, intent, primary_entity)

        # 获取推荐问题
        suggestions = conversation.get_suggested_questions()

        return QAResponse(
            success=True,
            question=question,
            intent=intent,
            entity=primary_entity,
            entity_type=entity_type,
            cypher=cypher,
            graph_result=graph_result,
            answer=answer,
            confidence=confidence,
            source=source,
            context=conversation.get_context(),
            suggested_questions=suggestions
        )

    except Exception as e:
        return QAResponse(
            success=False,
            question=request.question,
            error=f"系统错误: {str(e)}"
        )


@app.get("/kgqa")
def kgqa_get(question: str, use_llm: bool = True, session_id: str = "default"):
    """KGQA GET接口（兼容简单调用）"""
    request = QuestionRequest(question=question, use_llm=use_llm, session_id=session_id)
    return kgqa(request)


@app.post("/query")
def neo4j_query(request: Neo4jQueryRequest):
    """直接执行Cypher查询（高级接口）"""
    try:
        result = neo4j_session.query(request.cypher, request.parameters)
        return {
            "success": True,
            "cypher": request.cypher,
            "result": result,
            "count": len(result)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/entity/search")
def search_entity(keyword: str, limit: int = 10):
    """实体搜索接口"""
    try:
        cypher = """
            MATCH (m:Movie)
            WHERE m.title CONTAINS $keyword
            RETURN m.title AS title, m.year AS year, m.rating AS rating
            LIMIT $limit
        """
        movies = neo4j_session.query(cypher, {"keyword": keyword, "limit": limit})

        cypher_person = """
            MATCH (p:Person)
            WHERE p.name CONTAINS $keyword
            RETURN p.name AS name
            LIMIT $limit
        """
        persons = neo4j_session.query(cypher_person, {"keyword": keyword, "limit": limit})

        return {
            "success": True,
            "keyword": keyword,
            "movies": movies,
            "persons": persons
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/path")
def get_path(entity1: str, entity2: str):
    """获取两个实体之间的路径（用于可视化）"""
    try:
        cypher = """
            MATCH path = shortestPath(
                (a {name: $entity1})-[:DIRECTED|ACTED_IN*]-(b {name: $entity2})
            )
            RETURN path
        """
        paths = neo4j_session.query_with_path(cypher, {"entity1": entity1, "entity2": entity2})

        return {
            "success": True,
            "entity1": entity1,
            "entity2": entity2,
            "paths": paths
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/graph/stats")
def graph_statistics():
    """获取图谱统计信息"""
    try:
        stats = neo4j_session.get_statistics()
        return {
            "success": True,
            "statistics": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/neighbors")
def graph_neighbors(entity: str, depth: int = 1):
    """
    获取实体邻居子图（用于可视化）
    参数:
        entity: 实体名称（电影名或人名）
        depth: 探索深度（默认1，最大2）
    """
    try:
        if not entity or not entity.strip():
            return {"success": False, "error": "实体名称不能为空"}
        depth = min(max(depth, 1), 2)
        result = neo4j_session.get_neighbors(entity.strip(), depth)
        return {
            "success": True,
            "entity": entity,
            "depth": depth,
            "nodes": result.get("nodes", []),
            "edges": result.get("edges", []),
            "count": len(result.get("nodes", []))
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/chat/history")
def get_chat_history(session_id: str = "default"):
    """获取对话历史"""
    conversation = get_or_create_session(session_id)
    return {
        "success": True,
        "session_id": session_id,
        "history": conversation.history,
        "current_entity": conversation.current_entity
    }


@app.delete("/chat/history")
def clear_chat_history(session_id: str = "default"):
    """清空对话历史"""
    if session_id in sessions:
        sessions[session_id] = ConversationManager()
    return {
        "success": True,
        "message": f"会话 {session_id} 已清空"
    }


# ===================== 启动入口 =====================
if __name__ == "__main__":
    print("🚀 启动FastAPI服务...")
    print("📖 API文档地址: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
