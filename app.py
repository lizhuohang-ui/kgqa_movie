import streamlit as st
import streamlit.components.v1 as components
import requests
import json
import time

# ==========================================
# KGQA 交互前端 - Streamlit 应用
# 功能：用户交互界面、多轮对话、推理可视化展示
# 启动命令：streamlit run app.py
# ==========================================

# ===================== 配置 =====================
API_BASE_URL = "http://localhost:8000"
DEFAULT_SESSION_ID = "user_session_001"

# 页面配置
st.set_page_config(
    page_title="🎬 电影知识图谱智能问答系统",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== 样式自定义 =====================
st.markdown("""
<style>
    /* 全局字体与背景 */
    html, body, [class*="css"] {
        font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    /* 头部动效 */
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .main-header {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E88E5, #D81B60, #FB8C00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.3rem;
        animation: fadeInDown 0.8s ease-out;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #888;
        text-align: center;
        margin-bottom: 2rem;
        animation: fadeInDown 1s ease-out;
    }

    /* 聊天气泡 */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .chat-message-user {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        padding: 1rem 1.2rem;
        border-radius: 16px 16px 4px 16px;
        margin: 0.6rem 0 0.6rem 3rem;
        border: 1px solid #90CAF9;
        box-shadow: 0 2px 8px rgba(30,136,229,0.08);
        animation: fadeInUp 0.4s ease-out;
    }
    .chat-message-bot {
        background: linear-gradient(135deg, #F5F5F5 0%, #EEEEEE 100%);
        padding: 1rem 1.2rem;
        border-radius: 16px 16px 16px 4px;
        margin: 0.6rem 3rem 0.6rem 0;
        border: 1px solid #E0E0E0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        animation: fadeInUp 0.5s ease-out;
    }

    /* 意图标签 */
    .intent-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.8rem;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        transition: transform 0.2s;
    }
    .intent-badge:hover { transform: scale(1.05); }
    .intent-query_director { background: #E3F2FD; color: #1565C0; border: 1px solid #90CAF9; }
    .intent-query_actor    { background: #F3E5F5; color: #7B1FA2; border: 1px solid #CE93D8; }
    .intent-query_rating   { background: #FFF3E0; color: #E65100; border: 1px solid #FFCC80; }
    .intent-query_year     { background: #E8F5E9; color: #2E7D32; border: 1px solid #A5D6A7; }
    .intent-query_genre    { background: #FCE4EC; color: #C2185B; border: 1px solid #F48FB1; }

    /* 推荐问题 Chip */
    .suggestion-chip {
        display: inline-block;
        padding: 0.45rem 1rem;
        margin: 0.25rem;
        border-radius: 24px;
        background: linear-gradient(135deg, #E3F2FD, #F3E5F5);
        color: #1565C0;
        font-size: 0.9rem;
        font-weight: 500;
        cursor: pointer;
        border: 1px solid #BBDEFB;
        transition: all 0.25s ease;
        box-shadow: 0 1px 4px rgba(30,136,229,0.06);
    }
    .suggestion-chip:hover {
        background: linear-gradient(135deg, #1565C0, #7B1FA2);
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(30,136,229,0.2);
    }

    /* 推理步骤卡片 */
    .reason-step {
        background: #FAFAFA;
        border-left: 4px solid #1E88E5;
        border-radius: 0 8px 8px 0;
        padding: 0.6rem 1rem;
        margin: 0.4rem 0;
        transition: background 0.2s;
    }
    .reason-step:hover { background: #F5F5F5; }

    /* 空状态 */
    .empty-state {
        text-align: center;
        padding: 2.5rem 1rem;
        color: #666;
    }
    .empty-state h3 {
        color: #444;
        margin-bottom: 1rem;
    }
    .empty-icon {
        font-size: 3.5rem;
        margin-bottom: 1rem;
        opacity: 0.7;
    }

    /* 页脚 */
    .footer {
        text-align: center;
        color: #aaa;
        font-size: 0.8rem;
        padding: 1.5rem 0 0.5rem;
        border-top: 1px solid #eee;
        margin-top: 2rem;
    }

    /* 侧边栏样式微调 */
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .status-green { background: #4CAF50; box-shadow: 0 0 6px #4CAF50; }
    .status-red   { background: #F44336; box-shadow: 0 0 6px #F44336; }

    /* 隐藏 Streamlit 默认元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ===================== 工具函数 =====================
@st.cache_data(ttl=5)
def call_api(question, use_llm=False, session_id=DEFAULT_SESSION_ID, api_base=API_BASE_URL):
    """调用后端 API（带缓存）"""
    try:
        response = requests.get(
            f"{api_base}/kgqa",
            params={"question": question, "use_llm": use_llm, "session_id": session_id},
            timeout=15
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"success": False, "error": f"HTTP {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "无法连接到后端服务，请确保 FastAPI 服务已启动（uvicorn main_api:app --reload）"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@st.cache_data(ttl=10)
def get_health(api_base=API_BASE_URL):
    """检查后端健康状态"""
    try:
        response = requests.get(f"{api_base}/health", timeout=3)
        return response.json()
    except Exception:
        return {"status": "unhealthy"}


def get_suggestions(session_id=DEFAULT_SESSION_ID, api_base=API_BASE_URL):
    """获取推荐问题"""
    try:
        response = requests.get(
            f"{api_base}/chat/history",
            params={"session_id": session_id},
            timeout=5
        )
        data = response.json()
        if data.get("success"):
            history = data.get("history", [])
            if history:
                last_entity = history[-1].get("entity")
                if last_entity:
                    return [
                        f"{last_entity}的评分是多少？",
                        f"{last_entity}是哪一年上映的？",
                        f"{last_entity}是什么类型的电影？",
                        f"{last_entity}的主演都有谁？",
                    ]
        return []
    except Exception:
        return []


def get_graph_neighbors(entity, depth=1, api_base=API_BASE_URL):
    """获取实体邻居子图"""
    try:
        response = requests.get(
            f"{api_base}/graph/neighbors",
            params={"entity": entity, "depth": depth},
            timeout=10
        )
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}


def render_graph(entity, depth=1):
    """使用 ECharts 渲染力导向图"""
    data = get_graph_neighbors(entity, depth)
    if not data.get("success"):
        st.error(f"图谱加载失败: {data.get('error', '未知错误')}")
        return

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    if not nodes:
        st.info("未找到该实体的图谱数据，请检查实体名称是否正确")
        return

    # 为节点计算连接度（用于调整节点大小）
    degree_map = {}
    for e in edges:
        degree_map[e["source"]] = degree_map.get(e["source"], 0) + 1
        degree_map[e["target"]] = degree_map.get(e["target"], 0) + 1

    category_map = {"Movie": 0, "Person": 1, "Genre": 2}
    category_colors = {"Movie": "#1E88E5", "Person": "#43A047", "Genre": "#FB8C00"}

    echarts_nodes = []
    for n in nodes:
        label = n.get("label", "Unknown")
        echarts_nodes.append({
            "name": n["name"],
            "label": label,
            "category": category_map.get(label, 0),
            "symbolSize": 25 + degree_map.get(n["id"], 0) * 6,
            "value": degree_map.get(n["id"], 0),
            "itemStyle": {"color": category_colors.get(label, "#999")}
        })

    echarts_edges = []
    for e in edges:
        echarts_edges.append({
            "source": e["source"],
            "target": e["target"],
            "relation": e["relation"]
        })

    nodes_json = json.dumps(echarts_nodes, ensure_ascii=False)
    edges_json = json.dumps(echarts_edges, ensure_ascii=False)

    chart_html = f"""
    <div id="graph-chart" style="width:100%;height:520px;border-radius:12px;overflow:hidden;"></div>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
    <script>
        (function() {{
            var chartDom = document.getElementById('graph-chart');
            if (!chartDom) return;
            var chart = echarts.init(chartDom);
            var option = {{
                backgroundColor: '#fafafa',
                tooltip: {{
                    trigger: 'item',
                    formatter: function(p) {{
                        if (p.dataType === 'edge') {{
                            return p.data.source + ' → ' + p.data.target + '<br/>关系: ' + p.data.relation;
                        }}
                        return '<b>' + p.data.name + '</b><br/>类型: ' + p.data.label + '<br/>连接数: ' + p.data.value;
                    }}
                }},
                legend: {{
                    data: ['Movie', 'Person', 'Genre'],
                    bottom: 10,
                    textStyle: {{ fontSize: 12 }}
                }},
                animationDuration: 1500,
                animationEasingUpdate: 'quinticInOut',
                series: [{{
                    type: 'graph',
                    layout: 'force',
                    roam: true,
                    draggable: true,
                    label: {{
                        show: true,
                        position: 'bottom',
                        fontSize: 12,
                        color: '#333'
                    }},
                    edgeLabel: {{
                        show: true,
                        formatter: function(x) {{ return x.data.relation; }},
                        fontSize: 10,
                        color: '#666'
                    }},
                    emphasis: {{
                        focus: 'adjacency',
                        lineStyle: {{ width: 4 }}
                    }},
                    force: {{
                        repulsion: 350,
                        edgeLength: [60, 140],
                        gravity: 0.1
                    }},
                    categories: [
                        {{ name: 'Movie', itemStyle: {{ color: '#1E88E5' }} }},
                        {{ name: 'Person', itemStyle: {{ color: '#43A047' }} }},
                        {{ name: 'Genre', itemStyle: {{ color: '#FB8C00' }} }}
                    ],
                    data: {nodes_json},
                    links: {edges_json},
                    lineStyle: {{
                        color: 'source',
                        curveness: 0.2,
                        opacity: 0.7,
                        width: 2
                    }}
                }}]
            }};
            chart.setOption(option);
            window.addEventListener('resize', function() {{ chart.resize(); }});
        }})();
    </script>
    """
    components.html(chart_html, height=540, scrolling=False)


def render_intent_badge(intent):
    """渲染意图标签"""
    intent_names = {
        "query_director": "导演",
        "query_actor": "演员",
        "query_rating": "评分",
        "query_year": "年份",
        "query_genre": "类型"
    }
    name = intent_names.get(intent, intent)
    return f'<span class="intent-badge intent-{intent}">{name}</span>'


def render_message(msg):
    """渲染单条消息"""
    if msg["role"] == "user":
        st.markdown(f'''
            <div class="chat-message-user">
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
                    <span style="font-size:1.2rem;">👤</span>
                    <span style="font-weight:600;color:#1565C0;font-size:0.9rem;">你</span>
                </div>
                <div style="color:#333;line-height:1.5;">{msg["content"]}</div>
            </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
            <div class="chat-message-bot">
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
                    <span style="font-size:1.2rem;">🤖</span>
                    <span style="font-weight:600;color:#2E7D32;font-size:0.9rem;">智能助手</span>
                </div>
                <div style="color:#333;line-height:1.6;">{msg["content"]}</div>
            </div>
        ''', unsafe_allow_html=True)

        # 推理过程（可展开）
        metadata = msg.get("metadata", {})
        if metadata and not metadata.get("error"):
            with st.expander("🔍 查看推理过程", expanded=False):
                # 1. 意图识别
                intent = metadata.get("intent")
                confidence = metadata.get("confidence", 0)
                if intent:
                    st.markdown('<div class="reason-step">', unsafe_allow_html=True)
                    st.markdown("**1️⃣ 意图识别**")
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.markdown(render_intent_badge(intent), unsafe_allow_html=True)
                    with col2:
                        st.progress(min(confidence, 1.0), text=f"置信度: {confidence:.2%}")
                    st.markdown('</div>', unsafe_allow_html=True)

                # 2. 实体链接
                entity = metadata.get("entity")
                entity_type = metadata.get("entity_type")
                if entity:
                    st.markdown('<div class="reason-step">', unsafe_allow_html=True)
                    st.markdown("**2️⃣ 实体链接**")
                    st.info(f"实体：「{entity}」&nbsp;&nbsp;|&nbsp;&nbsp;类型: {entity_type}")
                    st.markdown('</div>', unsafe_allow_html=True)

                # 3. Cypher 生成
                cypher = metadata.get("cypher")
                if cypher:
                    st.markdown('<div class="reason-step">', unsafe_allow_html=True)
                    st.markdown("**3️⃣ Cypher 查询语句**")
                    st.code(cypher, language="cypher")
                    st.markdown('</div>', unsafe_allow_html=True)

                # 4. 图谱结果
                graph_result = metadata.get("graph_result")
                if graph_result:
                    st.markdown('<div class="reason-step">', unsafe_allow_html=True)
                    st.markdown("**4️⃣ 图谱原始结果**")
                    st.json(graph_result, expanded=False)
                    st.markdown('</div>', unsafe_allow_html=True)

                # 5. 答案来源
                source = metadata.get("source")
                if source:
                    st.markdown('<div class="reason-step">', unsafe_allow_html=True)
                    st.markdown("**5️⃣ 答案来源**")
                    if source == "llm":
                        st.success("🤖 大模型生成")
                    else:
                        st.info("📋 规则模板生成")
                    st.markdown('</div>', unsafe_allow_html=True)


# ===================== 侧边栏 =====================
with st.sidebar:
    # 个人信息卡片
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1E88E5 0%, #7B1FA2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(30,136,229,0.25);
    ">
        <div style="font-size: 2rem; margin-bottom: 0.3rem;">👨‍🎓</div>
        <div style="font-size: 1.1rem; font-weight: 700; margin-bottom: 0.2rem;">freebird</div>
        <div style="font-size: 0.85rem; opacity: 0.95; line-height: 1.5;">
            人工智能二班 李卓航<br>
            湘潭大学
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ 系统设置")

    # API 地址设置
    api_url = st.text_input("后端 API 地址", value=API_BASE_URL)
    if api_url != API_BASE_URL:
        API_BASE_URL = api_url

    # 大模型开关
    use_llm = st.checkbox("🤖 使用大模型生成答案", value=False,
                          help="开启后会调用 DeepSeek 等大模型 API 生成更自然的回答，需要配置 API 密钥")

    # 会话管理
    st.markdown("---")
    st.markdown("### 💬 会话管理")
    session_id = st.text_input("会话 ID", value=DEFAULT_SESSION_ID)

    if st.button("🗑️ 清空对话历史", use_container_width=True, type="secondary"):
        try:
            requests.delete(f"{API_BASE_URL}/chat/history", params={"session_id": session_id})
            if "messages" in st.session_state:
                st.session_state.messages = []
            st.success("✅ 对话历史已清空！")
            time.sleep(0.3)
            st.experimental_rerun()
        except Exception as e:
            st.error(f"清空失败: {e}")

    # 系统状态
    st.markdown("---")
    st.markdown("### 📊 系统状态")
    health = get_health(API_BASE_URL)
    if health.get("status") == "healthy":
        st.markdown('<span class="status-dot status-green"></span>**后端服务正常**', unsafe_allow_html=True)
        stats = health.get("graph_stats", {})
        if stats:
            nodes = stats.get("nodes", {})
            rels = stats.get("relationships", {})
            if nodes:
                st.markdown("**节点统计**")
                for label, count in nodes.items():
                    st.markdown(f"- `{label}`: **{count}**")
            if rels:
                st.markdown("**关系统计**")
                for rel_type, count in rels.items():
                    st.markdown(f"- `{rel_type}`: **{count}**")
    else:
        st.markdown('<span class="status-dot status-red"></span>**后端服务异常**', unsafe_allow_html=True)
        st.info("请确保已启动 FastAPI 服务：\n```\nuvicorn main_api:app --reload\n```")

    # 快速测试问题
    st.markdown("---")
    st.markdown("### 📝 快速测试")
    quick_questions = [
        "流浪地球的导演是谁？",
        "周星驰主演过哪些电影？",
        "我不是药神的评分是多少？",
        "2023年有哪些科幻电影？",
        "张艺谋和巩俐合作过哪些电影？"
    ]
    for q in quick_questions:
        if st.button(q, use_container_width=True, key=f"quick_{q}"):
            st.session_state.current_question = q
            st.experimental_rerun()

# ===================== 主界面 =====================
st.markdown('<div class="main-header">🎬 电影知识图谱智能问答系统</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">基于意图识别 + 实体链接 + Cypher 查询 + 大模型生成</div>', unsafe_allow_html=True)

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_question" not in st.session_state:
    st.session_state.current_question = ""

if "graph_entity" not in st.session_state:
    st.session_state.graph_entity = ""

# ===================== 图谱可视化 =====================
with st.expander("🕸️ 图谱可视化", expanded=False):
    st.markdown("探索知识图谱中的实体关系，输入电影名或人名即可查看关联子图。")
    g_col1, g_col2 = st.columns([4, 1])
    with g_col1:
        graph_input = st.text_input(
            "输入实体名称",
            value=st.session_state.graph_entity,
            placeholder="例如：流浪地球、周星驰",
            label_visibility="collapsed"
        )
    with g_col2:
        graph_search = st.button("🔍 探索图谱", use_container_width=True, type="primary", key="graph_search_btn")

    if graph_search and graph_input.strip():
        st.session_state.graph_entity = graph_input.strip()
        st.markdown(f"**当前实体：「{graph_input.strip()}」**")
        render_graph(graph_input.strip())
    elif st.session_state.graph_entity:
        st.markdown(f"**当前实体：「{st.session_state.graph_entity}」**")
        render_graph(st.session_state.graph_entity)

# ===================== 输入区域 =====================
input_col1, input_col2 = st.columns([6, 1])
with input_col1:
    question = st.text_input(
        "请输入您的问题：",
        value=st.session_state.current_question,
        placeholder="例如：流浪地球的导演是谁？",
        label_visibility="collapsed"
    )
with input_col2:
    search_clicked = st.button("🔍 查询", use_container_width=True, type="primary")

# 清除当前问题（避免重复）
if st.session_state.current_question:
    st.session_state.current_question = ""

# ===================== 推荐问题 =====================
suggestions = get_suggestions(session_id, API_BASE_URL)
if suggestions:
    st.markdown("**💡 推荐追问：**")
    sg_cols = st.columns(len(suggestions))
    for i, sq in enumerate(suggestions):
        with sg_cols[i]:
            if st.button(sq, key=f"sg_{i}", use_container_width=True):
                st.session_state.current_question = sq
                st.experimental_rerun()

st.markdown("---")

# ===================== 处理查询 =====================
if search_clicked and question.strip():
    # 添加到消息历史
    st.session_state.messages.append({"role": "user", "content": question})

    # 显示加载状态
    with st.spinner("🤔 正在思考，请稍候..."):
        result = call_api(question, use_llm=use_llm, session_id=session_id, api_base=API_BASE_URL)

    # 处理结果
    if result.get("success"):
        answer = result.get("answer", "无答案")
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "metadata": result
        })
    else:
        error_msg = result.get("error", "未知错误")
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"抱歉，处理出现问题：{error_msg}",
            "metadata": {"error": error_msg}
        })
    st.experimental_rerun()

# ===================== 显示对话历史 =====================
if st.session_state.messages:
    for msg in st.session_state.messages:
        render_message(msg)
else:
    # 空状态提示
    st.markdown('''
        <div class="empty-state">
            <div class="empty-icon">🎬</div>
            <h3>欢迎使用电影知识图谱智能问答系统</h3>
            <p>您可以问我关于电影的各类问题，例如：</p>
        </div>
    ''', unsafe_allow_html=True)

    empty_cols = st.columns(5)
    examples = [
        ("🎬", "导演相关", "流浪地球的导演是谁？"),
        ("🎭", "演员相关", "周星驰主演过哪些电影？"),
        ("⭐", "评分相关", "霸王别姬的评分是多少？"),
        ("📅", "年份相关", "泰坦尼克号是哪年的？"),
        ("🏷️", "类型相关", "有哪些好看的动画电影？"),
    ]
    for col, (icon, title, example) in zip(empty_cols, examples):
        with col:
            st.markdown(f"""
                <div style="text-align:center;padding:0.8rem;border-radius:12px;background:#FAFAFA;border:1px solid #eee;">
                    <div style="font-size:1.8rem;margin-bottom:0.3rem;">{icon}</div>
                    <div style="font-weight:600;color:#444;font-size:0.9rem;margin-bottom:0.3rem;">{title}</div>
                    <div style="font-size:0.8rem;color:#888;">{example}</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("""
        <div class="empty-state" style="margin-top:1.5rem;">
            <p>🔧 <strong>开始使用：</strong>确保 Neo4j 数据库已启动并导入数据，然后启动 FastAPI 后端和 Streamlit 前端。</p>
        </div>
    """, unsafe_allow_html=True)

# ===================== 页脚 =====================
st.markdown('<div class="footer">🎓 知识图谱实训项目 | 技术栈：Neo4j + BERT + FastAPI + Streamlit</div>', unsafe_allow_html=True)
