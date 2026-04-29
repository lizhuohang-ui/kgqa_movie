import requests
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_llm_config, has_llm_api_key

# ==========================================
# 大模型API连通性测试脚本
# 功能：测试 DeepSeek、ChatGLM 等 OpenAI 兼容大模型API是否可用
# 使用：设置 LLM_API_KEY / LLM_URL / LLM_MODEL 环境变量后直接运行
# ==========================================


def get_api_config(api_key=None, api_url=None, model=None):
    """读取大模型 API 配置，默认来自环境变量或本地 .env。"""
    return get_llm_config(api_key=api_key, api_url=api_url, model=model)


def check_llm_api(api_key=None, api_url=None, model=None):
    """测试大模型API连通性"""
    config = get_api_config(api_key=api_key, api_url=api_url, model=model)
    if not has_llm_api_key(config.api_key):
        print("⚠️ 未配置 LLM_API_KEY，已跳过大模型API连通性测试")
        print("💡 提示：请在环境变量或本地 .env 文件中设置 LLM_API_KEY")
        return False

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}"
    }
    
    data = {
        "model": config.model,
        "messages": [
            {"role": "user", "content": "测试连通性，简单回复即可"}
        ],
        "temperature": 0.7
    }
    
    try:
        print("🔄 正在测试API连通性...")
        response = requests.post(config.url, headers=headers, json=data, timeout=10)
        response.raise_for_status()  # 抛出HTTP请求异常
        
        result = response.json()
        answer = result["choices"][0]["message"]["content"]
        
        print("✅ 大模型API连通性测试成功！")
        print(f"🤖 模型回复：{answer}")
        return True
        
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP错误：{e}")
        if e.response is not None:
            if e.response.status_code == 401:
                print("💡 提示：API密钥无效，请检查 LLM_API_KEY")
            elif e.response.status_code == 429:
                print("💡 提示：请求过于频繁或额度不足")
        return False
        
    except requests.exceptions.ConnectionError:
        print("❌ 连接失败：无法连接到API服务器")
        print("💡 提示：请检查网络连接")
        return False
        
    except Exception as e:
        print(f"❌ 大模型API测试失败，错误信息：{str(e)}")
        return False


if __name__ == "__main__":
    success = check_llm_api()
    if not success:
        print("\n⚠️ 连通性测试未通过或已跳过，请检查：")
        print("  1. LLM_API_KEY 是否已设置且正确")
        print("  2. 网络是否正常")
        print("  3. API额度是否充足")
        print("  4. LLM_URL 和 LLM_MODEL 是否正确")
