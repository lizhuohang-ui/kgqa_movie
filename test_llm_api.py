import requests

# ==========================================
# 大模型API连通性测试脚本
# 功能：测试ChatGLM等大模型API是否可用
# 使用：修改 API_KEY 后直接运行
# ==========================================

# ===================== 配置区域 =====================
# 替换为自己的ChatGLM API密钥和请求地址
API_KEY = "your-api-key-here"
URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

# 备用配置（通义千问）
# API_KEY = "your-dashscope-key-here"
# URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
# ====================================================


def test_llm_api():
    """测试大模型API连通性"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    data = {
        "model": "glm-4",
        "messages": [
            {"role": "user", "content": "测试连通性，简单回复即可"}
        ],
        "temperature": 0.7
    }
    
    try:
        print("🔄 正在测试API连通性...")
        response = requests.post(URL, headers=headers, json=data, timeout=10)
        response.raise_for_status()  # 抛出HTTP请求异常
        
        result = response.json()
        answer = result["choices"][0]["message"]["content"]
        
        print("✅ 大模型API连通性测试成功！")
        print(f"🤖 模型回复：{answer}")
        return True
        
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP错误：{e}")
        if response.status_code == 401:
            print("💡 提示：API密钥无效，请检查API_KEY")
        elif response.status_code == 429:
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
    success = test_llm_api()
    if not success:
        print("\n⚠️ 测试失败，请检查：")
        print("  1. API密钥是否正确")
        print("  2. 网络是否正常")
        print("  3. API额度是否充足")
        print("  4. 请求地址是否正确")
