import pandas as pd

# ==========================================
# 电影数据预处理脚本
# 功能：去除空值、重复数据，规范字段格式
# 输出：预处理后的 movies_data.csv
# ==========================================

# 读取数据集
df = pd.read_csv("movies_data.csv", encoding="utf-8")

print(f"原始数据共 {len(df)} 条")

# 预处理：去除关键字段空值
df = df.dropna(subset=["title", "director", "actor", "genre"])

# 去除重复数据（同一电影、导演、演员组合视为重复）
df = df.drop_duplicates(subset=["title", "director", "actor"])

# 统一年份格式为字符串
df["year"] = df["year"].astype(str)

# 统一评分格式为字符串
df["rating"] = df["rating"].astype(str)

# 保存预处理后的数据集（覆盖原文件，供后续导入Neo4j使用）
df.to_csv("movies_data.csv", index=False, encoding="utf-8")

print(f"数据集预处理完成，共保留 {len(df)} 条有效数据")
print(f"电影数量：{df['title'].nunique()} 部")
print(f"人物数量：{pd.concat([df['director'], df['actor']]).nunique()} 人")
print(f"类型数量：{df['genre'].nunique()} 种")
