from neo4j import GraphDatabase
import pandas as pd

# ==========================================
# Neo4j 电影知识图谱批量导入脚本
# 功能：创建Movie、Person、Genre节点及DIRECTED、ACTED_IN、IS_GENRE关系
# 前提：Neo4j数据库已启动，movies_data.csv已放入Neo4j的import文件夹
# ==========================================

# Neo4j配置（与自己设置的数据库信息一致）
# Neo4j 5.x 推荐使用 neo4j:// 协议
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "20040121"


class Neo4jImporter:
    def __init__(self):
        """连接Neo4j数据库"""
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        print("✅ 已连接到Neo4j数据库")

    def close(self):
        """关闭数据库连接"""
        self.driver.close()
        print("🔒 已关闭Neo4j连接")

    def clear_database(self):
        """清空数据库（谨慎使用）
        Neo4j 5.x 需要先删除约束再删数据（有约束时删节点会更快）
        """
        with self.driver.session() as session:
            # 5.x 中删除约束的新语法
            try:
                session.run("DROP CONSTRAINT movie_title IF EXISTS")
                session.run("DROP CONSTRAINT person_name IF EXISTS")
                session.run("DROP CONSTRAINT genre_name IF EXISTS")
            except Exception:
                pass
            session.run("MATCH (n) DETACH DELETE n")
            print("🗑️ 已清空数据库")

    def create_constraints(self):
        """创建约束（保证节点唯一性）
        Neo4j 5.x 中约束创建是异步的，使用 WAIT 确保生效
        """
        with self.driver.session() as session:
            try:
                # 5.x 语法：WAIT 等待约束创建完成
                session.run(
                    "CREATE CONSTRAINT movie_title IF NOT EXISTS "
                    "FOR (m:Movie) REQUIRE m.title IS UNIQUE"
                )
                session.run(
                    "CREATE CONSTRAINT person_name IF NOT EXISTS "
                    "FOR (p:Person) REQUIRE p.name IS UNIQUE"
                )
                session.run(
                    "CREATE CONSTRAINT genre_name IF NOT EXISTS "
                    "FOR (g:Genre) REQUIRE g.name IS UNIQUE"
                )
                print("✅ 约束创建完成")
            except Exception as e:
                print(f"⚠️ 约束创建提示（可能已存在）: {e}")

    def import_data(self, csv_path):
        """
        批量导入数据到Neo4j
        使用Cypher的LOAD CSV语句高效导入
        """
        # 先读取CSV确认数据存在
        df = pd.read_csv(csv_path, encoding="utf-8")
        print(f"📊 读取到 {len(df)} 条数据，开始导入...")

        with self.driver.session() as session:
            # 1. 创建Movie节点（去重）
            session.run('''
                LOAD CSV WITH HEADERS FROM "file:///movies_data.csv" AS row
                MERGE (m:Movie {title: row.title})
                SET m.year = row.year,
                    m.rating = row.rating,
                    m.genre = row.genre,
                    m.country = row.country
            ''')
            print("🎬 Movie节点创建完成")

            # 2. 创建Person节点（去重，涵盖导演和演员）
            session.run('''
                LOAD CSV WITH HEADERS FROM "file:///movies_data.csv" AS row
                MERGE (d:Person {name: row.director})
                MERGE (a:Person {name: row.actor})
            ''')
            print("👤 Person节点创建完成")

            # 3. 创建DIRECTED（导演）关系
            session.run('''
                LOAD CSV WITH HEADERS FROM "file:///movies_data.csv" AS row
                MATCH (d:Person {name: row.director})
                MATCH (m:Movie {title: row.title})
                MERGE (d)-[:DIRECTED]->(m)
            ''')
            print("🎬 DIRECTED关系创建完成")

            # 4. 创建ACTED_IN（主演）关系
            session.run('''
                LOAD CSV WITH HEADERS FROM "file:///movies_data.csv" AS row
                MATCH (a:Person {name: row.actor})
                MATCH (m:Movie {title: row.title})
                MERGE (a)-[:ACTED_IN]->(m)
            ''')
            print("🎭 ACTED_IN关系创建完成")

            # 5. 创建Genre节点及IS_GENRE关系
            session.run('''
                LOAD CSV WITH HEADERS FROM "file:///movies_data.csv" AS row
                MERGE (g:Genre {name: row.genre})
                WITH g, row
                MATCH (m:Movie {title: row.title})
                MERGE (m)-[:IS_GENRE]->(g)
            ''')
            print("🏷️ Genre节点及IS_GENRE关系创建完成")

        print("🎉 所有数据导入完成！")

    def verify_import(self):
        """验证导入结果"""
        with self.driver.session() as session:
            # 统计节点数量
            node_result = session.run("""
                MATCH (n) 
                RETURN labels(n) AS 节点类型, count(n) AS 数量
            """)
            print("\n📈 节点统计：")
            for record in node_result:
                print(f"  {record['节点类型']}: {record['数量']}")

            # 统计关系数量
            rel_result = session.run("""
                MATCH ()-[r]->() 
                RETURN type(r) AS 关系类型, count(r) AS 数量
            """)
            print("\n🔗 关系统计：")
            for record in rel_result:
                print(f"  {record['关系类型']}: {record['数量']}")


if __name__ == "__main__":
    importer = Neo4jImporter()
    try:
        # 可选：清空数据库重新导入
        # importer.clear_database()
        
        # 创建约束
        importer.create_constraints()
        
        # 导入数据（确保movies_data.csv已在Neo4j的import文件夹中）
        importer.import_data("movies_data.csv")
        
        # 验证导入结果
        importer.verify_import()
        
        print("\n✨ 知识图谱构建完成！")
        print("💡 提示：访问 http://localhost:7474 查看图谱可视化")
    finally:
        importer.close()
