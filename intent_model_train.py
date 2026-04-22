import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import numpy as np
import os

# ==========================================
# 意图识别模型训练脚本
# 功能：基于BERT+Softmax训练意图分类模型
# 标签：0-导演 1-演员 2-评分 3-年份 4-类型
# 输出：intent_model.pth（模型权重）
# ==========================================

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ 使用设备: {DEVICE}")

# BERT模型名称（中文）
MODEL_NAME = "bert-base-chinese"

# 意图标签映射
INTENT_MAP = {
    0: "query_director",    # 查询导演
    1: "query_actor",       # 查询演员
    2: "query_rating",      # 查询评分
    3: "query_year",        # 查询年份
    4: "query_genre"        # 查询类型
}

INTENT_LABELS = {v: k for k, v in INTENT_MAP.items()}

# ===================== 训练数据（问句样本） =====================
# 数据覆盖：数据集中的实际电影/人物名称 + 通用问法 + 变体句式
TRAIN_DATA = [
    # ========== query_director (0) - 查询导演 ==========
    ("流浪地球的导演是谁？", 0),
    ("谁导演了战狼2？", 0),
    ("这部电影的导演是谁？", 0),
    ("霸王别姬是谁拍的？", 0),
    ("阿凡达的导演是谁？", 0),
    ("张艺谋导演了哪些电影？", 0),
    ("卡梅隆拍了哪些电影？", 0),
    ("导演是谁？", 0),
    ("这部电影是谁执导的？", 0),
    ("谁是这部电影的导演？", 0),
    ("谁拍了泰坦尼克号？", 0),
    ("复仇者联盟是谁导演的？", 0),
    ("千与千寻的导演是？", 0),
    ("寄生虫是谁的作品？", 0),
    ("导演名单", 0),
    ("这部电影谁拍的", 0),
    ("谁是导演", 0),
    ("导演信息", 0),
    ("谁在幕后导演", 0),
    ("哪吒之魔童降世的导演", 0),
    ("红高粱谁拍的", 0),
    ("满江红导演", 0),
    ("我不是药神谁导演的", 0),
    ("盗梦空间的导演", 0),
    ("星际穿越谁拍的", 0),
    ("宫崎骏的作品", 0),
    ("新海诚导演了哪些", 0),
    ("冯小刚的电影", 0),
    ("姜文拍了什么", 0),
    ("奉俊昊的作品", 0),
    ("陈凯歌的电影", 0),
    ("请问流浪地球是谁拍的", 0),
    ("这部电影的导演资料", 0),
    ("幕后导演", 0),
    ("执导者", 0),
    (" filmmaker 是谁", 0),
    ("总导演", 0),
    ("谁执导的", 0),
    ("拍这部电影的人", 0),
    ("创作这部电影的导演", 0),
    ("这部影片谁导演的", 0),
    ("导戏的是谁", 0),
    ("导演是哪位", 0),
    ("谁当导演", 0),
    ("导演名单有哪些", 0),
    ("谁拍的电影", 0),
    ("这部影片谁拍的", 0),
    ("这部电影的 filmmaker", 0),
    ("求导演", 0),
    ("导演介绍", 0),
    ("谁创作的", 0),
    ("制作导演", 0),
    ("导演信息查询", 0),
    ("这部电影谁执导", 0),
    ("这部电影导演资料", 0),
    ("幕后主创", 0),
    ("主创导演", 0),

    # ========== query_actor (1) - 查询演员 ==========
    ("流浪地球的主演是谁？", 1),
    ("战狼2是谁主演的？", 1),
    ("周星驰演过哪些电影？", 1),
    ("这部电影的主演有哪些？", 1),
    ("阿凡达的主演是谁？", 1),
    ("演员有哪些？", 1),
    ("谁出演了这部电影？", 1),
    ("主演名单是什么？", 1),
    ("吴京演过哪些电影？", 1),
    ("巩俐参演了哪些作品？", 1),
    ("莱昂纳多演过什么？", 1),
    ("谁是主演", 1),
    ("演员表", 1),
    ("谁演的这部电影", 1),
    ("主演都有谁", 1),
    ("演员阵容", 1),
    ("出演人员", 1),
    ("谁参与了演出", 1),
    ("表演者是谁", 1),
    ("电影演员", 1),
    ("沈腾演过什么", 1),
    ("徐峥的电影", 1),
    ("梁朝伟主演", 1),
    ("张国荣演过哪些", 1),
    ("刘德华出演", 1),
    ("葛优的电影", 1),
    ("邓超演过什么", 1),
    ("黄渤参演", 1),
    ("张译的作品", 1),
    ("王宝强电影", 1),
    ("易烊千玺主演", 1),
    ("宋康昊的电影", 1),
    ("孔刘演过什么", 1),
    ("主演名单", 1),
    ("领衔主演", 1),
    ("联袂主演", 1),
    ("特别出演", 1),
    ("友情出演", 1),
    ("客串演员", 1),
    ("角色扮演者", 1),
    ("谁扮演主角", 1),
    ("男主角是谁", 1),
    ("女主角是谁", 1),
    ("男一号", 1),
    ("女一号", 1),
    ("演员列表", 1),
    ("卡司阵容", 1),
    ("cast", 1),
    ("出演者", 1),
    ("参演明星", 1),
    ("演员信息", 1),
    ("谁演的", 1),
    ("扮演者", 1),
    ("饰演者", 1),
    ("谁饰演的", 1),
    ("表演阵容", 1),
    ("演员表查询", 1),
    ("主要演员", 1),
    ("全体演员", 1),

    # ========== query_rating (2) - 查询评分 ==========
    ("流浪地球评分多少？", 2),
    ("这部电影的评分是多少？", 2),
    ("霸王别姬评分高吗？", 2),
    ("阿凡达多少分？", 2),
    ("豆瓣评分", 2),
    ("评分怎么样", 2),
    ("多少分", 2),
    ("这部电影评价如何", 2),
    ("口碑怎么样", 2),
    ("评分多少分", 2),
    ("打分多少", 2),
    ("分数高吗", 2),
    (" rating是多少", 2),
    ("这部电影得分", 2),
    ("评分信息", 2),
    ("几分", 2),
    ("多少星", 2),
    ("评价分数", 2),
    ("值多少分", 2),
    ("评分情况", 2),
    ("泰坦尼克号多少分", 2),
    ("盗梦空间评分", 2),
    ("星际穿越口碑", 2),
    ("千与千寻几分", 2),
    ("龙猫评分", 2),
    ("你的名字多少分", 2),
    ("釜山行评分", 2),
    ("熔炉多少分", 2),
    ("素媛评分", 2),
    ("豆瓣评分多少", 2),
    ("imdb评分", 2),
    ("烂番茄评分", 2),
    ("猫眼评分", 2),
    ("淘票票评分", 2),
    ("大众评分", 2),
    ("专业评分", 2),
    ("影迷评分", 2),
    ("观众打分", 2),
    ("评论家评分", 2),
    ("平均评分", 2),
    ("综合评分", 2),
    ("总评分", 2),
    ("评分排名", 2),
    ("评分对比", 2),
    ("好评率", 2),
    ("推荐度", 2),
    ("值得看吗", 2),
    ("口碑如何", 2),
    ("评价怎么样", 2),
    ("观感如何", 2),
    ("影迷评价", 2),
    ("网友评分", 2),
    ("评分详情", 2),
    ("打分情况", 2),
    ("星评", 2),
    ("几颗星", 2),

    # ========== query_year (3) - 查询年份 ==========
    ("流浪地球是哪一年上映的？", 3),
    ("这部电影什么时候上映的？", 3),
    ("战狼2是哪年的？", 3),
    ("哪一年拍的", 3),
    ("上映时间", 3),
    ("什么时候的电影", 3),
    ("哪年出品", 3),
    ("发行年份", 3),
    ("什么时候的片子", 3),
    ("年代", 3),
    ("出品年份", 3),
    ("哪一年", 3),
    ("何时的电影", 3),
    ("上映日期", 3),
    ("年份信息", 3),
    ("什么时候首映", 3),
    ("发行时间", 3),
    ("制作年份", 3),
    ("几几年的", 3),
    ("哪年上映", 3),
    ("泰坦尼克号哪年的", 3),
    ("复仇者联盟4什么时候", 3),
    ("蜘蛛侠哪年", 3),
    ("蝙蝠侠黑暗骑士年份", 3),
    ("芳华哪年", 3),
    ("集结号年份", 3),
    ("唐山大地震时间", 3),
    ("让子弹飞哪年", 3),
    ("一步之遥时间", 3),
    ("长城哪年", 3),
    ("上映档期", 3),
    ("首映礼", 3),
    ("公映时间", 3),
    ("院线时间", 3),
    ("上线时间", 3),
    ("播出时间", 3),
    ("首播年份", 3),
    ("重映时间", 3),
    ("什么时候拍", 3),
    ("拍摄年份", 3),
    ("开机时间", 3),
    ("杀青时间", 3),
    ("后期年份", 3),
    ("立项时间", 3),
    ("备案年份", 3),
    ("问世时间", 3),
    ("诞生年份", 3),
    ("问世年代", 3),
    ("出品时间", 3),
    ("发行日期", 3),
    ("首映日期", 3),
    ("上映年代", 3),
    ("哪年拍的", 3),
    ("什么时候出品", 3),
    ("什么时候发行", 3),
    ("什么时候制作", 3),
    ("电影年代", 3),

    # ========== query_genre (4) - 查询类型 ==========
    ("流浪地球是什么类型的电影？", 4),
    ("这部电影属于什么类型？", 4),
    ("喜剧片推荐", 4),
    ("科幻电影有哪些", 4),
    ("什么类型", 4),
    ("属于哪类", 4),
    ("哪种题材", 4),
    ("类型标签", 4),
    ("风格是什么", 4),
    ("什么题材", 4),
    ("类型信息", 4),
    ("电影分类", 4),
    ("什么片", 4),
    ("哪种类型", 4),
    ("题材", 4),
    ("类型", 4),
    ("风格", 4),
    ("什么风格", 4),
    ("影片类型", 4),
    ("电影种类", 4),
    ("动作片", 4),
    ("爱情片", 4),
    ("悬疑片", 4),
    ("惊悚片", 4),
    ("战争片", 4),
    ("传记片", 4),
    ("奇幻片", 4),
    ("动画片", 4),
    ("剧情片", 4),
    ("喜剧电影", 4),
    ("科幻片推荐", 4),
    ("悬疑类型", 4),
    ("电影题材", 4),
    ("类型划分", 4),
    ("影片风格", 4),
    ("电影流派", 4),
    ("类型归属", 4),
    ("分类标签", 4),
    ("标签类型", 4),
    ("影片分类", 4),
    ("电影题材有哪些", 4),
    ("属于什么片", 4),
    ("算哪种电影", 4),
    ("归为什么类型", 4),
    ("什么类别的电影", 4),
    ("哪类影片", 4),
    ("片种", 4),
    ("电影类目", 4),
    ("电影体裁", 4),
    ("影片体裁", 4),
    ("是什么片", 4),
    ("类型查询", 4),
    ("风格查询", 4),
    ("题材查询", 4),
    ("分类查询", 4),
    ("种类查询", 4),
    ("这是什么类型的片子", 4),
    ("这部电影什么风格", 4),
    ("该片类型", 4),
    ("该片题材", 4),
]


# ===================== 数据集定义 =====================
class IntentDataset(Dataset):
    """意图识别数据集"""
    def __init__(self, data, tokenizer, max_len=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }


# ===================== 模型定义 =====================
class IntentModel(nn.Module):
    """基于BERT的意图识别模型"""
    def __init__(self, num_classes=5):
        super(IntentModel, self).__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


# ===================== 训练函数 =====================
def train_model(model, train_loader, optimizer, scheduler, criterion, epochs=10):
    """训练模型"""
    model.train()
    best_loss = float('inf')

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        print(f"📊 Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        # 保存最优模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "intent_model.pth")
            print(f"💾 模型已保存（loss={avg_loss:.4f}）")

    print("🎉 训练完成！")


def evaluate_model(model, test_loader):
    """评估模型"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"📈 测试准确率: {accuracy:.4f}")
    return accuracy


def predict_intent(model, tokenizer, text):
    """单条预测"""
    model.eval()
    encoding = tokenizer(
        text,
        max_length=64,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        _, predicted = torch.max(outputs, 1)
        intent = INTENT_MAP[predicted.item()]
        confidence = torch.softmax(outputs, dim=1)[0][predicted.item()].item()

    return intent, confidence


# ===================== 主函数 =====================
if __name__ == "__main__":
    print("🚀 开始训练意图识别模型...")

    # 初始化tokenizer
    print("📥 加载BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # 创建数据集
    dataset = IntentDataset(TRAIN_DATA, tokenizer)

    # 划分训练集和验证集（80%训练，20%验证）
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    print(f"📊 训练集: {len(train_dataset)}条, 验证集: {len(val_dataset)}条")

    # 初始化模型
    model = IntentModel(num_classes=5).to(DEVICE)

    # 优化器和损失函数
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # 学习率调度器
    total_steps = len(train_loader) * 10
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # 训练
    print("\n🏋️ 开始训练...")
    train_model(model, train_loader, optimizer, scheduler, criterion, epochs=10)

    # 评估
    print("\n🔍 评估模型...")
    evaluate_model(model, val_loader)

    # 加载最佳模型进行测试
    print("\n📝 加载最佳模型进行测试...")
    model.load_state_dict(torch.load("intent_model.pth", map_location=DEVICE))

    test_questions = [
        "流浪地球的导演是谁？",
        "周星驰演过哪些电影？",
        "这部电影评分多少？",
        "哪一年上映的？",
        "什么类型的电影？"
    ]

    print("\n🎯 预测测试：")
    for q in test_questions:
        intent, conf = predict_intent(model, tokenizer, q)
        print(f"  问句: {q}")
        print(f"  → 意图: {intent} (置信度: {conf:.4f})\n")

    print("✅ 模型训练完成！权重已保存至 intent_model.pth")
    print("💡 提示：若无法下载bert-base-chinese，可改用 bert-base-chinese-local 或联网下载")
