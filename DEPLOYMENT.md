# 🖥️ 跨平台部署指南：Arch Linux + Hyprland / Windows 10

本项目支持在 Arch Linux + Hyprland 和 Windows 10 双环境下部署，以下是详细步骤。

---

## 📋 环境对比速查

| 组件 | Arch Linux + Hyprland | Windows 10 |
|------|----------------------|------------|
| Python | `pacman` / `pyenv` | 官网安装 / Microsoft Store |
| Neo4j | Docker (推荐) / AUR | Docker / Neo4j Desktop |
| 浏览器 | Firefox/Chrome (Wayland) | Chrome/Edge |
| 终端 | kitty / alacritty / foot | PowerShell / CMD / Windows Terminal |
| 包管理 | pacman / yay | pip / chocolatey |

---

## 一、通用准备（两种系统都需要）

### 1.1 克隆/下载项目代码

```bash
# 将项目放到合适的位置
mkdir -p ~/Projects
cd ~/Projects
# 解压或克隆项目到 kgqa_movie 目录
cd kgqa_movie
```

### 1.2 确保 Python 3.8~3.10 可用

```bash
python --version    # 应显示 Python 3.8/3.9/3.10
```

**如果不符合**，见下方各系统的 Python 安装方法。

---

## 二、Arch Linux + Hyprland 部署

### 2.1 安装 Python（如未安装）

```bash
# 方式1：直接用 pacman
sudo pacman -S python python-pip python-virtualenv

# 方式2：用 pyenv 管理多版本（推荐）
yay -S pyenv          # 或 paru -S pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc
pyenv install 3.10.13
pyenv global 3.10.13
```

### 2.2 创建 Python 虚拟环境

在 Hyprland 下推荐用终端模拟器（kitty/alacritty/foot）执行：

```bash
cd ~/Projects/kgqa_movie
python -m venv kgqa_env
source kgqa_env/bin/activate
```

> 💡 **Hyprland 提示**：可以将上述命令绑定到 hyprland.conf 的快捷键，快速启动开发环境。

### 2.3 安装 Neo4j（Arch 有三种方式）

#### 方式 A：Docker 部署（⭐ 最推荐，最干净）

```bash
# 安装 Docker（如未安装）
sudo pacman -S docker
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
# 重新登录生效

# 启动 Neo4j 容器
docker run -d \
    --name neo4j-kgqa \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/123456 \
    -e NEO4J_dbms_memory_heap_max__size=1G \
    -v ~/Projects/kgqa_movie/neo4j_data:/data \
    -v ~/Projects/kgqa_movie:/var/lib/neo4j/import \
    neo4j:5

# 查看状态
docker ps
docker logs neo4j-kgqa
```

> 📌 **关键**：`-v ~/Projects/kgqa_movie:/var/lib/neo4j/import` 将项目目录挂载到容器的 import 目录，这样 `LOAD CSV` 可以直接读取 `movies_data.csv`。

#### 方式 B：AUR 安装

```bash
yay -S neo4j-community
# 启动
sudo systemctl enable --now neo4j
# 默认密码修改
cypher-shell -u neo4j -p neo4j
# 然后按提示修改密码为 123456
```

#### 方式 C：Binary 手动安装

```bash
cd /opt
sudo curl -O https://neo4j.com/artifact.php?name=neo4j-community-5.20.0-unix.tar.gz
sudo tar -xzf neo4j-community-5.20.0-unix.tar.gz
sudo ln -s /opt/neo4j-community-5.20.0 /opt/neo4j
# 配置环境变量
echo 'export PATH="/opt/neo4j/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
# 启动
neo4j console
```

### 2.4 安装 Python 依赖

```bash
# 确保在虚拟环境中
source ~/Projects/kgqa_movie/kgqa_env/bin/activate

# 安装核心依赖
pip install neo4j==5.27.0 torch==1.13.1 transformers==4.30.2 \
    streamlit==1.24.0 fastapi==0.103.1 uvicorn==0.23.2 \
    scikit-learn==1.3.0 pandas==2.0.3 requests==2.31.0 \
    sentencepiece==0.1.99

# Arch 可能需要额外安装
gcc --version   # 确认有编译器，torch需要
```

### 2.5 数据导入 Neo4j

```bash
cd ~/Projects/kgqa_movie
python data_preprocess.py

# 如果使用 Docker Neo4j（文件已挂载，直接执行）
python neo4j_import.py
```

### 2.6 启动系统

**终端 1 - 启动后端：**

```bash
cd ~/Projects/kgqa_movie
source kgqa_env/bin/activate
uvicorn main_api:app --reload --host 0.0.0.0 --port 8000
```

**终端 2 - 启动前端：**

```bash
cd ~/Projects/kgqa_movie
source kgqa_env/bin/activate
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

**打开浏览器访问：**

```bash
# Hyprland 下用浏览器打开（推荐 Firefox/Chrome with Wayland）
firefox http://localhost:8501 &
# 或
chrome --enable-features=UseOzonePlatform --ozone-platform=wayland http://localhost:8501 &
```

> 💡 **Hyprland 窗口布局建议**：
> ```conf
> # ~/.config/hypr/hyprland.conf
> # 左右分屏，左边放终端（后端），右边放浏览器
> bind = ALT, K, exec, kitty --directory ~/Projects/kgqa_movie -e sh -c "source kgqa_env/bin/activate && uvicorn main_api:app --reload"
> bind = ALT, L, exec, firefox http://localhost:8501
> ```

---

## 三、Windows 10 部署

### 3.1 安装 Python

1. 官网下载 Python 3.10：https://www.python.org/downloads/release/python-31011/
2. **安装时勾选** "Add Python to PATH"
3. 验证：

```powershell
python --version
pip --version
```

### 3.2 创建虚拟环境

用 PowerShell 或 Windows Terminal：

```powershell
cd C:\Projects\kgqa_movie
python -m venv kgqa_env
.\kgqa_env\Scripts\Activate.ps1
# 如果执行策略限制，先运行：
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3.3 安装 Neo4j（Windows 两种方案）

#### 方案 A：Docker Desktop（推荐）

1. 安装 Docker Desktop：https://www.docker.com/products/docker-desktop
2. 启动 Docker Desktop
3. 运行 Neo4j：

```powershell
# PowerShell
docker run -d `
    --name neo4j-kgqa `
    -p 7474:7474 -p 7687:7687 `
    -e NEO4J_AUTH=neo4j/123456 `
    -e NEO4J_dbms_memory_heap_max__size=1G `
    -v C:\Projects\kgqa_movie\neo4j_data:/data `
    -v C:\Projects\kgqa_movie:/var/lib/neo4j/import `
    neo4j:5

# 如果 Docker volume 映射有问题，可以手动复制文件到容器：
docker cp movies_data.csv neo4j-kgqa:/var/lib/neo4j/import/
```

#### 方案 B：Neo4j Desktop

1. 下载 Neo4j Desktop 5.x：https://neo4j.com/download
2. 安装并启动
3. 创建 Project：`KGQA_Movie`
4. 添加 Local DBMS：`movie_kg`，密码设为 `123456`
5. Start 启动数据库
6. 浏览器访问 http://localhost:7474 验证

> ⚠️ **重要**：使用 Neo4j Desktop 时，需要将 `movies_data.csv` 复制到数据库的 import 文件夹：
> ```
> C:\Users\<你的用户名>\.Neo4jDesktop\relate-data\dbmss\dbms-<hash>\import\
> ```

### 3.4 安装 Python 依赖

```powershell
# 确保在虚拟环境中 (.\kgqa_env\Scripts\Activate.ps1)
pip install neo4j==5.27.0 torch==1.13.1 transformers==4.30.2 streamlit==1.24.0 fastapi==0.103.1 uvicorn==0.23.2 scikit-learn==1.3.0 pandas==2.0.3 requests==2.31.0 sentencepiece==0.1.99
```

如果遇到 `Microsoft Visual C++` 编译错误：

```powershell
# 安装 Visual C++ Build Tools
# https://visualstudio.microsoft.com/visual-cpp-build-tools/
# 或安装已编译好的 wheel
pip install --only-binary :all: torch transformers
```

### 3.5 数据导入

```powershell
cd C:\Projects\kgqa_movie
python data_preprocess.py
python neo4j_import.py
```

### 3.6 启动系统

**PowerShell 窗口 1 - 后端：**

```powershell
cd C:\Projects\kgqa_movie
.\kgqa_env\Scripts\Activate.ps1
uvicorn main_api:app --reload --host 0.0.0.0 --port 8000
```

**PowerShell 窗口 2 - 前端：**

```powershell
cd C:\Projects\kgqa_movie
.\kgqa_env\Scripts\Activate.ps1
streamlit run app.py --server.port 8501
```

**浏览器访问：**
- 前端：http://localhost:8501
- API 文档：http://localhost:8000/docs

---

## 四、Hyprland 环境特别优化

### 4.1 Wayland 下的浏览器支持

```bash
# Firefox 原生 Wayland
echo 'MOZ_ENABLE_WAYLAND=1' >> ~/.config/environment.d/envvars.conf

# Chrome/Chromium 原生 Wayland
chrome --enable-features=UseOzonePlatform --ozone-platform=wayland

# 或设置环境变量
echo 'CHROME_FLAGS="--enable-features=UseOzonePlatform --ozone-platform=wayland"' >> ~/.bashrc
```

### 4.2 推荐 hyprland.conf 工作区布局

```conf
# ~/.config/hypr/hyprland.conf

# 工作区 5 专门用于 KGQA 开发
workspace = name:kgqa, persistent:true

# 一键启动完整开发环境
bind = ALT SHIFT, K, exec, hyprctl dispatch workspace name:kgqa
bindr = ALT SHIFT, K, exec, ~/.local/bin/kgqa-start.sh

# 分屏规则
windowrulev2 = workspace name:kgqa, class:^(kitty)$, title:(kgqa-backend|kgqa-frontend)
windowrulev2 = workspace name:kgqa, class:^(firefox|chromium|chrome)$
```

### 4.3 一键启动脚本

创建 `~/.local/bin/kgqa-start.sh`：

```bash
#!/bin/bash
PROJECT_DIR="$HOME/Projects/kgqa_movie"
VENV="$PROJECT_DIR/kgqa_env/bin/activate"

# 切换到项目工作区
hyprctl dispatch workspace name:kgqa

# 启动 Neo4j（Docker）
if ! docker ps | grep -q neo4j-kgqa; then
    notify-send "KGQA" "正在启动 Neo4j..."
    docker start neo4j-kgqa
    sleep 5
fi

# 左半屏 - 后端
kitty --title kgqa-backend --directory "$PROJECT_DIR" sh -c "source '$VENV' && uvicorn main_api:app --reload; read"

# 等待后端启动
sleep 2

# 右半屏上半 - 前端
kitty --title kgqa-frontend --directory "$PROJECT_DIR" sh -c "source '$VENV' && streamlit run app.py; read"

# 右半屏下半 - 浏览器
firefox http://localhost:8501 &

notify-send "KGQA" "开发环境已启动 🎬"
```

```bash
chmod +x ~/.local/bin/kgqa-start.sh
```

### 4.4 终端中文字体配置

确保终端支持中文显示（Neo4j 查询结果含中文）：

```conf
# kitty.conf
font_family      JetBrains Mono
bold_font        JetBrains Mono Bold
font_size        12.0
# 确保有 fallback 中文字体
symbol_map U+4E00-U+9FFF,U+3400-U+4DBF Noto Sans CJK SC
```

---

## 五、跨系统共享数据（双系统用户）

如果你的 Arch 和 Windows 共享同一个 NTFS 数据盘：

```bash
# 在 Arch 下挂载 Windows 分区到 /mnt/data
# /etc/fstab
/dev/nvme0n1p3  /mnt/data  ntfs-3g  defaults,uid=1000,gid=1000,umask=022  0  0

# 项目放在共享盘
ln -s /mnt/data/Projects/kgqa_movie ~/Projects/kgqa_movie
```

Neo4j Docker 在两个系统下都可以指向同一数据目录：

```bash
# Linux
-v /mnt/data/Projects/kgqa_movie/neo4j_data:/data

# Windows (Docker Desktop)
-v D:\Projects\kgqa_movie\neo4j_data:/data
```

---

## 六、常见问题排查

### Q1: Arch 下 `pip install torch` 很慢/失败

```bash
# 使用清华镜像
pip install torch==1.13.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或用 pacman 安装（Arch 官方仓库有 torch）
sudo pacman -S python-pytorch
```

### Q2: Neo4j Docker 容器无法访问

```bash
# 检查防火墙
sudo ufw allow 7474/tcp
sudo ufw allow 7687/tcp

# 检查 Docker 网络
docker network ls
docker inspect neo4j-kgqa

# 查看日志
docker logs neo4j-kgqa
```

### Q3: Hyprland 下 Streamlit 弹不出浏览器

```bash
# Streamlit 使用 xdg-open，Wayland 下可能有问题
# 手动指定浏览器
export BROWSER=firefox
# 或在代码中设置
# echo '[browser]
# gatherUsageStats = false
# serverAddress = "localhost"
# ' >> ~/.streamlit/config.toml
```

### Q4: Windows 下 PowerShell 执行策略限制

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# 输入 Y 确认
```

### Q5: BERT 模型下载慢（两个系统都适用）

```bash
# 使用 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com
# 或在代码中设置
# import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 手动下载放到缓存目录
# ~/.cache/huggingface/hub/  (Linux)
# C:\Users\<用户名>\.cache\huggingface\hub\  (Windows)
```

### Q6: Arch 下 Docker 权限 denied

```bash
# 重新登录 docker 用户组
newgrp docker
# 或注销重新登录
```

---

## 七、一键部署脚本

### Arch Linux 一键脚本

```bash
#!/bin/bash
# deploy_arch.sh
set -e

echo "🚀 开始在 Arch Linux + Hyprland 上部署 KGQA..."

# 检查 Python
if ! command -v python &> /dev/null; then
    echo "📦 安装 Python..."
    sudo pacman -S --needed python python-pip python-virtualenv
fi

# 检查 Docker
if ! command -v docker &> /dev/null; then
    echo "📦 安装 Docker..."
    sudo pacman -S --needed docker
    sudo systemctl enable --now docker
    sudo usermod -aG docker $USER
    echo "⚠️ 请注销并重新登录以使 Docker 权限生效"
    exit 1
fi

# 创建虚拟环境
echo "🐍 创建虚拟环境..."
cd "$(dirname "$0")"
python -m venv kgqa_env
source kgqa_env/bin/activate

# 安装依赖
echo "📥 安装 Python 依赖..."
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 启动 Neo4j
echo "🗄️ 启动 Neo4j..."
if ! docker ps | grep -q neo4j-kgqa; then
    if docker ps -a | grep -q neo4j-kgqa; then
        docker start neo4j-kgqa
    else
        docker run -d \
            --name neo4j-kgqa \
            -p 7474:7474 -p 7687:7687 \
            -e NEO4J_AUTH=neo4j/123456 \
            -e NEO4J_dbms_memory_heap_max__size=1G \
            -v "$(pwd)/neo4j_data:/data" \
            -v "$(pwd):/var/lib/neo4j/import" \
            neo4j:5
    fi
    sleep 5
fi

# 预处理数据
echo "🔧 预处理数据..."
python data_preprocess.py

# 导入 Neo4j
echo "📊 导入知识图谱..."
python neo4j_import.py

echo "✅ 部署完成！"
echo ""
echo "启动后端: source kgqa_env/bin/activate && uvicorn main_api:app --reload"
echo "启动前端: source kgqa_env/bin/activate && streamlit run app.py"
echo "打开浏览器: firefox http://localhost:8501"
```

### Windows 一键脚本

```powershell
# deploy_windows.ps1
Write-Host "🚀 开始在 Windows 10 上部署 KGQA..." -ForegroundColor Green

# 检查 Python
if (!(Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "❌ 请先安装 Python 3.10" -ForegroundColor Red
    exit 1
}

# 检查 Docker
if (!(Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "⚠️ Docker 未安装，将使用 Neo4j Desktop 模式" -ForegroundColor Yellow
    Write-Host "请手动安装 Neo4j Desktop 并创建数据库"
}

# 创建虚拟环境
Write-Host "🐍 创建虚拟环境..." -ForegroundColor Cyan
$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectDir
python -m venv kgqa_env
.\kgqa_env\Scripts\Activate.ps1

# 安装依赖
Write-Host "📥 安装 Python 依赖..." -ForegroundColor Cyan
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 启动 Neo4j Docker
if (Get-Command docker -ErrorAction SilentlyContinue) {
    Write-Host "🗄️ 启动 Neo4j Docker..." -ForegroundColor Cyan
    $running = docker ps --format "{{.Names}}" | Select-String "neo4j-kgqa"
    if (!$running) {
        $exists = docker ps -a --format "{{.Names}}" | Select-String "neo4j-kgqa"
        if ($exists) {
            docker start neo4j-kgqa
        } else {
            docker run -d `
                --name neo4j-kgqa `
                -p 7474:7474 -p 7687:7687 `
                -e NEO4J_AUTH=neo4j/123456 `
                -e NEO4J_dbms_memory_heap_max__size=1G `
                -v "${projectDir}\neo4j_data:/data" `
                -v "${projectDir}:/var/lib/neo4j/import" `
                neo4j:5
        }
        Start-Sleep 5
    }
    
    # Docker 方式下复制 CSV 进容器
    docker cp movies_data.csv neo4j-kgqa:/var/lib/neo4j/import/
}

# 预处理数据
Write-Host "🔧 预处理数据..." -ForegroundColor Cyan
python data_preprocess.py

# 导入 Neo4j
Write-Host "📊 导入知识图谱..." -ForegroundColor Cyan
python neo4j_import.py

Write-Host "✅ 部署完成！" -ForegroundColor Green
Write-Host ""
Write-Host "启动后端: .\kgqa_env\Scripts\Activate.ps1; uvicorn main_api:app --reload" -ForegroundColor Yellow
Write-Host "启动前端: .\kgqa_env\Scripts\Activate.ps1; streamlit run app.py" -ForegroundColor Yellow
Write-Host "打开浏览器: http://localhost:8501" -ForegroundColor Yellow
```

---

## 八、系统启动后验证清单

| 检查项 | 命令/方法 | 预期结果 |
|--------|----------|----------|
| Neo4j 运行中 | `docker ps` / Neo4j Desktop | 看到 neo4j 进程 |
| Neo4j 浏览器 | http://localhost:7474 | 登录成功 |
| 后端运行中 | http://localhost:8000/docs | Swagger UI |
| 前端运行中 | http://localhost:8501 | Streamlit 界面 |
| 图谱有数据 | Neo4j 中 `MATCH (n) RETURN count(n)` | > 50 节点 |
| API 问答测试 | `curl "http://localhost:8000/kgqa?question=流浪地球的导演是谁"` | 返回 JSON 含答案 |

---

## 📎 附录：系统服务化（Arch Linux）

如果你想让 Neo4j 和 KGQA 后端开机自启：

### Neo4j Docker 自启

```bash
# Docker 容器已设置 -d，只需设置重启策略
docker update --restart unless-stopped neo4j-kgqa
```

### KGQA 后端 Systemd 服务

```ini
# ~/.config/systemd/user/kgqa-backend.service
[Unit]
Description=KGQA FastAPI Backend
After=docker.service

[Service]
Type=simple
WorkingDirectory=%h/Projects/kgqa_movie
ExecStart=%h/Projects/kgqa_movie/kgqa_env/bin/uvicorn main_api:app --host 0.0.0.0 --port 8000
Restart=on-failure
Environment=PATH=%h/Projects/kgqa_movie/kgqa_env/bin:/usr/bin

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable kgqa-backend
systemctl --user start kgqa-backend
systemctl --user status kgqa-backend
```

---

**有任何问题随时提问！**
