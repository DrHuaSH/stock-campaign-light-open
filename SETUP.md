# 🚀 快速设置指南

## 环境要求

- **Python**: 3.10 - 3.14（推荐 3.11 或 3.12）
- **操作系统**: macOS / Linux / Windows
- **内存**: 建议 4GB+

## 第一步：创建虚拟环境（推荐）

Mac/Linux 用户需要使用虚拟环境：

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate
```

> 💡 **提示**：激活虚拟环境后，命令行前面会显示 `(venv)`

### 虚拟环境损坏时重建

如果遇到奇怪的错误，可以重建虚拟环境：

```bash
# 1. 删除旧环境
rm -rf venv

# 2. 重新创建
python3 -m venv venv
source venv/bin/activate

# 3. 重新安装依赖（见下一步）
```

## 第二步：安装依赖

### 🚀 方式一：使用本地缓存（最快，推荐）

项目已包含预下载的依赖包，无需联网即可秒装：

```bash
# 使用本地 packages 目录安装（超快）
./venv/bin/pip install --no-index --find-links=packages -r requirements.txt

# 或使用提供的脚本
./install_deps.sh
```

> ⚠️ **注意**：scipy 安装可能需要 2-5 分钟，请耐心等待，不要中断

### 方式二：使用国内镜像（需要联网）

#### 临时使用镜像
```bash
# 使用清华镜像加速
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 永久设置镜像
```bash
# 设置清华镜像为默认
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 然后再安装
pip install -r requirements.txt
```

### 其他国内镜像源

| 镜像源 | 地址 |
|--------|------|
| **清华** | `https://pypi.tuna.tsinghua.edu.cn/simple` |
| **阿里云** | `https://mirrors.aliyun.com/pypi/simple` |
| **中科大** | `https://pypi.mirrors.ustc.edu.cn/simple` |
| **豆瓣** | `https://pypi.douban.com/simple` |

---

### 📦 更新本地缓存（开发者）

如果需要更新本地包缓存（例如依赖版本变更）：

```bash
# 重新下载所有依赖到本地
./venv/bin/pip download -r requirements.txt -d packages
```

## 第三步：启动程序

```bash
# 确保虚拟环境已激活（看到前面的 venv）
streamlit run main.py
```

程序会自动打开浏览器，访问 `http://localhost:8501`

### 指定端口（如果 8501 被占用）

```bash
streamlit run main.py --server.port=8504
```

## 第四步：配置 API 密钥

在左侧边栏依次配置：

### 1. Tushare Pro Token（必填）
- 注册地址：https://tushare.pro/register
- 将获取的 Token 填入输入框

### 2. LLM API 配置（必填）
支持任何 OpenAI 兼容的 API：

| 服务商 | Base URL | Model |
|--------|----------|-------|
| **OpenAI** | `https://api.openai.com/v1` | `gpt-3.5-turbo` |
| **Kimi** | `https://api.moonshot.cn/v1` | `moonshot-v1-8k` |
| **DeepSeek** | `https://api.deepseek.com/v1` | `deepseek-chat` |
| **Ollama** | `http://localhost:11434/v1` | `llama2` |

💡 **快捷方式**：使用"快速选择预设"自动填充

### 3. Tavily API Key（可选）
- 用于增强分析的新闻搜索功能
- 注册地址：https://tavily.com/
- 免费版每月 1000 次调用

## 第五步：保存并测试

1. 点击 **💾 保存密钥** 按钮
2. 点击 **🧪 测试连接** 按钮验证 LLM API 是否正常

## 第六步：开始使用

选择功能开始分析：
- **📊 单股分析** - 分析单只股票
- **🔥 增强分析** - 三维度综合分析（需要 Tavily Key）
- **📈 批量分析** - 同时分析多只股票

---

## 🔄 下次使用

```bash
# 1. 进入项目目录
cd stock-campaign-light-open

# 2. 激活虚拟环境
source venv/bin/activate

# 3. 启动程序
streamlit run main.py
```

---

## ⚠️ 常见问题

**Q: pip 下载很慢？**  
A: 优先使用本地缓存安装：`./install_deps.sh`，无需下载秒装完成。如果本地缓存不存在，使用国内镜像加速：`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`

**Q: 安装 scipy 时卡住很久？**  
A: scipy 是大型科学计算库，安装需要 2-5 分钟，这是正常现象。请耐心等待，不要中断安装。

**Q: pip 出现 "Ignoring invalid distribution" 警告？**  
A: 这是由于 macOS 的 ._ 元数据文件导致的，可以安全忽略。如需清理：
```bash
cd venv/lib/python3.*/site-packages && rm -rf ._* 2>/dev/null
```

**Q: pip 报错 "externally-managed-environment"？**  
A: Mac 需要使用虚拟环境，按照上面的步骤创建并激活 venv 即可。

**Q: 启动时报 "端口已被占用"？**  
A: 使用其他端口启动：`streamlit run main.py --server.port=8504`

**Q: 应用一直显示 "Running..."？**  
A: 可能是依赖损坏，尝试重建虚拟环境并重新安装。

**Q: 配置保存在哪里？**  
A: 保存在 `user_config.json` 文件中，已加入 `.gitignore` 不会提交到 Git。

**Q: 没有 Tavily Key 能用吗？**  
A: 可以，单股分析和批量分析功能不受影响，只是增强分析的信息面功能无法使用。

**Q: 支持哪些数据源？**  
A: Tushare Pro（主）+ AKShare（备份），自动切换。

**Q: 如何彻底清理并重新开始？**  
A: 执行以下命令：
```bash
# 停止所有相关进程
pkill -f "streamlit run main.py"

# 删除虚拟环境
rm -rf venv

# 重新创建并安装
python3 -m venv venv
source venv/bin/activate
./install_deps.sh

# 启动
streamlit run main.py
```

---

## 🐛 故障排查

### 检查安装状态

```bash
# 验证关键依赖
source venv/bin/activate
python -c "import streamlit, pandas, numpy, scipy; print('✅ 所有依赖正常')"
```

### 查看错误日志

```bash
# 直接运行查看详细错误
streamlit run main.py 2>&1 | tee streamlit_error.log
```

### 端口冲突解决

```bash
# 查看占用 8501 的进程
lsof -i :8501

# 结束该进程
kill -9 <PID>
```

### Python 版本检查

```bash
# 检查 Python 版本（需要 3.10+）
python3 --version

# 如果不符合，使用 pyenv 安装合适版本
pyenv install 3.11.9
pyenv local 3.11.9
```
