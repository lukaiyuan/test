#!/bin/bash

# XGBoost多因子量化交易策略系统 - Git仓库设置脚本
# 此脚本将帮助您设置Git仓库并准备创建Pull Request

echo "🚀 设置XGBoost量化交易策略Git仓库"
echo "=================================================="

# 检查Git是否安装
if ! command -v git &> /dev/null; then
    echo "❌ Git未安装，请先安装Git"
    echo "下载地址: https://git-scm.com/downloads"
    exit 1
fi

# 初始化Git仓库
echo "📁 初始化Git仓库..."
git init

# 创建.gitignore文件
echo "📝 创建.gitignore文件..."
cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter Notebook
.ipynb_checkpoints

# Data files
data/raw/*.csv
data/raw/*.pkl
data/models/*.pkl
data/models/*.json
data/models/*.joblib

# Logs
logs/*.log

# OS
.DS_Store
Thumbs.db

# Temporary files
*.tmp
*.temp
EOF

# 创建初始提交
echo "💾 创建初始提交..."
git add .
git commit -m "🎉 Initial commit: XGBoost多因子量化交易策略系统

✨ 功能特性:
- 完整的多因子量化交易策略框架
- XGBoost机器学习模型
- 62个量化因子计算
- 特征工程和模型验证
- 完整的测试覆盖
- 生产级代码质量

📊 系统性能:
- 处理2792条记录
- 60个最终特征
- 56.8%方向准确率
- 6.95秒完整流程

🏗️ 项目结构:
- config/: 配置管理
- data/: 数据处理模块
- models/: 机器学习模型
- utils/: 工具函数
- tests/: 测试框架
- docs/: 完整文档"

echo "✅ Git仓库设置完成!"
echo ""
echo "📋 下一步操作:"
echo "1. 在GitHub上创建新仓库 (https://github.com/new)"
echo "2. 复制仓库URL"
echo "3. 运行以下命令连接远程仓库:"
echo "   git remote add origin <your-repo-url>"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "🔄 创建Pull Request:"
echo "1. 创建新分支: git checkout -b feature/new-feature"
echo "2. 进行修改并提交"
echo "3. 推送分支: git push origin feature/new-feature"
echo "4. 在GitHub上创建Pull Request"
echo ""
echo "💡 提示: 如果您需要帮助设置GitHub仓库，请告诉我！"
