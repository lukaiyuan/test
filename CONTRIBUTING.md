# 🤝 贡献指南

感谢您对XGBoost多因子量化交易策略系统的贡献兴趣！

## 🚀 快速开始

1. Fork此仓库
2. 创建您的功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的变更 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## 📋 开发流程

### 1. 环境设置
```bash
# 克隆仓库
git clone https://github.com/your-username/xgboost-quant-strategy.git
cd xgboost-quant-strategy

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2. 代码规范
- 使用Black进行代码格式化
- 使用flake8进行代码检查
- 使用isort进行导入排序
- 遵循PEP 8编码规范

### 3. 测试要求
- 所有新功能必须包含测试
- 测试覆盖率应保持在90%以上
- 运行 `pytest` 确保所有测试通过

### 4. 提交信息格式
使用以下格式编写提交信息：
```
<type>(<scope>): <subject>

<body>

<footer>
```

类型包括：
- `feat`: 新功能
- `fix`: Bug修复
- `docs`: 文档更新
- `style`: 代码格式化
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建过程或辅助工具的变动

## 🐛 报告Bug

请使用GitHub Issues报告Bug，包含以下信息：
- Bug描述
- 重现步骤
- 预期行为
- 实际行为
- 环境信息（Python版本、操作系统等）

## 💡 功能请求

我们欢迎新功能建议！请在Issue中详细描述：
- 功能描述
- 使用场景
- 预期收益
- 实现建议

## 📞 联系方式

如有任何问题，请通过以下方式联系：
- GitHub Issues
- 邮件：developer@xgboost-quant.com

感谢您的贡献！🎉
