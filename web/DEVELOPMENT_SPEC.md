# Osteoporosis ML Predictor Website Development Specification

## 1. 技术选型

当前版本采用无依赖静态网页：

- `index.html`：页面结构和内容。
- `calculator.html`：独立模型计算页面。
- `styles.css`：视觉系统、响应式布局、动画和组件状态。
- `app.js`：表单交互、BMI/OST 计算、演示风险输出。
- `model_service.py`：本地模型 API，加载保存的堆叠模型并返回三分类概率。

静态页面可部署到 GitHub Pages、Netlify、Vercel Static、服务器 Nginx 静态目录或期刊补充网站。若需要真实模型预测，必须同时部署模型 API。

## 2. 视觉规范

### 2.1 风格关键词

- Minimal editorial
- Clinical research
- White space
- Photo-led sections
- Small uppercase labels
- High-contrast typography

### 2.2 颜色

| Token | Value | 用途 |
|---|---:|---|
| `--ink` | `#111318` | 主文字 |
| `--muted` | `#6d717b` | 辅助文字 |
| `--line` | `#e8eaf0` | 分割线 |
| `--blue` | `#243cc9` | 品牌主色 |
| `--teal` | `#0d9488` | 正常/低风险 |
| `--amber` | `#c77700` | 骨量减少 |
| `--red` | `#c4362e` | 骨质疏松高风险 |
| `--paper` | `#f7f8fb` | 区块背景 |

### 2.3 字体

- 英文优先：Inter、Arial、Helvetica、system-ui。
- 中文优先：PingFang SC、Microsoft YaHei、Noto Sans CJK SC、sans-serif。
- 论文文档仍按此前要求使用中文宋体、英文 Times New Roman；网页本身采用现代无衬线更符合界面阅读。

## 3. 前端组件

| Component | 说明 |
|---|---|
| Header | 固定宽度导航，移动端自动换行 |
| Hero | 大标题、摘要、统计指标 |
| Image Strip | 三张医学/科研图像，形成视觉可信度 |
| Evidence Blocks | 展示数据来源和模型结论 |
| Calculator | 输入表单、自动计算、预测结果 |
| Validation Table | 展示关键验证结果 |
| Footer | 研究声明和版本信息 |

## 4. 计算逻辑

当 `http://127.0.0.1:8038/api/predict` 可访问时，计算器调用本地模型 API。该 API 实际加载保存的 `web/model_artifacts/model.pkl`，并按 18 个 LASSO 入模特征顺序构建输入矩阵：

```text
BMI, Weight, BRI, Waist, Age, Height, ALP, Race, Creatinine, eGFR,
Physical_Activity, Energy, Income, DBP, Calcium, Education, Gender, SBP
```

当前 API 的预处理是根据本地 NHANES `v2_rf` 数据重建标准化流程，用于网站评审和演示。正式上线时应将训练阶段的填补器、编码器、派生变量和标准化器保存为明确版本化的 pipeline。

当请求体包含 `"include_shap": true` 时，API 使用 SHAP `PermutationExplainer` 为当前预测类别计算 local SHAP explanation，并返回 top feature contributions。由于最终模型是 stacking ensemble，单次 SHAP 解释约需 10-15 秒，因此前端仅在用户点击 `Calculate with model` 时计算，不在每次输入变化时自动计算。

当模型 API 不可访问时，前端回退到透明、可解释的演示评分逻辑：

- BMI = weight / height²。
- OST = floor[0.2 × (weight kg - age years)]。
- 年龄升高、OST 降低、BMI 偏低、ALP 升高、低活动水平会提高异常风险。
- 输出三类比例归一化至 100%。

回退逻辑仅用于网页交互展示，不应写入论文方法学作为最终模型。

## 5. 正式模型 API 规范

正式部署时建议新增后端接口：

```http
POST /api/predict
Content-Type: application/json
```

请求体：

```json
{
  "age": 66,
  "sex": "female",
  "race_ethnicity": "non_hispanic_white",
  "height_cm": 158,
  "weight_kg": 54,
  "waist_cm": 82,
  "sbp": 132,
  "dbp": 78,
  "alp": 96,
  "calcium": 930,
  "creatinine": 72.5,
  "energy": 1900,
  "activity": "moderate",
  "education": 3,
  "income": 2,
  "include_shap": true
}
```

响应体：

```json
{
  "model_version": "stacking_20260113",
  "preprocessing_version": "oe_rf_v1",
  "prediction": "Osteopenia",
  "probabilities": {
    "Normal": 0.31,
    "Osteopenia": 0.52,
    "Osteoporosis": 0.17
  },
  "explanation": {
    "method": "SHAP PermutationExplainer",
    "class_name": "Osteopenia",
    "base_value": 0.57,
    "output_value": 0.68,
    "top_features": [
      {"feature": "Weight", "contribution": 0.085}
    ]
  },
  "ost_score": -2,
  "recommendation": "Consider DXA referral according to clinical context."
}
```

## 6. 安全与合规

1. 页面不得声称可以诊断骨质疏松。
2. 结果页必须提示 DXA 仍为确认骨密度状态的重要检查。
3. 种族/族群变量只用于模型校准、外部适用性和公平性评估，不得作为单独诊断规则展示。
4. 不收集姓名、手机号、身份证号等直接身份标识。
5. 若部署后端，应使用 HTTPS。
6. 如记录日志，只记录匿名输入范围和预测结果，不记录可识别个人身份的信息。

## 7. 后续开发任务

1. 将训练时预处理流程保存为可复现 pipeline。
2. 用 FastAPI 或 Flask 封装真实模型接口。
3. 为 `/api/predict` 增加单元测试和边界值测试。
4. 部署到公开 HTTPS 域名。
5. 在论文中加入网站 URL、访问日期、版本号和用途声明。
