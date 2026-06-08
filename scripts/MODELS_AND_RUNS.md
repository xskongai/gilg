# 模型接入与运行命令备份

本文件记录 gilg 项目支持的 LLM 后端、各模型的接入方式，以及 baseline / proposed 两路评测的运行命令。用于实验复现和备份。

## 1. 架构要点

- 所有生成模型通过 `src/gilg/generation/llm.py` 的 `get_llm()` 统一构造（基于 LangChain）。
- 模型选择由 `config/config.py` 的 `set_model(name)` 按模型名自动路由到对应后端，**业务代码无需改动**。
- 评测裁判（judge）固定为 **GPT-4o**（`src/gilg/evaluation/judge.py`，用 OpenAI 官方 SDK），与生成端独立。**因此任何模型跑评测都需要 `OPENAI_API_KEY`。**
- prompt 用 Jinja2 模板（`prompts/en/`、`prompts/zh/`），不写死在代码里。
- `invoke_text()`（llm.py）统一提取各后端返回的纯文本，避免 chat 后端的 metadata 污染输出。

## 2. 支持的后端

| 后端 | 适用模型 | base_url | 需要的环境变量 |
|---|---|---|---|
| `openai` | GPT 系列（gpt-4o, gpt-4o-mini, o1, o3-mini ...） | 官方默认 | `OPENAI_API_KEY` |
| `gemini` | Google Gemini（gemini-2.5/3.5-flash ...）、AI Studio 上的 Gemma | 官方默认 | `GOOGLE_API_KEY` |
| `openai_compat` | 任意 OpenAI 兼容端点：DeepSeek、Qwen/DashScope、Moonshot、GLM、自建 vLLM/SGLang | 由预设或逃生通道指定 | 见预设表 |
| `ollama` | 本地 Ollama（mistral, qwen2.5, llama3.x, gemma 等，含各种小模型） | 本地 | 无 |
| `hf_hub` / `local` | HuggingFace 端点 / 本地 transformers | — | `HF_TOKEN` |

## 3. 模型名 → 后端 路由规则（set_model）

按以下顺序判断（`config/config.py` 的 `_resolve_model`）：

1. 显式前缀逃生通道：`openai:`、`gemini:`、`ollama:`、`compat:model@url`
2. 已知云端家族前缀：`gpt`/`o1`/`o3`/`o4`/`chatgpt` → openai；`gemini` → gemini
3. OpenAI 兼容预设表 `_COMPAT_PRESETS` 前缀匹配（deepseek、qwen-、moonshot、glm）
4. 兜底：其余一律当本地 Ollama 模型

`_COMPAT_PRESETS`（仓库默认）：

```python
_COMPAT_PRESETS = {
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "qwen-":    ("https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
    "moonshot": ("https://api.moonshot.cn/v1", "MOONSHOT_API_KEY"),
    "glm":      ("https://open.bigmodel.cn/api/paas/v4", "ZHIPU_API_KEY"),
}
```

### 本地自定义：Qwen3.7-Max 专属部署（maas）

`qwen3.7-max` 以 `qwen3` 开头，不匹配 `qwen-` 前缀，会被误判到 Ollama。
本项目在 `_COMPAT_PRESETS` 中**额外加了一条** `qwen3` 预设，指向专属 maas endpoint：

```python
    "qwen3": ("https://ws-qda2js0k5wga6npk.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1", "QWEN_API_KEY"),
```

注意事项：
- 必须用专属 maas URL（不是通用 dashscope.aliyuncs.com），否则 key 与地址不匹配，报 401。
- API key 放在 `.env` 的 `QWEN_API_KEY`。
- Qwen3 默认开 thinking 模式，会把答案放进 reasoning_content、content 留空。已在 `llm.py` 的 `openai_compat` 分支对 qwen 模型加 `extra_body={"enable_thinking": False, "chat_template_kwargs": {"enable_thinking": False}}` 关闭。

## 4. 已接入并跑过的模型

| 模型 | --model 取值 | 后端 | 备注 |
|---|---|---|---|
| 本地 Qwen2.5 | `qwen2.5` | ollama | 本地免费，输出干净 |
| GPT-4o | `gpt-4o` | openai | 论文同款，也是固定 judge |
| Gemini 3.5 Flash | `gemini-3.5-flash` | gemini | 论文用 2.5，被限后改 3.5 |
| Gemma 4 31B (AI Studio) | `gemini:gemma-4-31b-it` | gemini | 走 Gemini API，逃生通道 |
| DeepSeek | `deepseek-chat` | openai_compat | 用非思考版；v4-flash/reasoner 带 thinking 会污染，勿用 |
| Qwen3.7-Max | `qwen3.7-max` | openai_compat | 专属 maas 部署，需关 thinking |

## 5. 运行命令

测试集：`data/testsets/queries_en_paper20.txt`（作者 20 题，前 16 completion + 后 4 description）。
每个模型跑两路：baseline（裸模型，加 `--baseline`）和 proposed（完整 RAG+CoT，不加）。
`--temperature 0` 保证可复现。结果存到 `results/runs/<时间戳>_<模型>[_baseline]/`。

### 本地 Qwen2.5

```bash
python scripts/run_evaluation.py --file data/testsets/queries_en_paper10.txt --lang en --model qwen2.5 --baseline --temperature 0
python scripts/run_evaluation.py --file data/testsets/queries_en_paper10.txt --lang en --model qwen2.5 --temperature 0
```

### GPT-4o

```bash
python scripts/run_evaluation.py --file data/testsets/queries_en_paper10.txt --lang en --model gpt-4o --baseline --temperature 0
python scripts/run_evaluation.py --file data/testsets/queries_en_paper10.txt --lang en --model gpt-4o --temperature 0
```

### Gemini 3.5 Flash

```bash
python scripts/run_evaluation.py --file data/testsets/queries_en_paper10.txt --lang en --model gemini-3.5-flash --baseline --temperature 0
python scripts/run_evaluation.py --file data/testsets/queries_en_paper10.txt --lang en --model gemini-3.5-flash --temperature 0
```

### Gemma 4 31B（Google AI Studio）

```bash
python scripts/run_evaluation.py --file data/testsets/queries_en_paper10.txt --lang en --model "gemini:gemma-4-31b-it" --baseline --temperature 0
python scripts/run_evaluation.py --file data/testsets/queries_en_paper10.txt --lang en --model "gemini:gemma-4-31b-it" --temperature 0
```

### DeepSeek（用非思考版）

```bash
python scripts/run_evaluation.py --file data/testsets/queries_en_paper10.txt --lang en --model deepseek-chat --baseline --temperature 0
python scripts/run_evaluation.py --file data/testsets/queries_en_paper10.txt --lang en --model deepseek-chat --temperature 0
```

### Qwen3.7-Max（专属 maas 部署）

```bash
python scripts/run_evaluation.py --file data/testsets/queries_en_paper10.txt --lang en --model qwen3.7-max --baseline --temperature 0
python scripts/run_evaluation.py --file data/testsets/queries_en_paper10.txt --lang en --model qwen3.7-max --temperature 0
```

## 6. .env 需要的密钥（按用到的模型填）

```
OPENAI_API_KEY=        # GPT 生成 + 固定 judge（必填）
GOOGLE_API_KEY=        # Gemini / Gemma(AI Studio)
DEEPSEEK_API_KEY=      # deepseek-chat
QWEN_API_KEY=          # qwen3.7-max 专属部署（本项目自定义预设读这个名）
DASHSCOPE_API_KEY=     # 通用 qwen-max / qwen-plus
COMPAT_API_KEY=        # compat: 逃生通道读这个
HF_TOKEN=              # HuggingFace 端点
```

## 7. 已知坑与排查

- **输出全空**：reasoning 模型（qwen3.x、deepseek-v4-flash/reasoner）默认开 thinking，答案进 reasoning_content、content 留空。对策：关 thinking（qwen 已在 llm.py 处理；deepseek 改用 `deepseek-chat`）。
- **输出带一大坨 metadata**（`content=... additional_kwargs=...`）：调用点没用 `invoke_text`。确认 baseline.py / first_pass.py / second_pass.py / verifier.py 都用 `invoke_text(self._llm, prompt)`。
- **401 认证失败**：base_url 与 key 不匹配（如专属 maas key 配了通用 dashscope 地址），或 key 环境变量名和预设里读的名字不一致。
- **model not found / 404**：模型名被误路由到本地 Ollama。检查 `set_model` 路由，必要时用 `compat:model@url` 逃生通道。
- **baseline 分偏高**：新模型（如 qwen3.7-max GA 66）自带对齐强，baseline 偏见本就少；且 judge 的 GA 易被中性词糊弄而虚高，属论文已知局限。重点看 baseline 与 proposed 的差值，而非绝对值。
