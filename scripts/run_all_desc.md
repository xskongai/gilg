## 设计

**顶部 CONFIG 区**集中所有可改项,文件名抽成变量（你要的）:

* `EN_FILE` / `ZH_FILE` —— 英文/中文题集文件名，只在这里改一处
* `TEMP` —— 温度（默认 0）
* 四个模型清单：`EN_BIG / ZH_BIG`（大模型）、`EN_SMALL / ZH_SMALL`（小模型）
* 四个开关：`RUN_EN_BIG / RUN_EN_SMALL / RUN_ZH_BIG / RUN_ZH_SMALL`

**分组**：英文部分（大+小）、中文部分（大+小），共四组。每组里每个模型**自动跑 baseline + proposed 两路**。

## 用法

```bash
# 默认:只跑小模型(本地免费),大模型组默认关(怕烧 API)
bash scripts/run_all.sh

# 只跑英文小模型
RUN_EN_SMALL=1 RUN_ZH_SMALL=0 bash scripts/run_all.sh

# 跑全部(含大模型,会调 API)
RUN_REST=1 bash scripts/run_all.sh

# 只跑中文那套(大+小)
RUN_EN_SMALL=0 RUN_EN_BIG=0 RUN_ZH_SMALL=1 RUN_ZH_BIG=1 bash scripts/run_all.sh
```

## 几个要点

1. **默认大模型组是关的**（`RUN_EN_BIG=0`/`RUN_ZH_BIG=0`），避免你一跑就烧一堆 API 钱。想跑大模型时显式开 `RUN_REST=1` 或单独开某组。
2. **容错**：某个模型失败（如 key 没配、模型没拉）不会中断整批，会打印 `!!! FAILED` 然后继续下一个。
3. **文件名确认**：脚本里 `ZH_FILE=queries_zh_paper20.txt` 是我刚给你的中文题集，确认你已经放进 `data/testsets/`。英文是 `queries_en_paper20.txt`。
4. **qwen3.7-max 在 ZH\_BIG 里**：如果你跑中文大模型组，它会用到——但记得它要专属 maas 配置和关 thinking（你之前配好的）。纯跑小模型组用不到。
5. **judge 仍是 gpt-4o**：所有跑法都需要 `OPENAI_API_KEY`，哪怕生成端是本地小模型。

建议先跑默认（小模型两套）验证整批能跑通:

```bash
bash scripts/run_all.sh
```

跑完结果都在 `results/runs/<时间戳>_<模型>[_baseline]/`,每个模型四个目录(en/zh × baseline/proposed)。要不要我再写一个**汇总脚本**,把这些 run 目录的 `eval_summary.csv` 自动收集成一张总表（模型 × 语言 × baseline/proposed 的 GA/GN/QR），方便你直接对比？
