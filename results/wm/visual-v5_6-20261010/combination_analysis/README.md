# V5.6 J/X 与 axis 独立复核

仅分析明确传入的已完成结果。方向统计的运动轴、≥1px 轴、静止轴和零预测分别保留；没有重训或重新选择参数。

## 已看 Dev：固定原始 000→111

- J: dx MSE 0.000118380737；motion skill 0.435273342254914。
- X: dx MSE 9.6283084e-05；motion skill 0.5315224451096928。
- J−X dx MSE = 2.20976534e-05；配对 CI [1.569848113597269e-05, 2.8509053023389415e-05]。

## Fresh64：固定 all32 / old16 / mixed16

| 子集 | J dx MSE | X dx MSE | J−X [配对 95% CI] |
|---|---:|---:|---|
| all_views | 0.000106922162 | 8.30314479e-05 | 2.38907138e-05 [1.5458311569102207e-05, 3.20021708254026e-05] |
| old_joint_views | 0.000100465823 | 7.89785175e-05 | 2.14873051e-05 [8.542633912619563e-06, 3.462854859907289e-05] |
| mixed_object_views | 0.000113378501 | 8.70843784e-05 | 2.62941224e-05 [1.6555889122424713e-05, 3.627792044514979e-05] |

4 个黄色路径、逐轴方向、静止轴和 center 负控全部保留在 JSON；不以子组挑选赢家。

## 同输入 axis：水平输出被垂直变化影响

| reader / 目标 | y-only false-x RMS px [95% CI] | x-only 错 / xy 错 | 正确→错 / 错→正确 | ≥1px 分母 |
|---|---|---:|---:|---:|
| global_W/A | 0.820127 [0.7612815238462588, 0.8792280090813402] | 45 / 64 | 40 / 21 | 510 |
| global_W/B | 0.809332 [0.7474186426528919, 0.8712309183364684] | 56 / 71 | 38 / 23 | 512 |
| lane_W/A | 0.912493 [0.8433549216179743, 0.9823095173573773] | 24 / 30 | 19 / 13 | 510 |
| lane_W/B | 0.841271 [0.781088421091671, 0.9006732709224722] | 29 / 43 | 28 / 14 | 512 |
| J/A | 0.821332 [0.7626536660328214, 0.8803071584614705] | 45 / 64 | 40 / 21 | 510 |
| J/B | 0.810236 [0.7482476329937074, 0.872050305586035] | 57 / 69 | 37 / 25 | 512 |
| X/A | 1.016123 [0.9452766108717551, 1.0867544118115848] | 27 / 48 | 34 / 13 | 510 |
| X/B | 0.940086 [0.8742962485345713, 1.006663924142344] | 35 / 51 | 32 / 16 | 512 |

y-only 项和二阶 interaction 均保留在 JSON；xy 错误变化同时包含伤害与救回，不能全部归因到 y-only 项。

![概览](overview.png)

## 公平比较与范围

- J vs X uses the same256 worlds, same X8192 reference, free-intercept kernel and alpha; J duplicated supervision collapses exactly to4096 rows. This contrast changes target-object sign-combination coverage.
- Fresh64 confirms object-sign composition only. Within-object x/y remain coupled; fresh32 views are not32 independent worlds.
- Known000-to111 has256 worlds with2 variants. Fresh all32 has64 new worlds. They use different populations and must never be pooled or treated as one headline denominator.
- Old lane uses restricted spatial access and the old N+W reference. J/X use full spatial access and the new X reference. J/X-versus-lane is not an isolated coverage effect.
- All reports retain both raw MSE and directional counts. Static targets are excluded from sign denominators and report false-motion RMS separately; zero predictions are neither correct nor wrong sign and are counted separately.
- At-least1px direction counts are separate from all nonzero target counts. Mean-motion skill is null if any fixed-baseline denominator is zero.
- New intervals added here are conditional posthoc paired-world intervals, not multiplicity-corrected new gates. Only the already registered freshall32 J-X comparison is the prespecified primary.
- Axis interventions and knownDev remain exploratory. No memory-module benefit, original gate recovery or automatic P0 authorization follows.
