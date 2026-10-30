# The statistics behind AdapBench

A short account for readers who know e-values and anytime-valid inference. For the systems side, see the [README](../README.md) and the paper.

## Setup

Per benchmark we observe duet differences $d_t = (s_{1,t} - s_{2,t}) / (s_{1,t} + s_{2,t})$ between the two versions, averaged in blocks to damp autocorrelation and clipped to $[-h, h]$ with $h = 0.06$. Let $\mu$ be the mean of the clipped block means and $\varepsilon$ the duet equivalent of a 3&nbsp;% change. Each benchmark tests a composite band null:

$$H_0:\ |\mu| \le \varepsilon \qquad \text{vs.} \qquad H_1:\ |\mu| > \varepsilon.$$

- **Evidence.** Two betting games run at the band edges: $K_t^{+}$ bets on $\mu > \varepsilon$, $K_t^{-}$ on $\mu < -\varepsilon$, with aGRAPA stakes that use past samples only [2]. Under every $\mu$ in the null both are nonnegative supermartingales, so $E_t = \max(K_t^{+}, K_t^{-}) / 2$ is an e-process for $H_0$.
- **Claim.** Flag a change once $E_t \ge 1/\alpha_{\text{claim}}$. Ville's inequality bounds the false-claim probability by $\alpha_{\text{claim}}$ at any stopping time, so checking after every block costs nothing [1].
- **Futility.** The mirror games, betting on $\mu < \varepsilon$ and $\mu > -\varepsilon$, both crossing $2/\alpha_{\text{claim}}$ certify $|\mu| < \varepsilon$. The benchmark stops with no change.
- **Budget.** A benchmark that reaches its cap without either event is decided by a fixed-$n$ bootstrap CI at level $\alpha_{\text{pool}}$. The split $(\alpha_{\text{claim}}, \alpha_{\text{pool}})$ is calibrated on A/A data so the measured false-claim rate stays below $\alpha = 1$&nbsp;%.
- **Allocation.** The scheduler picks the next benchmark by its estimated distance to a decision, with a share of uniform exploration. Each test sees only its own benchmark's samples, and whether to sample it depends only on the past, so adaptive allocation keeps every test valid.

## Open questions

1. **Fully anytime-valid.** The bootstrap at budget is the one fixed-$n$ step, and its α split is calibrated, not proven. Can one e-process cover the whole run without losing the recall the bootstrap adds?
2. **Dependence.** Benchmark timings are autocorrelated and blocking is a heuristic. Which e-processes stay valid under mixing, and what power do they cost?
3. **Many benchmarks per commit.** Error control is per benchmark today. e-BH [3] would control the false discovery rate across the suite. How does it interact with adaptive allocation and early stopping?
4. **Evidence across commits.** Every commit starts at $K_0 = 1$. When consecutive commits don't touch a benchmark, could its evidence carry forward?
5. **Clipping.** With asymmetric tails, clipping to $[-h, h]$ shifts the tested mean. Is there a cheap correction, or a better functional to test?

## References

[1] Ramdas, Grünwald, Vovk, Shafer. [Game-theoretic statistics and safe anytime-valid inference](https://arxiv.org/abs/2210.01948). Statistical Science, 2023.\
[2] Waudby-Smith, Ramdas. [Estimating means of bounded random variables by betting](https://arxiv.org/abs/2010.09686). JRSS-B, 2024.\
[3] Wang, Ramdas. [False discovery rate control with e-values](https://arxiv.org/abs/2009.02824). JRSS-B, 2022.
