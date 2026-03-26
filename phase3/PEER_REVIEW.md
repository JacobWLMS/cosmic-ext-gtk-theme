# Phase 3 Peer Review

*Reviewer notes, 2026-03-26. Covers findings docs, experiment READMEs, and raw CSV data.*

---

## Findings From Raw Data Not Captured in Documentation

### 1. Dev Patel Outlier is EXPLAINED — Not a Mystery

The Exp 3 Part A CSV reveals Dev Patel's nearest neighbour is **Riz Ahmed with 0.859 ArcFace similarity**. That's not just "nearest" — that's near-duplicate territory. For comparison, the next-highest nearest-neighbour similarity in the dataset is Samuel L. Jackson → Trevor Noah at 0.655. Most are in the 0.3-0.45 range.

Dev Patel's 0.725 triangulation score isn't showing that the method works exceptionally well — it's showing that when your nearest neighbour is almost identical to you, interpolation trivially succeeds. This should be moved from serendipity.md to a methodological note: the outlier is an artefact of having near-duplicate celebrities in the database, not evidence of a high ceiling for the triangulation approach.

**Action**: Update serendipity.md to include the Riz Ahmed explanation. Consider whether Dev Patel should be excluded from aggregate statistics since his result is driven by database composition, not method quality.

### 2. The n_name_tokens Column Partially Explains the Bimodal Split

Exp 4 Part B CSV includes `n_name_tokens` per celebrity. The pattern:

| Celebrity | name_tokens score | n_name_tokens | Group |
|-----------|------------------|---------------|-------|
| Brad Pitt | 0.927 | 2 | Works |
| Morgan Freeman | 0.913 | 2 | Works |
| Will Smith | 0.916 | 2 | Works |
| Tom Cruise | 0.868 | 2 | Works |
| Leonardo DiCaprio | 0.780 | 1 | Works (barely) |
| Denzel Washington | 0.463 | 1 | Middle |
| Keanu Reeves | 0.341 | 1 | Fails |
| Robert Downey Jr | 0.262 | 2 | Fails |
| Johnny Depp | 0.109 | 1 | Fails |
| Tom Hanks | 0.052 | 1 | Fails |

**Observations**:
- All 2-token celebrities that work (Brad Pitt, Morgan Freeman, Will Smith, Tom Cruise) have scores > 0.86
- Robert Downey Jr is the exception: 2 tokens but only 0.262. His name is 3 words — "Robert Downey Jr" likely tokenises to 2 tokens but they may not correspond to the semantically meaningful parts
- Most 1-token celebrities fail (Tom Hanks 0.052, Johnny Depp 0.109, Keanu Reeves 0.341)
- DiCaprio is a 1-token outlier that works (0.780) — worth investigating his tokenisation

This isn't the full explanation (Robert Downey Jr breaks the pattern), but there's a clear correlation between having 2 name tokens and successful name-token injection. This is cheap to verify in Phase 4 — just check exact tokenisation for all 10 celebrities.

**Action**: Add n_name_tokens analysis to the bimodal encoding section of state_of_affairs.md. This is the first concrete predictor candidate.

### 3. Exp 4 Part C Has a Negative ArcFace Score

Denzel Washington's beach scene injection produced ArcFace = -0.096. That's not just "essentially random" — it's anti-correlated with the target identity. The documentation rounds this to the 0.155 mean across all scenes/celebrities, hiding the fact that some combinations are actively worse than chance.

**Action**: Note in Exp 4 README that Part C occasionally produces anti-correlated results, suggesting the injection mechanism can actively interfere with identity rather than simply failing to transfer it.

### 4. Exp 3 Part B Nearest Neighbours Reveal More Than Mode Collapse

The Part B CSV shows:
- Simu Liu: 6/10 faces (not 7 as reported in some docs)
- Constance Wu: 3/10 faces
- Ken Watanabe: 1/10 faces

All three are East Asian celebrities. This confirms the mode collapse hypothesis but also reveals the database has very few East Asian celebrities — the model generates Asian-presenting faces and the triangulation repeatedly falls back to the same 2-3 lookup points. This is a diversity problem in both the prompt design AND the celebrity database.

### 5. Exp 3 Nearest-Neighbour Similarity Predicts Triangulation Quality

From the Part A CSV, plotting nearest_sim against mean_arcface:

| Celebrity | Nearest Sim | Best Strategy Score |
|-----------|------------|-------------------|
| Dev Patel | 0.859 | 0.725 |
| Samuel L. Jackson | 0.655 | 0.473 |
| Idris Elba | 0.660 | 0.523 |
| Oscar Isaac | 0.633 | 0.508 |
| Denzel Washington | 0.542 | 0.449 |
| Will Smith | 0.452 | 0.233 |
| Matt Damon | 0.471 | 0.409 |
| Brad Pitt | 0.391 | 0.407 |
| Keanu Reeves | 0.265 | 0.296 |

There's a clear positive correlation: celebrities whose nearest neighbour is more similar get better triangulation scores. This confirms the earlier prediction that triangulation quality is driven by database density around the target, not by the interpolation method itself. A denser database would improve results across the board.

**Action**: Add this correlation to Exp 3 README. Consider computing Pearson/Spearman correlation coefficient in Phase 4.

---

## Documentation Quality Assessment

### What's Good

- **Rejected hypotheses doc** is excellent. Framing Exp 1 as "inconclusive, not rejected" is the right call. The distinction between rejecting a hypothesis and identifying a methodological issue is handled well.
- **Serendipity doc** correctly elevates the bimodal encoding as the headline finding.
- **Master log** timeline is clear and the "Resolved Questions" section gives a good status overview.
- **Experiment READMEs** follow the template consistently — hypotheses, predictions, methods, success criteria all present before results.
- **state_of_affairs.md** Section 5 (low-identity positions as safe manipulation targets) is well-placed — this finding tends to get overshadowed by the more dramatic results but is practically useful.

### What Needs Fixing

1. **Dev Patel outlier**: Currently listed as unexplained in serendipity.md. The raw data explains it completely (Riz Ahmed at 0.859 similarity). Update this.

2. **Simu Liu count inconsistency**: Exp 3 README says "7/10 random faces" but the CSV shows 6/10 with Simu Liu as nearest. Check which is correct.

3. **Exp 3 README has Dev Patel's scores wrong**: The README shows `Dev Patel: all_positions=0.725 (extreme outlier)` but the CSV shows all_positions=0.375, high_identity_only=0.725. The outlier is in high_identity_only, not all_positions. This is a significant error — it changes which strategy produced the outlier.

4. **No plots saved in the repository**. The experiment results directories have CSVs but no PNG files. The box plot shown during the review session exists somewhere but isn't committed. All plots should be saved to `phase3/experiment_*/results/plots/` for reproducibility.

5. **Missing Part A safety metric**: Exp 4 Part A used "similarity to baseline" but the success criterion was "CLIP score > 0.85". The results show 0.38-0.47 similarity — well below 0.85. The README doesn't explicitly flag this as a failure. If the metric is different from CLIP score, clarify. If it is CLIP score, Part A technically failed too.

6. **Exp 2 position analysis CSV**: Position 15 is labelled "scene" with discrimination ratio 1.459. There's only one scene token? With prompts like "portrait photo of [name], natural lighting, neutral background" you'd expect multiple scene-description tokens. Either the labelling heuristic is very narrow or there's a classification issue.

---

## Charting Guidance for Phase 3 visualisation.py

The existing `identity_analysis/plotting.py` was built for Phase 1-2 VAE latent analysis. Phase 3 needs different chart types. Here's what to implement, with rationale for each.

### Chart Types Needed

#### 1. Strip Plot with Violin Overlay (for Exp 4 Part B, any bimodal data)

**Why**: Box plots assume unimodal distributions. The Exp 4 injection data is bimodal — medians and quartiles don't represent any real data point. A strip plot shows every individual celebrity as a dot, the violin shows the distribution shape, and the bimodality becomes immediately visible.

**Use for**: Any comparison where n < 50 and the distribution might not be normal. This should be the DEFAULT for small-sample comparisons in Phase 3.

**Implementation**: `seaborn.violinplot()` with `inner=None`, overlaid with `seaborn.stripplot()` using `jitter=True, size=8`. Add horizontal threshold lines (dashed green) for success criteria. Colour individual points by celebrity or by group membership if the bimodal split is known.

#### 2. Paired Line Plot / Slope Chart (for Exp 3 strategy comparison)

**Why**: The current summary table (mean ± std per strategy) hides whether the strategy ranking is consistent per-celebrity. A slope chart draws one line per celebrity connecting their score across strategies. If all lines slope the same way, the ranking is robust. If lines cross, the ranking depends on which celebrity you're looking at.

**Use for**: Any within-subject comparison across conditions (strategies, layers, PCA dimensions).

**Implementation**: One line per celebrity, x-axis = strategy, y-axis = ArcFace score. Highlight the mean with a thick line. Use transparency (alpha=0.3) for individual celebrities to avoid clutter. Optionally colour by nearest-neighbour similarity to show whether database density predicts the pattern.

#### 3. Annotated Position Heatmap (for Exp 2 token position analysis)

**Why**: The Exp 2 results are fundamentally about *positions in a sequence* and their properties. A table of numbers is hard to scan. A heatmap with token text annotations makes the spatial structure immediately visible — you can see the causal attention gradient (zero signal before the name, signal in the name, signal accumulating in suffix) at a glance.

**Use for**: Any per-position analysis. This will be reused in Phase 4 for multi-layer position analysis.

**Implementation**: Heatmap with positions on x-axis, metrics (discrimination ratio, silhouette, between-var, within-var) on y-axis. Below the x-axis, annotate each position with the actual token text (e.g., `<|im_start|>`, `user`, `\n`, `portrait`, `photo`, `of`, `[NAME]`...). Colour by magnitude using a diverging colourmap (RdBu) for silhouette (centred on 0) and sequential (viridis) for discrimination ratio.

#### 4. Scatter with Marginal Distributions (for Exp 3 nearest-neighbour correlation)

**Why**: The correlation between nearest-neighbour similarity and triangulation score is a key finding that's currently only visible in the raw CSV. A scatter plot with regression line, Pearson r annotation, and marginal histograms makes this relationship immediately clear.

**Use for**: Any two-variable correlation analysis. Will be reused for "consistency score vs injection success" in Phase 4.

**Implementation**: `seaborn.jointplot()` with `kind="reg"`. Annotate with Pearson r and p-value. Label each point with the celebrity name (using `matplotlib.pyplot.annotate` with offset to avoid overlap, or `adjustText` library).

#### 5. Grouped Bar with Individual Points (for Exp 4 Part A safety test)

**Why**: Part A has only 3 conditions (noise, zeros, mean) with very few data points each. A bar chart alone hides the actual data. Overlay the individual measurements as dots on the bars.

**Use for**: Any small-n comparison where you want to show both the summary and the raw data.

#### 6. t-SNE / UMAP Scatter (for Exp 1 clustering, fix existing)

**Why**: A silhouette score of +0.071 doesn't tell you HOW clustering fails. A 2D projection shows whether identities overlap uniformly, form loose neighbourhoods, or have a few clean clusters drowning in noise.

**Fix needed**: The existing `plot_pca_scatter` uses `tab10` which only has 10 colours. With 99 celebrities, use a hash-based colour assignment or group by demographic/consistency tier and colour by group. Don't try to have 99 distinct legend entries — use a colourbar or group legend instead.

### General Principles

- **Always show individual data points when n < 50.** Summary statistics (mean, median, box) should supplement the raw data, not replace it.
- **Use seaborn on top of matplotlib** for statistical plots. It handles confidence intervals, distribution fitting, and paired comparisons much better.
- **Consistent colour palette**: Use a categorical palette (Set2 or tab10) for strategies, a sequential palette (viridis) for continuous variables like ArcFace scores, and a diverging palette (RdBu) for metrics centred on zero (silhouette).
- **Always include threshold lines** when success criteria exist. Dashed horizontal lines at 0.3, 0.4, etc. make pass/fail immediately visible.
- **Label outliers by name.** When a data point is > 2 std from the mean, annotate it with the celebrity name directly on the plot.
- **Save all plots to `results/plots/`** within each experiment directory. Every plot referenced in a README must exist as a file.

---

## Statistical Concerns

1. **n=20 for Exp 3 Part A strategy comparison**: With coefficient of variation ~30%, the 0.037 difference between high_identity_only (0.373) and all_positions (0.336) is unlikely to be significant. A paired Wilcoxon signed-rank test on the per-celebrity scores would confirm. The consistent direction is suggestive but not conclusive.

2. **n=10 for Exp 4 Part B bimodal claim**: The bimodal split is visually obvious but n=10 is very small to characterise two populations. With 5 in each group (roughly), you can't reliably estimate within-group variance. Phase 4 should test more celebrities to confirm the split holds at larger n.

3. **No confidence intervals reported anywhere.** The CSVs contain per-seed scores (5 seeds per celebrity), so bootstrap CIs are straightforward to compute. Every mean score should have a 95% CI attached.

4. **Multiple comparisons**: Three strategies tested on the same data in Exp 3 and 4. No correction applied. With only 3 comparisons this isn't severe, but worth noting.

---

## Phase 4 Recommendations (Ordered by Priority)

1. **Tokenisation analysis of bimodal split** — cheapest test, highest information value. Check exact Qwen3 tokenisation for all 10 Exp 4 celebrities. The n_name_tokens data already suggests a pattern (2-token names tend to work).

2. **Multi-layer sweep** (layers -2, -4, -6) on 99 celebrities for Exp 1. Include the 5-celebrity replication check at layer -6 to validate methodology.

3. **Layer -6 triangulation** — re-run Exp 3 Part A at layer -6 to see if it clears 0.4.

4. **Nearest-neighbour correlation analysis** — compute Pearson/Spearman between nearest_sim and triangulation score. This is already in the data, just needs the statistical test.

5. **Expand Exp 4 Part B to more celebrities** — the bimodal claim needs larger n.

6. **Fix Part B random face prompts** — use diverse demographic prompts to avoid mode collapse.
