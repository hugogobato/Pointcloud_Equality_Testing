# Phase 0.7 — Citation Verification Log (VERIFIED_NOTES.md)

Date of verification: 2026-08-13.
Method: arXiv API (`export.arxiv.org/api/query?id_list=...`), arXiv abs pages, publisher
DOI records (Springer/JMLR/ProjectEuclid/Wiley/AIMS/ACM/IOP/Oxford/msp), OpenReview, and
journal pages. All 13 arXiv IDs were queried directly through the arXiv API; all
non-arXiv entries were checked against publisher or database records.

Verdict scale per reference:
- **found-and-confirmed**: title/authors/venue match the plan; no material discrepancy.
- **found-with-different-details**: exists, but one or more bibliographic details
  (year, venue, author list, title) differ from the plan.
- **NOT FOUND**: ID does not exist (none in this sweep).

Total entries resolved: **31** (30 found-and-confirmed or found-with-different-details;
0 NOT FOUND). All 31 have a URL or DOI; entries lacking a DOI are flagged in
`VERIFIED.bib` with `%` comments.

---

## 1. The three load-bearing claims (priority verification)

### 1.1 Saki & Faghihi, arXiv:2603.14169 — [B] — EXISTS AND CONFIRMED
- Checked: arXiv API `https://export.arxiv.org/api/query?id_list=2603.14169` (v2, submitted 2026-03-15).
- Record: **Amir Saki, Usef Faghihi**, "Beyond Means: Topological Causal Effects under
  Persistent-Homology Ignorability", arXiv:2603.14169v2, stat.ME. No journal ref, no DOI.
- Plan claim: *"a marginal persistence-diagram effect is NOT identified from conditional
  topological ignorability alone"* — abstract reads, **verbatim**: "a marginal
  persistence-diagram effect is not identified from conditional topological ignorability
  alone because persistent homology does not in general commute with mixtures over
  covariates."
- **Verdict: claim TRUE. The paper exists, and the negative identification claim is
  stated in the paper's own abstract.** The paper additionally proves identifiability up
  to an explicit error bound under *approximate* topological ignorability, which is
  relevant to P1 §1/C2's identification argument.

### 1.2 Faghihi, arXiv:2606.01184 — [B] — EXISTS AND CONFIRMED
- Checked: arXiv API (v2, submitted 2026-05-31).
- Record: **Usef Faghihi** (single author), "Topological Ignorability for Structural Causal
  Effects Beyond Means", arXiv:2606.01184v2, stat.ME. No journal ref, no DOI.
- Plan claim: *"topological ignorability = weak ignorability when the summary is
  injective"* — abstract reads, **verbatim**: "When the chosen summary is injective, this
  condition coincides with weak ignorability; for noninjective summaries, it can identify
  the structural feature of interest without identifying the full interventional law."
- **Verdict: claim TRUE.** Also relevant: the paper studies density-superlevel Betti
  summaries and Euler signatures (P1's Betti/Euler-statistic scope).

### 1.3 Han, Kim & Kim, arXiv:2607.20893 — [A] — EXISTS AND CONFIRMED
- Checked: arXiv API (v1, submitted 2026-07-23).
- Record: **Yeongung Han, Ilmun Kim, Jisu Kim**, "A Two-Sample Test on Weighted Persistence
  Intensity Functions in Topological Data Analysis", arXiv:2607.20893v1, stat.ME. No DOI.
- Plan claim: *"weighted persistence intensity functions, minimax optimal"* — abstract
  confirms a kernel-based permutation test on persistence intensity functions with a
  sharp variance bound and **"we establish minimax optimality of the proposed test"**
  (verbatim). Also derives an explicit PD of the Čech complex on the circle.
- **Verdict: claim TRUE.** The one-line description in the plan matches the abstract.

---

## 2. Per-reference verification log

### Foundations

| Reference | Checked at | Verdict | Discrepancies / notes |
|---|---|---|---|
| Kim & Lee, *Topological Causal Effects*, arXiv:2603.02289, ICLR 2026 [A] | arXiv API; OpenReview forum dYaos1ITw4; ICLR 2026 site | **found-and-confirmed** | Authors Kwangho Kim, Hajin Lee. arXiv journal_ref: "Proceedings of the Fourteenth International Conference on Learning Representations (ICLR 2026)". OpenReview: ICLR 2026 Poster. Abstract confirms TATE, doubly robust estimator, functional weak convergence, test of no topological effect. Lemma 2.1 / Thm 5.3 numbering not visible in abstract, but a published paper note confirms Thm 5.2 (weak convergence in $\ell^\infty(\mathcal T)$) and Thm 5.3 (stability bound $\|\phi-\phi'\|_\infty \le (1+2Lr\,c^{r-1})\,W_1(D,D')$). Local PDF `../CP_TATE/2603.02289v1.pdf` and code repo exist (kwangho-joshua-kim/top-causal-effect). |
| Souto & Diamantis, *A Mathematical Framework for TCDA*, arXiv:2607.28161 [A] | arXiv API (v1, 2026-07-30) | **found-and-confirmed** | Hugo Gobato Souto, Ioannis Diamantis. Abstract matches the four-layer architecture, outcome- vs distribution-level contrast, $g$-formula identification, DR representations, stability-transfer bounds. No DOI. |
| Chazal, Fasy, Lecci, Rinaldo, Wasserman, SoCG 2014 [A] | ACM DL; JoCG; HAL | **found-and-confirmed** | "Stochastic Convergence of Persistence Landscapes and Silhouettes", SoCG'14 pp. 474-483, doi:10.1145/2582112.2582128. Extended version JoCG 6(2):140-161 (2015), doi:10.20382/jocg.v6i2a8. |
| Bubenik, JMLR 2015 [A] | jmlr.org | **found-and-confirmed** | "Statistical Topological Data Analysis Using Persistence Landscapes", JMLR 16(3):77-102. JMLR has no journal DOI; ACL-anthology DOI 10.5555/2789272.2789275 resolves; URL to JMLR page given. SLLN + CLT claim confirmed from abstract. |
| Petersen & Müller, AoS 2019 [A] | Project Euclid | **found-and-confirmed** | Ann. Statist. 47(2):691-719, doi:10.1214/17-AOS1624. |
| Kennedy, DR-Learner 2023 [A] | Project Euclid | **found-and-confirmed** | "Towards optimal doubly robust estimation of heterogeneous causal effects", Electron. J. Statist. 17(2):3008-3049, doi:10.1214/23-EJS2157. arXiv:2004.14497. |

### Competitors

| Reference | Checked at | Verdict | Discrepancies / notes |
|---|---|---|---|
| Robinson & Turner, arXiv:1310.7467, JACT 2017 [A] | arXiv API; Springer | **found-and-confirmed** | "Hypothesis Testing for Topological Data Analysis", JACT 1(2):241-261 (2017), doi:10.1007/s41468-017-0008-7. |
| Kwitt, Huber, Niethammer, Lin, Bauer, NeurIPS 2015 [A] | proceedings.neurips.cc; dblp | **found-and-confirmed** | NeurIPS 28 (2015), pp. 3070-3078. NeurIPS 2015 issues no DOI; official proceedings URL given. |
| Han, Kim & Kim, arXiv:2607.20893 [A] | arXiv API | **found-and-confirmed** | See §1.3. No DOI. |
| Murris, Stolz & Borgwardt, arXiv:2606.11911 [A] | arXiv API (v1, 2026-06-10) | **found-with-different-details** | Authors and claim (survival framing, calibrated type I) confirmed. Title is "From Persistence to Survival: Hypothesis Testing, Effect Sizes and Vectorisation for Topological Features"; the shorthand "STRAND" does not appear in the arXiv title (presumably the method/software name). No DOI. |
| Moon & Lazar, arXiv:2006.05466, JRSS-C 2023 [A] | arXiv API; Oxford DOI record | **found-and-confirmed** | JRSS-C 72(3):628-648 (2023), doi:10.1093/jrsssc/qlad024. |
| Dubey & Müller, Biometrika 2019 [A] | Oxford DOI record; RePEc | **found-and-confirmed** | Biometrika 106(4):803-821, doi:10.1093/biomet/asz052. Fréchet-ANOVA/CLT-for-Fréchet-variance claim confirmed from abstract. |
| Krebs & Rademacher, arXiv:2401.10349 [A] | arXiv API | **found-and-confirmed (preprint-only)** | Exact title/authors confirmed; relevant-difference framing confirmed. **No journal version found as of 2026-08-13** — flagged in .bib. |
| Cericola et al., Involve 2018 [A] | msp.org DOI record | **found-and-confirmed** | "Extending hypothesis testing with persistent homology to three or more groups", Involve 11(1):27-51, doi:10.2140/involve.2018.11.27. Authors: Cericola, Johnson, Kiers, Krock, Purdy, Torrence. |
| Berry, Chen, Cisewski-Kehe, Fasy, JACT 2020 [A] | arXiv API; Springer/NSF-PAR | **found-and-confirmed** | JACT 4(2):211-262, doi:10.1007/s41468-020-00048-w. |
| Vejdemo-Johansson & Mukherjee, arXiv:1812.06491 [B] | arXiv API; AIMS | **found-with-different-details** | arXiv title "Multiple testing with persistent homology" (2018); published version "Multiple hypothesis testing with persistent homology", FoDS 4(4):667-705 (2022), doi:10.3934/fods.2022018. Use the DOI record. |
| Kumar & Dhar, arXiv:2211.13959 [B] | arXiv API | **found-and-confirmed (preprint-only)** | "Testing Homological Equivalence Using Betti Numbers" (v3, 2023-12-01). Authors' site lists a journal version "to appear" in SIAM Theory of Probability and Its Applications (2025+), not yet indexed — flagged in .bib. |
| Nakayama, arXiv:2511.00938 [C] | arXiv API (v2) | **found-and-confirmed** | "Persistence-Based Statistics for Detecting Structural Changes in High-Dimensional Point Clouds", single author Toshiyuki Nakayama. No DOI. The [C] risk ("must be located") is retired. |
| Islambekov et al., FoDS 2023 [B] | AIMS; arXiv:2306.06257 | **found-with-different-details** | Actual: Islambekov & Pathirana (two authors), "Vector summaries of persistence diagrams for permutation-based hypothesis testing", FoDS 6(1):41-61, **2024** (received Jul 2023, online 2024-02-06), doi:10.3934/fods.2024002. The plan's "FoDS 2023" is the submission year. |

### Geometry hazards

| Reference | Checked at | Verdict | Discrepancies / notes |
|---|---|---|---|
| Mileyko, Mukherjee, Harer 2011 [A] | IOP | **found-and-confirmed** | Inverse Problems 27(12):124007, doi:10.1088/0266-5611/27/12/124007. Completeness+separability of $(\mathcal D_p, W_p)$ claim confirmed. |
| Turner, Mileyko, Mukherjee, Harer 2014 [A] | Springer | **found-and-confirmed** | DCG 52(1):44-70, doi:10.1007/s00454-014-9604-7. Non-uniqueness of Fréchet means context confirmed (algorithm finds *local* minima). |
| Che, Galaz-García, Guijarro, Membrillo Solis, arXiv:2109.14697 [A] | Springer; arXiv | **found-with-different-details** | Published: JACT 8(8):2197-2246 (**2024**), doi:10.1007/s41468-024-00189-2, open access. The plan cited only the arXiv preprint. Claim confirmed: $(\mathcal D_2, W_2)$ is a geodesic Alexandrov space of nonnegative curvature; the article proves infinite covering/Hausdorff/asymptotic/Assouad dimensions (plan's "infinite-dimensional in every standard sense" is consistent). |
| Roycraft, Krebs, Polonik, AoS 2023 [A] | Project Euclid | **found-and-confirmed** | Ann. Statist. 51(4):1484-1509, doi:10.1214/23-AOS2277. Smoothed-bootstrap consistency and naive-bootstrap failure for persistent Betti numbers confirmed from abstract. |
| Divol & Lacombe, JACT 2021 [A] | Springer; arXiv | **found-and-confirmed** | JACT 5(1):1-53, doi:10.1007/s41468-020-00061-z. Persistence measures via optimal partial transport confirmed. |
| Divol & Chazal, JoCG 2021 [A] | jocg.org | **found-with-different-details** | Actual: JoCG **10(2):127-153 (2020)**, doi:10.20382/jocg.v10i2a7 (short version SoCG 2018, LIPIcs 99:26, doi:10.4230/LIPIcs.SoCG.2018.26). The plan's "JoCG 2021" is off by a year. |
| Blumberg, Gal, Mandell, Pancia, FoCM 2014 [A] | Springer | **found-with-different-details** | Published title "Robust statistics, hypothesis testing, and confidence intervals for persistent homology on metric measure spaces" — this is the subsampling-for-PH paper the plan means; the arXiv ID is 1206.4581 (not listed in the plan). FoCM 14(4):745-789, doi:10.1007/s10208-014-9201-4. |

### Ecology (deferred track, Phases 11-12 incumbents)

| Reference | Checked at | Verdict | Discrepancies / notes |
|---|---|---|---|
| Warren, Glor & Turelli, niche-equivalency permutation test [B] | Wiley DOI record | **found-and-confirmed** | "Environmental Niche Equivalency versus Conservatism: Quantitative Approaches to Niche Evolution", Evolution 62(11):2868-2883 (2008), doi:10.1111/j.1558-5646.2008.00482.x. The permutation test with Schoener D and Warren I is the paper's randomization procedure. |
| Broennimann et al., PCA-env overlap (Schoener D, Hellinger I) [B] | Wiley DOI record | **found-and-confirmed** | "Measuring ecological niche overlap from occurrence and spatial environmental data", Global Ecology and Biogeography 21(4):481-497 (2012), doi:10.1111/j.1466-8238.2011.00698.x. 12 authors (Broennimann, Fitzpatrick, Pearman, Petitpierre, Pellissier, Yoccoz, Thuiller, Fortin, Randin, Zimmermann, Graham, Guisan). Kernel-smoother PCA-env framework confirmed. |
| Blonder et al., hypervolume overlap [B] | Wiley DOI record | **found-and-confirmed** | "The n-dimensional hypervolume", Global Ecology and Biogeography 23(5):595-609 (2014), doi:10.1111/geb.12146. KDE-based hypervolume overlap (R package `hypervolume`) confirmed. |

---

## 3. Summary

- **Total entries in VERIFIED.bib: 31** (Foundations 8, Competitors 13, Geometry hazards 7, Ecology 3).
- **NOT FOUND: 0.** Every reference in §2 (including all 2026 IDs) exists. This includes
  all three load-bearing [B] papers, which were the main risk in the sweep.
- **found-with-different-details (5):** Murris et al. (title/STRAND shorthand),
  Islambekov & Pathirana (year 2024, two authors), Divol & Chazal (JoCG 2020 not 2021),
  Che et al. (published JACT 2024, plan cited preprint only), Blumberg et al. (published
  title; arXiv ID 1206.4581). None change the substance of the plan.
- **Preprint-only (no DOI) as of verification (10):** kim_lee_tce (ICLR, no DOI),
  souto_diamantis, saki_faghihi, faghihi, han_kim_kim, murris, krebs_rademacher,
  kumar_dhar, nakayama, kwitt (NeurIPS, no DOI). All flagged with `%` comments in the .bib.
  Kumar & Dhar has a "to appear" journal announcement (SIAM TVP); Krebs & Rademacher and
  Nakayama have none.
- **Action for the plan:** §2's [C] entry (Nakayama) is resolved; the three [B] entries
  (Saki & Faghihi, Faghihi, Vejdemo-Johansson & Mukherjee) are now confirmed, so the
  confidence grades can be upgraded to [A] for the two identification papers whose claims
  were verified verbatim (Saki & Faghihi, Faghihi) and for the ecology trio.
