---
name: research-paper-analysis
description: Use when reading, citing, deriving from or reproducing academic finance/econometrics/ML papers for ASTRA - checking a method's assumptions, deriving formulas, planning a reproduction, comparing to benchmarks, or writing a critical literature review and limitations section.
---

# Research Paper Analysis (ASTRA)

## Purpose
Ground ASTRA methods in real, verified literature and reproduce claims honestly.
Notes live in `docs/literature/`.

## Activate when
A method is taken from a paper; a citation is about to be written; a "reproduction" is claimed;
a literature review or limitations section is drafted.

## Workflow
1. **Verify the source exists** (DOI, SSRN/arXiv ID, journal, year) via a search or the paper itself.
   If it cannot be verified, do not cite it - write "unverified" or drop it. Never invent authors,
   titles, years or results.
2. **Extract**: research question, data (universe, period, frequency), method with equations,
   evaluation metric, headline result, stated limitations.
3. **Assumption audit**: which assumptions break on ASTRA data (free data, small universe, costs,
   different market)?
4. **Derivation**: re-derive key formulas in the note; flag discrepancies with the paper.
5. **Reproduction plan**: what exactly can be reproduced with available data; label outcome
   `reproduced | partially reproduced | not reproduced | not attempted`, with numbers side by side.
6. **Critical review**: publication bias, sample period dependence, transaction costs ignored,
   post-publication decay (McLean & Pontiff, 2016, Journal of Finance).

## Note template
`docs/literature/<first-author>-<year>-<slug>.md` with sections: Citation (verified link), Question,
Data, Method, Result, Assumptions vs ASTRA, Reproduction status, Critique.

## Failure modes
Hallucinated citations; claiming reproduction of a result on different data; copying long passages
(summarize, quote <15 words); treating in-sample paper results as expected live performance.

## Verification / acceptance
Every citation in docs has a verifiable identifier; reproduction claims link to experiment IDs.

## Limitations
Many papers rely on paid data (CRSP, TAQ, full-depth books) ASTRA cannot obtain; such results can at
best be partially reproduced on free proxies, and that must be stated next to every comparison.

## Interacts with
Supplies methods to all other skills; `quantitative-research` and `financial-ml-research` own the
experiments that test them.
