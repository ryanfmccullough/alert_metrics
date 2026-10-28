---
title: 'alert_metrics: event-level evaluation of continuous early-warning scores'
tags:
  - Python
  - clinical decision support
  - early warning
  - alarm burden
  - calibration
authors:
  - name: Ryan F. McCullough
    orcid: 0009-0005-4107-1408
    affiliation: 1
affiliations:
  - name: Independent researcher
    index: 1
date: 28 October 2026
bibliography: paper.bib
---

# Summary

Continuous risk scores for hospitalised patients are usually reported by their discrimination (AUROC) on time windows. What a unit experiences is different: a stream of alerts produced by a threshold, a refractory rule, and a warning window, with a burden in alerts per patient-day, a share of alerts that precede an event, and a warning time before each event. `alert_metrics` computes those event-level quantities from three plain tables (alerts, events, observation intervals) with textbook definitions, and produces operating-point tables that make the trade-off between burden and warning time explicit.

# Statement of need

Alarm fatigue is the main reason bedside early-warning systems are switched off, and warning time is the main reason they are switched on. Neither appears in a window-level AUROC, and the published scores that do report alert rates compute them in incompatible ways. A small, tested, dependency-light library that fixes the definitions (closed-left warning window, refractory suppression, burden normalised by observed time, calibration slope and intercept) lets benchmark papers and model cards report the same numbers. The package is used as the scoring library of a neonatal early-warning benchmark and is independent of any dataset.

# Acknowledgements

Definitions follow standard references [@degroot1983; @naeini2015; @brier1950; @murphy1973; @cox1958; @miller1991].

# References
