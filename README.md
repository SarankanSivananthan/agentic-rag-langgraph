# Agentic RAG with LangGraph

An agent that decides *how* to answer a question — look it up in a Wikipedia knowledge base, compute something, chain two lookups, or admit the answer isn't there — built on top of the retrieval pipeline from [rag-wikipedia](https://github.com/SarankanSivananthan/rag-wikipedia), and evaluated the same way: measured, with intervals, against a plain fixed-pipeline baseline on the same questions.

> **Status: in progress.** This README is updated as each piece lands; nothing below is claimed until it is in the repo with its results.

## Goal

A tool-using agent is easy to demo and hard to evaluate. The question this project asks is the unglamorous one: **does the agent actually beat the fixed retrieve-then-generate pipeline, and where?** A plausible outcome is "no gain on simple lookups, a real gain on multi-step and arithmetic questions" — the evaluation is there to find out, not to confirm.

## Plan

- [ ] Retriever tool over the Wikipedia passages (same chunking and embeddings as rag-wikipedia)
- [ ] Calculator tool (arithmetic and date differences)
- [ ] Hand-built evaluation set, labelled with the tools each question needs: lookup, calculation, multi-step, unanswerable
- [ ] Fixed-pipeline baseline (plain RAG) scored on that set
- [ ] LangGraph agent: routing, retrieval, relevance check with query rewriting, step limit
- [ ] Evaluation: tool-choice accuracy, answer exact match / F1, correct abstention, steps, latency, cost — each with 95% intervals
- [ ] Results and honest limitations in this README

## Evaluation design

The evaluation set is small and written by hand, so its intervals will be wide; that is stated next to every number rather than hidden. Every question records which tools a correct solution uses, so tool choice can be scored separately from answer quality. Agent and baseline are scored on identical questions with the same metrics (SQuAD-style exact match and token F1, as in rag-wikipedia).
