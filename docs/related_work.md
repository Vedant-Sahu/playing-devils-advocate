# Related Work and Project Goals

## Overview

This project explores whether a teacher-student LLM architecture can improve performance on graduate-level physics questions through iterative explanation and critique-based learning.

---

## 1. Related Work

### 1.1 Physics Reasoning Benchmarks

#### GPQA (Graduate-Level Google-Proof Q&A Benchmark)
- **Paper**: Rein et al. (2023) - [arXiv:2311.12022](https://arxiv.org/abs/2311.12022)
- **Description**: 448 multiple-choice questions in biology, physics, and chemistry written by domain experts
- **Difficulty**: PhD experts achieve 65% accuracy; non-experts with web access achieve 34%
- **Why we use it**: Graduate-level difficulty, "Google-proof" design prevents memorization, well-validated benchmark

#### Other Physics Benchmarks
- **SciBench**: College-level scientific problems requiring multi-step reasoning and calculus (Wang et al., 2024)
- **ABench-Physics**: Dynamic physics problems with parameter variation to test generalization (Xu et al., 2025)
- **MMLU**: Includes physics but limited to undergraduate level

**Gap**: Most benchmarks test one-shot performance rather than learning/improvement over time.

### 1.2 LLM Teaching and Tutoring Systems

#### Teacher-Student Architectures
- **Self-Consistency**: Multiple reasoning paths improve accuracy (Wang et al., 2022)
- **Constitutional AI**: AI provides feedback to improve AI responses (Bai et al., 2022)
- **Critique-based Learning**: Models learn from critiques of their own outputs

**Gap**: Limited work on iterative explanation-based improvement for domain-specific reasoning (physics).

### 1.3 Explanation Generation

#### Educational Explanations
- **Contrastive Explanations**: Explaining why correct answer is right AND why wrong answers are wrong
- **Misconception Identification**: Detecting and addressing common student errors
- **Socratic Questioning**: Leading students to understanding through questions

**Gap**: Most explanation work focuses on general Q&A, not graduate-level technical content.

### 1.4 LLM Capabilities in Physics

#### Recent Findings
- **Quantum Physics Calculations**: GPT-4 can perform graduate-level Hartree-Fock calculations with 87.5% accuracy (Pan et al., 2024)
- **GPQA Performance**: State-of-the-art models (GPT-4, Claude 3.5) achieve ~50-60% on physics questions
- **Reasoning vs Memorization**: Dynamic benchmarks reveal models rely heavily on pattern matching

**Gap**: Understanding whether explanatory teaching can help models genuinely reason vs memorize.

---

## 2. Research Questions

### Primary Question
**Can a teacher-student LLM architecture with iterative critique improve performance on graduate-level physics questions compared to baseline approaches?**

### Secondary Questions
1. Does explanation quality correlate with performance improvement?
2. Which types of physics problems benefit most from this approach?
3. Can we identify common misconceptions automatically from wrong answers?
4. How does this compare to few-shot prompting and chain-of-thought?

---

## 3. Our Approach

### 3.1 System Architecture

**Teacher Agent:**
- Generates detailed explanations for correct answers
- Identifies why wrong answers are tempting
- Addresses common misconceptions

**Student Agent (Devil's Advocate):**
- Critiques teacher explanations
- Identifies gaps or unclear reasoning
- Simulates different student understanding levels

**Critique Evaluator:**
- Assesses quality of critiques
- Determines if explanation needs refinement

**Iterative Loop:**
- Continue until critique quality drops below threshold
- Or maximum iterations reached

### 3.2 What Makes This Different

**Novel Contributions:**
1. Application to graduate-level physics specifically
2. Devil's advocate critique mechanism
3. Comparative evaluation methodology (LLM judges vs human)
4. Focus on misconception identification from multiple-choice distractors

**Compared to existing work:**
- More challenging domain than most tutoring systems
- Iterative refinement vs one-shot explanation
- Multiple evaluation approaches tested

---

## 4. Key References

### Primary Sources

1. **GPQA Benchmark**
   - Rein, D., et al. (2023). "GPQA: A Graduate-Level Google-Proof Q&A Benchmark"
   - [arXiv:2311.12022](https://arxiv.org/abs/2311.12022)

2. **LLMs on Physics**
   - Pan, H., et al. (2024). "Quantum Many-Body Physics Calculations with Large Language Models"
   - [arXiv:2403.03154](https://arxiv.org/abs/2403.03154)

3. **SciBench**
   - Wang, X., et al. (2024). "SciBench: Evaluating College-Level Scientific Problem-Solving"
   - [arXiv:2307.10635](https://arxiv.org/abs/2307.10635)

### Additional Reading

- Self-consistency in reasoning (Wang et al., 2022)
- Constitutional AI (Bai et al., 2022)
- Physics education research on misconceptions
- Explanation generation in AI systems

---
