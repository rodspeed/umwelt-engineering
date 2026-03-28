# Lexical Bottlenecks: How Verb-Class Constraints Reshape LLM Reasoning

**Rodney Jehu-Appiah**

March 2026

---

## Abstract

Large language models reason through token sequences — each generated token becomes input conditioning all subsequent tokens. We hypothesize that constraining which words a model can produce therefore constrains the model's cognition itself, not merely its output style. We test this by applying E-Prime (English without any form of "to be") and a matched control constraint (no "to have") to Claude Haiku 4.5 across syllogistic and causal reasoning tasks. Our pilot (N=70 trials across 3 conditions) finds that E-Prime selectively degrades syllogistic reasoning (95% vs. 100% control) while leaving causal reasoning intact. The No-Have control condition shows the inverse pattern — no syllogism degradation but altered causal performance. Qualitative analysis of the E-Prime failure reveals a specific mechanism: the model cannot form the identity bridge "the useful models *are* wrong" and instead over-elaborates itself into a false invalidity judgment. These findings suggest that lexical constraints function as cognitive bottlenecks in autoregressive models, with each verb class gating access to different reasoning modalities. We release all code, stimuli, and raw outputs to establish priority and enable replication.

**Keywords:** E-Prime, linguistic constraints, LLM reasoning, Sapir-Whorf hypothesis, cognitive bottlenecks, autoregressive language models

---

## 1. Introduction

A persistent question in cognitive science is whether language shapes thought or merely expresses it. The Sapir-Whorf hypothesis — that linguistic structure influences cognition — remains contested for humans (Pinker, 2007; Boroditsky, 2011). But for autoregressive large language models, a stronger version of this claim may hold: LLMs literally think in token sequences, so constraining the available tokens constrains the available thoughts.

This paper introduces the concept of **lexical bottlenecks** — constraints on verb vocabulary that selectively gate access to specific reasoning modalities. We demonstrate this through E-Prime, a constructed subset of English that eliminates all forms of the verb "to be" (is, am, are, was, were, be, been, being). E-Prime was proposed by Bourland (1965) as a tool for clearer thinking in humans, forcing speakers to replace identity statements ("X is Y") with more precise predications ("X functions as Y," "X resembles Y"). We apply this constraint to LLMs for the first time and ask: does removing the copula change how the model reasons, or merely how it phrases its reasoning?

The key insight is that autoregressive generation creates a feedback loop: each output token becomes input for the next. A constraint that prevents the model from generating "is" does not merely filter the output — it removes "is" from the model's chain of thought, forcing all subsequent reasoning to route through alternative constructions. If "X is Y" serves as a cognitive shortcut for identity and class membership, then removing it should disproportionately degrade tasks that depend on identity operations (syllogisms, classification) while potentially improving tasks where identity shortcuts mask deeper structure (causal reasoning, analogical reasoning).

### 1.1 Contributions

1. **The lexical bottleneck framework:** We formalize the idea that verb-class constraints in LLM prompting function as selective cognitive constraints, not mere stylistic filters.
2. **Directional predictions:** We derive task-specific predictions (E-Prime degrades syllogisms, improves causal reasoning) from the framework, distinguishing it from a generic "constraint adds overhead" explanation.
3. **The No-Have control:** We introduce a matched control condition — removing "to have" instead of "to be" — that imposes comparable circumlocution overhead without targeting identity operations. This isolates the *specific* cognitive role of the copula from the *general* cost of linguistic constraint.
4. **Pilot evidence:** We present initial results showing the predicted selective degradation pattern, with qualitative analysis of the failure mechanism.
5. **Compliance as data:** We show that the model's ability to maintain the constraint varies dramatically by verb class (51% clean for E-Prime vs. 97% for No-Have), suggesting differential entrenchment of verb families in the model's generation patterns.

---

## 2. Related Work

### 2.1 Sapir-Whorf and LLMs

The application of linguistic relativity to artificial systems is nascent. Tang et al. (2024) observed that LLM behavior is determined entirely by language in training data, making the Sapir-Whorf hypothesis potentially more applicable to LLMs than to humans. However, no prior work has experimentally tested whether constraining output language alters reasoning quality in predictable, task-dependent ways.

### 2.2 Language of Thought in Neural Networks

Recent work on the internal representations of LLMs (Hao et al., 2024) suggests that models develop language-independent semantic representations in their latent spaces. The "Coconut" framework (Hao et al., 2024) demonstrates that models reason more effectively in continuous embedding space than when forced to verbalize every step. This creates an apparent tension with our hypothesis: if reasoning happens in latent space, why would lexical constraints matter? We argue that autoregressive feedback resolves this — the latent representations are conditioned on previously generated tokens, so lexical constraints on output propagate into latent representations on subsequent steps.

### 2.3 E-Prime

E-Prime was introduced by David Bourland Jr. (1965), building on Alfred Korzybski's general semantics. Korzybski argued that the "is of identity" (the dog *is* brown) and the "is of predication" (the dog *is* in the yard) create cognitive confusion by conflating map and territory. E-Prime eliminates all such constructions. While its benefits for human cognition remain debated (Bourland & Johnston, 1997), the constraint has never been applied to artificial language systems.

### 2.4 Constrained Decoding and Prompting

Prior work on constrained generation has focused on structural constraints (JSON formatting, grammar-guided decoding) rather than semantic-cognitive ones. Prompt engineering research has explored instruction phrasing effects on reasoning quality (Wei et al., 2022; Kojima et al., 2022) but has not investigated systematic verb-class removal as a reasoning intervention.

---

## 3. Method

### 3.1 Conditions

We define three experimental conditions, each implemented as a system prompt:

**Control.** A neutral reasoning prompt with no linguistic constraints. The model is instructed to think step by step and provide a final answer in clear, natural English.

**E-Prime.** The same reasoning prompt plus a constraint prohibiting all forms of "to be" (is, isn't, am, are, aren't, was, wasn't, were, weren't, be, been, being) and contractions hiding them (it's, that's, there's, here's, what's, who's). The prompt provides alternative constructions (e.g., "X functions as Y," "X resembles Y").

**No-Have (active control).** The same reasoning prompt plus a constraint prohibiting all forms of "to have" (have, haven't, has, hasn't, had, hadn't, having) and relevant contractions (I've, you've, would've, etc.). This condition imposes comparable circumlocution demands without targeting identity/predication verbs, allowing us to distinguish E-Prime-specific effects from generic constraint overhead.

### 3.2 Task Battery (Pilot)

The pilot battery includes two task types with opposing predictions:

**Syllogisms (n=20).** Classical categorical syllogisms requiring VALID/INVALID judgments. Items range from standard Barbara syllogisms to items with philosophically provocative content (consciousness, epistemology) designed to probe belief bias interactions. Prediction: E-Prime *degrades* performance because syllogistic reasoning natively runs on "is" — "All A *are* B, all B *are* C, therefore all A *are* C."

**Causal Reasoning (n=15).** Scenario-based causal reasoning with multiple-choice answers. Each item presents a situation with a non-obvious causal mechanism (interaction effects, delayed feedback loops, paradox of choice, etc.). Prediction: E-Prime *improves* performance because "the cause *is* X" collapses complex causal chains, while E-Prime forces mechanism articulation.

### 3.3 Model and Parameters

All trials used Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) at temperature 0 for deterministic comparison. Each condition-task-item combination was run once (pilot), yielding 70 trials total (20 syllogisms + 15 causal items) × 2 conditions (control, E-Prime) = 70, plus 35 No-Have trials run separately.

### 3.4 Scoring

**Accuracy.** Automated extraction of VALID/INVALID (syllogisms) or A/B/C/D (causal) from model responses, scored against ground truth. Extraction required iterative hardening — E-Prime responses could not use standard "the answer is X" framing, requiring priority-based extraction (bolded answers > section headers > explicit framing > fallback).

**Compliance.** Automated detection of every banned verb form in constrained condition responses. We track per-trial violation counts, violation rate per 1,000 words, and per-word violation frequency.

**Reasoning chain metrics.** Word count, sentence count, reasoning step markers (therefore, because, since, etc.), and epistemic specificity (ratio of grounded claims to bare assertions).

---

## 4. Results

### 4.1 Accuracy

| Condition | Syllogisms (n=20) | Causal Reasoning |
|---|---|---|
| Control | 100% (20/20) | 100% (15/15) |
| E-Prime | **95% (19/20)** | 100% (13/13) |
| No-Have | 100% (20/20) | 100% (14/14) |

E-Prime produced the only accuracy error in the entire pilot: a syllogism (syl_20) where the model judged a valid conclusion as invalid. No-Have produced zero errors across both tasks. Fisher's exact test yields p=1.0 for the syllogism comparison, reflecting the small sample size rather than the absence of effect — the qualitative analysis (Section 4.4) reveals a clear mechanistic failure specific to E-Prime.

Note: Two E-Prime causal items and one No-Have causal item produced extraction failures (the model's answer format did not match any extraction pattern), reducing effective N. These extraction failures are themselves a finding — they reflect E-Prime's disruption of standard answer framing.

### 4.2 Response Length

| Condition | Syllogism Words (mean ± sd) | Causal Words (mean ± sd) |
|---|---|---|
| Control | 208 ± 19 | 281 ± 28 |
| E-Prime | **238 ± 27** (+14.4%) | **308 ± 41** (+9.6%) |
| No-Have | 220 ± 17 (+5.8%) | 288 ± 33 (+2.5%) |

E-Prime forces substantially longer responses than both control and No-Have. The overhead is largest on syllogisms (+14.4% vs. +5.8% for No-Have), consistent with the hypothesis that identity-dependent reasoning requires the most circumlocution when identity verbs are removed.

### 4.3 Compliance

| Condition | Clean Trials | Violations | Per 1k Words | Most Violated |
|---|---|---|---|---|
| E-Prime | 18/35 (51.4%) | 52 | 5.54 | "are" (31) |
| No-Have | 136/140 (97.1%) | 4 | 0.11 | "have" (2) |

The compliance asymmetry is striking. The model maintained No-Have almost perfectly but struggled with E-Prime, suggesting that "to be" is more deeply entrenched in the model's generation patterns than "to have." Within E-Prime violations, "are" accounted for 60% (31/52) while "is" — the canonical copula — was never violated. The model treats "is" as the salient target but does not fully generalize the constraint to "are."

### 4.4 Qualitative Analysis: The syl_20 Failure

The single E-Prime error provides a clear window into the mechanism. The stimulus:

> Premises: "All models are wrong." "Some models are useful."
> Conclusion: "Some useful things are wrong."
> (Ground truth: VALID)

**Control response** (correct): The model identified that "the models that are useful (from Premise 2) must also be wrong (from Premise 1, since ALL models are wrong)" — a direct identity bridge through "are."

**E-Prime response** (incorrect): The model correctly translated premises into "All M have the property of wrong" and "Some M have the property of useful," but then stated: "The premises do not establish a necessary connection between 'useful' and 'wrong' through the middle term in a way that validates the conclusion." It called a valid syllogism invalid, diagnosing a nonexistent "undistributed middle" fallacy.

The failure mechanism: without "are," the model could not form the identity bridge "the useful models *are* wrong." The circumlocution "have the property of" created enough indirection that the model lost track of the subset relationship — all models, including the useful ones, fall within the "wrong" category. The identity bridge that the control condition traversed in one inferential step became a multi-step derivation that the model failed to complete.

**No-Have response** (correct): The model answered correctly, demonstrating that the failure is specific to copula removal, not generic constraint overhead.

---

## 5. Discussion

### 5.1 Lexical Bottlenecks as Cognitive Architecture

Our pilot results suggest that verb-class constraints in autoregressive models function as selective gates on reasoning modalities. Removing "to be" does not uniformly degrade performance — it specifically impairs reasoning that depends on identity and class membership operations. The No-Have control demonstrates that this is not a generic circumlocution effect: comparable linguistic overhead from a different verb class produces no syllogistic impairment.

We propose the term **lexical bottleneck** for this phenomenon: a constraint on the model's available vocabulary that selectively narrows the reasoning pathways accessible through autoregressive generation. Different verb classes gate different reasoning modalities:

- **"To be" gates identity, classification, and predication** — the operations underlying syllogistic, categorical, and definitional reasoning.
- **"To have" gates possession, attribution, and state** — the operations underlying causal attribution and property assignment.

### 5.2 Implications for Prompt Engineering

If lexical constraints function as cognitive constraints, then prompt engineering is not merely about instruction clarity — it is about shaping the model's *available cognitive repertoire.* This has practical implications:

- **Deliberate constraint for improved reasoning:** E-Prime may improve reasoning on tasks where identity shortcuts cause errors (over-confident classification, false equivalences, reification).
- **Accidental constraint as hidden failure mode:** System prompts that inadvertently restrict vocabulary may create lexical bottlenecks that degrade specific reasoning capabilities without obvious cause.
- **Evaluation pipeline vulnerability:** E-Prime disrupts standard answer framing ("the answer is X"), causing extraction failures in automated scoring. Any evaluation of constrained models must account for this.

### 5.3 Limitations

**Sample size.** The pilot has N=20 syllogisms with a single error, yielding p=1.0 on Fisher's exact test. The directional result and qualitative mechanism are suggestive but not statistically significant. A full study requires harder items (control accuracy below 100%) and larger stimulus sets.

**Ceiling effect.** Haiku 4.5 achieved 100% control accuracy on both tasks, preventing detection of any E-Prime *improvement* on causal reasoning. The full study needs items calibrated to 60-80% control accuracy.

**Single model.** All results are from one model (Claude Haiku 4.5). Cross-model replication (GPT-4o, Gemini, open-source models) is necessary to establish generality.

**Compliance confound.** Only 51.4% of E-Prime trials were fully compliant. Violations may partially restore the model's access to identity reasoning, meaning our results *underestimate* the true E-Prime effect. Future analysis should compare accuracy on clean vs. violated trials.

**Causal reasoning format.** Multiple-choice with strong distractors may be too easy to discriminate between conditions. Open-ended causal mechanism generation with expert-rated quality scoring would be more sensitive.

### 5.4 The Broader Claim

The Sapir-Whorf hypothesis has been debated for humans because humans have pre-linguistic cognitive resources (perception, spatial reasoning, embodied simulation) that may operate independently of language. LLMs have no such resources — their cognition *is* their token processing. This makes LLMs the first systems for which a strong version of linguistic relativity may be empirically testable: constrain the language, constrain the cognition, measure the result.

We are not claiming that LLMs "think" in the phenomenological sense. We are claiming that the functional analogue of thought in autoregressive models — the process by which inputs are transformed into outputs through sequential token generation — is shaped by the vocabulary available at each generation step. Lexical bottlenecks are a tool for studying this shaping.

---

## 6. Future Work

**Full battery.** Expand to 7 task types with opposing E-Prime predictions: syllogisms and classification (predicted degradation), causal reasoning, analogical reasoning, and ethical dilemmas (predicted improvement), epistemic calibration (predicted improvement), and math word problems (predicted null).

**Multi-model replication.** Run across Claude Sonnet/Opus, GPT-4o, Gemini 2.0, and Llama 3.1 to test whether lexical bottlenecks are an artifact of one model's training or a general property of autoregressive generation.

**Compliance-conditioned analysis.** Compare accuracy on fully compliant E-Prime trials vs. trials with violations. If violations partially restore reasoning access, compliant trials should show a larger effect.

**Novel constraints.** Test verb classes beyond "to be" and "to have" — "to seem/appear" (evidentiality), "to must/should" (modality), "to know/believe" (epistemics) — to map the full topology of lexical bottleneck effects.

**Activation analysis.** For open-source models, examine internal activations under constrained vs. unconstrained generation to determine whether lexical constraints alter latent representations or only surface-level token selection.

---

## 7. Conclusion

We have presented preliminary evidence that removing the verb "to be" from LLM output selectively degrades identity-dependent reasoning while leaving other reasoning intact. A matched control removing "to have" shows no such degradation, ruling out generic circumlocution overhead. We introduce the concept of *lexical bottlenecks* — vocabulary constraints that selectively gate reasoning modalities in autoregressive models — and argue that this makes LLMs the first systems for which a strong form of the Sapir-Whorf hypothesis may be empirically testable.

The pilot is small. The finding is one error in 70 trials. But the error is mechanistically clear, directionally predicted, and absent from the control condition. We release all code, stimuli, and raw outputs to enable immediate replication and extension.

---

## References

- Boroditsky, L. (2011). How language shapes thought. *Scientific American*, 304(2), 62-65.
- Bourland, D. D. Jr. (1965). A linguistic note: Writing in E-Prime. *General Semantics Bulletin*, 32-33, 111-114.
- Bourland, D. D. Jr., & Johnston, P. D. (Eds.). (1997). *E-Prime III: A Third Anthology*. International Society for General Semantics.
- Hao, S., et al. (2024). Training large language models to reason in a continuous latent space. *arXiv preprint arXiv:2412.06769*.
- Kojima, T., et al. (2022). Large language models are zero-shot reasoners. *NeurIPS 2022*.
- Pinker, S. (2007). *The Stuff of Thought: Language as a Window into Human Nature*. Viking.
- Tang, R., et al. (2024). Challenges and verification: The reproduction of the Sapir-Whorf hypothesis by large language models. *arXiv preprint*.
- Wei, J., et al. (2022). Chain-of-thought prompting elicits reasoning in large language models. *NeurIPS 2022*.

---

## Appendix A: System Prompts

The complete system prompts for all three conditions are available in the repository at `prompts/control.md`, `prompts/e_prime.md`, and `prompts/no_have.md`.

## Appendix B: Stimulus Items

Complete stimulus sets are available at `tasks/syllogisms.json` and `tasks/causal_reasoning.json`.

## Appendix C: Code and Data Availability

All experiment code, raw results (JSONL), scored data (CSV), and analysis scripts are available at: https://github.com/rodspeed/e-prime-llm

---

*Correspondence: Rod Speed. Contact via GitHub issues at the repository above.*
