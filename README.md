# Umwelt Engineering: Designing the Cognitive Worlds of Linguistic Agents

**Rodney Jehu-Appiah**

> For a language model, the available language is not a transparent medium through which cognition passes — it *is* the cognition. Change the available language and you change the cognition itself.

**[Read the paper (PDF)](paper/Umwelt_Engineering.pdf)**

## What is Umwelt Engineering?

Jakob von Uexküll coined *Umwelt* to describe the perceptual world an organism's biology makes available to it. A tick's world contains butyric acid, temperature, and tactile density — nothing else exists for it.

A language model's cognition unfolds entirely in the token stream. The words don't describe the thinking; they *are* the thinking. Umwelt engineering is the deliberate design of this linguistic cognitive environment — a third layer in the agent design stack, above prompt engineering (what the agent is asked) and context engineering (what the agent knows).

## Key Findings

**Experiment 1** — 4,470 trials across three models (Claude Haiku 4.5, GPT-4o-mini, Gemini 2.5 Flash Lite), three conditions, and seven reasoning tasks:

- Removing "to be" (E-Prime) improves ethical reasoning by **+15.5pp** and causal reasoning by **+14.1pp**, while degrading syllogisms by **-3.4pp**
- Removing possessive "to have" (No-Have) improves ethical reasoning by **+19.1pp** and classification by **+6.5pp**, with near-perfect compliance (<1% violation rate vs. E-Prime's 52%)
- Effects are **model-dependent**: Gemini benefits enormously from E-Prime (+42pp on ethical reasoning), GPT-4o-mini collapses on epistemic calibration (-27.5pp), Haiku shows small effects. Cross-model correlations are negative — evidence that different models occupy different native Umwelten.

**Experiment 2** — 16 linguistically constrained agents on 17 debugging problems:

- No constrained agent outperforms the control individually
- A 3-agent ensemble selected for linguistic diversity achieves **100% ground-truth coverage** vs. 88.2% for the control
- The constraints make agents *different from each other* in useful ways, even when they don't make any single agent better

## Repository Structure

```
paper/                  # LaTeX source, bibliography, compiled PDF
tasks/                  # 130 task items across 7 reasoning categories
prompts/                # System prompts for control, E-Prime, No-Have conditions
scripts/                # Experiment runner, scoring, statistical analysis
results/                # Raw trial data (JSONL) and scored exports (CSV)
analysis/               # Statistical reports and visualizations
config/                 # Experiment configuration
```

## Reproducing the Experiments

```bash
pip install anthropic openai google-genai pyyaml

# Set API keys as environment variables
# ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY

# Run the full experiment (~5.5 hours, ~$15 in API costs)
python scripts/run_experiment.py

# Score results
python scripts/score_results.py

# Statistical analysis
python scripts/analyze_statistics.py
```

## Citation

Paper is forthcoming on ArXiv. In the meantime:

```
Jehu-Appiah, R. (2026). Umwelt Engineering: Designing the Cognitive Worlds
of Linguistic Agents. https://github.com/rodspeed/umwelt-engineering
```

## License

Code: MIT. Paper: All rights reserved.
