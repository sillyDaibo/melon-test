# Reinforcement Learning

## Slides

- [强化学习引入](slides/1.强化学习引入.pdf)
- [蒙特卡洛与时序差分](slides/强化学习-合%281%29.pdf)
- [DQN](slides/DQN.pdf)
- [Deepseek-R1 & DAPO](slides/deepseek-r1.pdf)
- [过程奖励模型PRM](slides/20250521-PRM%281%29.pdf)
- [强化学习面临的挑战](slides/Taxonomy_of_RL_Algorithms.pdf)
- [DDPG](slides/Pre_4_9.pdf)
- [RLHF](slides/InstructGPT.pdf)
- [GRPO](slides/20250430 GRPO.pdf)

## Books

- [深度强化学习：基础、研究与应用](https://deepreinforcementlearningbook.org/assets/pdfs/%E6%B7%B1%E5%BA%A6%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%28%E4%B8%AD%E6%96%87%E7%89%88-%E5%BD%A9%E8%89%B2%E5%8E%8B%E7%BC%A9%29.pdf)
- [强化学习 (by Sutton)](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf)

## Papers

### Surveys

Recommended:

- [Towards Reasoning Era: A Survey of Long Chain-of-Thought for Reasoning Large Language Models](https://arxiv.org/pdf/2503.09567)
- [Reinforcement Learning Enhanced LLMs: A Survey](https://arxiv.org/pdf/2412.10400)

Optional:

- [A Survey of Mathematical Reasoning in the Era of Multimodal Large Language Model: Benchmark, Method & Challenges](https://arxiv.org/pdf/2412.11936)
- [A Comprehensive Survey of Scientific Large Language Models and Their Applications in Scientific Discovery](https://aclanthology.org/2024.emnlp-main.498.pdf)
- [A Comprehensive Survey of LLM Alignment Techniques: RLHF, RLAIF, PPO, DPO and More](https://arxiv.org/pdf/2407.16216)
- [A Survey of Reinforcement Learning from Human Feedback](https://arxiv.org/pdf/2312.14925)

### Basic RL Algorithms

Recommended:

- [Trust Region Policy Optimization](https://arxiv.org/pdf/1502.05477)(TRPO)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347)(PPO)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/pdf/2402.03300)(GRPO)

Optional:

- [Actor-Critic Algorithms](https://proceedings.neurips.cc/paper_files/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)
- [A2C(from Hugging Face)](https://huggingface.co/blog/deep-rl-a2c)
- [A2C is a special case of PPO](https://arxiv.org/pdf/2205.09123)
- [Asynchronous methods for deep reinforcement learning](https://arxiv.org/pdf/1602.01783)(A3C)
- [Continuous control with deep reinforcement learning](https://arxiv.org/pdf/1509.02971)(DDPG)
- [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/pdf/1801.01290)(SAC)
- [Mirror Descent Policy Optimization](https://arxiv.org/abs/2005.09814)(MDPO)
- [Revisiting Group Relative Policy Optimization: Insights into On-Policy and Off-Policy Training](https://arxiv.org/pdf/2505.22257v1)(Off-Policy GRPO)
- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/pdf/2503.14476)(tuned GRPO)
- [What is the Alignment Objective of GRPO?](https://arxiv.org/pdf/2502.18548)

### LLM Alignment Techniques

#### RLHF

- [InstructGPT：Training language models to follow instructions with human feedback](https://arxiv.org/pdf/2203.02155)
- [Safe RLHF: Safe Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2310.12773)

#### RLAIF

- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/pdf/2212.08073)
- [RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback](https://arxiv.org/pdf/2309.00267)
- [Generative Verifiers: Reward Modeling as Next-Token Prediction](https://arxiv.org/pdf/2408.15240)
- [Self-Rewarding Language Models](https://arxiv.org/pdf/2401.10020)

#### More About Reward Models For Alignment

- [Rule Based Rewards for Language Model Safety](https://arxiv.org/pdf/2411.01111)
- [Beyond Scalar Reward Model: Learning Generative Judge from Preference Data](https://arxiv.org/pdf/2410.03742)
- [Uncertainty-aware Reward Model: Teaching Reward Models to Know What is Unknown](https://arxiv.org/pdf/2410.00847)
- [Regularizing Hidden States Enables Learning Generalizable Reward Model for LLMs](https://arxiv.org/pdf/2406.10216)
- [Interpretable Preferences via Multi-Objective Reward Modeling and Mixture-of-Experts](https://arxiv.org/pdf/2406.12845)

#### DPO

- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/pdf/2305.18290)

Analysis:

- [Unpacking DPO and PPO: Disentangling Best Practices for Learning from Preference Feedback](https://arxiv.org/pdf/2406.09279)
- [Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study](https://arxiv.org/pdf/2404.10719)
- [Insights into Alignment: Evaluating DPO and its Variants Across Multiple Tasks](https://arxiv.org/pdf/2404.14723)
- [Iterative Preference Learning from Human Feedback: Bridging Theory and Practice for RLHF under KL-Constraint](https://arxiv.org/pdf/2312.11456)

### Long CoT Related

Read the [survey](https://arxiv.org/pdf/2503.09567) for a lot more information.

#### Aha Moment

- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/pdf/2501.12948)
- [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/pdf/2503.20783v1)
- [Understanding Aha Moments: from External Observations to Internal Mechanisms](https://arxiv.org/pdf/2504.02956)

#### Process Reward Model

- [Let's Verify Step by Step](https://arxiv.org/pdf/2305.20050)
- [Do We Need to Verify Step by Step? Rethinking Process Supervision from a Theoretical Perspective](https://arxiv.org/pdf/2502.10581)
- [Process Reinforcement Through Implicit Rewards](https://arxiv.org/pdf/2502.01456)

#### Test-Time Scaling

Analysis:

- [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/pdf/2408.03314v1)
- [ECM: A Unified Electronic Circuit Model for Explaining the Emergence of In-Context Learning and Chain-of-Thought in Large Language Model](https://arxiv.org/pdf/2502.03325)

Vertical Scaling:

- [s1: Simple test-time scaling](https://arxiv.org/pdf/2501.19393)

Parallel Scaling:

- [Self-Consistency Improves Chain Of Thought Reasoning In Language Models](https://arxiv.org/pdf/2203.11171)
- [Sample, Scrutinize and Scale: Effective Inference-Time Search by Scaling Verification](https://arxiv.org/pdf/2502.01839)
- [ExACT: Teaching AI Agents to Explore with Reflective-MCTS and Exploratory Learning](https://arxiv.org/pdf/2410.02052)

#### Reward-Model-Based RL Algorithms

- [ReMax: A Simple, Effective, and Efficient Reinforcement Learning Method for Aligning Large Language Models](https://arxiv.org/pdf/2310.10505)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/pdf/2402.03300)(GRPO)
- [Reinforce++: A simple and efficient approach for aligning large language models](https://arxiv.org/pdf/2501.03262v1)

#### Reward-Model-Free RL Algorithms (mostly about DPO, thus optional)

- [Offline Reinforcement Learning for LLM Multi-Step Reasoning](https://arxiv.org/pdf/2412.16145)
- [Improving Multi-Step Reasoning Abilities of Large Language Models with Direct Advantage Policy Optimization](https://arxiv.org/pdf/2412.18279)
- [Critical plan step learning boosts llm generalization in reasoning tasks](https://arxiv.org/pdf/2409.08642)
- [Critical Tokens Matter: Token-Level Contrastive Estimation Enhances LLM’s Reasoning Capability](https://arxiv.org/pdf/2411.19943)
- [Focused-DPO: Enhancing Code Generation Through Focused Preference Optimization on Error-Prone Points](https://arxiv.org/pdf/2502.11475)
- [Thinking Preference Optimization](https://arxiv.org/pdf/2502.13173)

### Appendix: Math Provers

- [Generative Language Modeling for Automated Theorem Proving](https://arxiv.org/pdf/2009.03393)(GPT-f)
- [HyperTree Proof Search for Neural Theorem Proving](https://arxiv.org/pdf/2205.11491)
- [DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for Reinforcement Learning and Monte-Carlo Tree Search](https://arxiv.org/pdf/2408.08152)
- [InternLM2.5-StepProver: Advancing Automated Theorem Proving via Expert Iteration on Large-Scale LEAN Problems](https://arxiv.org/pdf/2410.15700v1)
- [HunyuanProver: A Scalable Data Synthesis Framework and Guided Tree Search for Automated Theorem Proving](https://arxiv.org/abs/2412.20735)

2025

- [STP: Self-play LLM Theorem Provers with Iterative Conjecturing and Proving](https://arxiv.org/abs/2502.00212)
- [BFS-Prover: Scalable Best-First Tree Search for LLM-based Automatic Theorem Proving](https://arxiv.org/abs/2502.03438)
- [Goedel-Prover: A Frontier Model for Open-Source Automated Theorem Proving](https://arxiv.org/pdf/2502.07640)
- [Leanabell-Prover: Posttraining Scaling in Formal Reasoning](https://arxiv.org/pdf/2504.06122v2)
- [REAL-Prover: Retrieval Augmented Lean Prover for Mathematical Reasoning](https://arxiv.org/pdf/2505.20613)
- [Kimina-Prover Preview: Towards Large Formal Reasoning Models with Reinforcement Learning](https://arxiv.org/pdf/2504.11354v1)
- [DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning via Reinforcement Learning for Subgoal Decomposition](https://arxiv.org/pdf/2504.21801)

### Appendix: Other DPO related Methods or Improvements

- [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/pdf/2403.07691)
- [SLiC-HF: Sequence Likelihood Calibration with Human Feedback](https://arxiv.org/pdf/2305.10425)
- [β-DPO: Direct Preference Optimization with Dynamic β](https://arxiv.org/pdf/2407.08639)
- [sDPO: Don't Use Your Data All at Once](https://arxiv.org/pdf/2403.19270)
- [mDPO: Conditional Preference Optimization for Multimodal Large Language Models](https://arxiv.org/abs/2406.11839)
- [Statistical Rejection Sampling Improves Preference Optimization](https://arxiv.org/pdf/2309.06657)
- [Generalized Preference Optimization: A Unified Approach to Offline Alignment](https://arxiv.org/pdf/2402.05749)
- [Offline Regularised Reinforcement Learning for Large Language Models Alignment](https://arxiv.org/pdf/2405.19107)
- [Negating Negatives: Alignment with Human Negative Samples via Distributional Dispreference Optimization](https://arxiv.org/pdf/2403.03419)
- [Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning](https://arxiv.org/pdf/2404.05868)
- [Direct Nash Optimization: Teaching Language Models to Self-Improve with General Preferences](https://arxiv.org/pdf/2404.03715)
- [Self-Play Preference Optimization for Language Model Alignment](https://arxiv.org/pdf/2405.00675)
- [A Minimaximalist Approach to Reinforcement Learning from Human Feedback](https://arxiv.org/pdf/2401.04056)
- [Smaug: Fixing Failure Modes of Preference Optimisation with DPO-Positive](https://arxiv.org/pdf/2402.13228)
- [Token-level Direct Preference Optimization](https://arxiv.org/pdf/2404.11999)
- [A General Theoretical Paradigm to Understand Learning from Human Preferences](https://arxiv.org/pdf/2310.12036)

### Appendix：Some LLM technical reports with related RL algorithms

(Here RLHF and RLAIF utilize PPO by default)

2022

- [InstructGPT](https://arxiv.org/pdf/2203.02155): proposed RLHF

2023

| Model | Algorithm |
| --- | --- |
| [GPT4](https://arxiv.org/pdf/2303.08774) | RLHF + Rule-Based Reward Model(RBRM) |
| [Llama2](https://arxiv.org/pdf/2307.09288) | RLHF + Rejection Sampling |
| [qwen](https://arxiv.org/pdf/2309.16609) | RLHF |
| [zephyr-7B](https://arxiv.org/pdf/2310.16944) | direct distillation of LM Alignment |
| [starling-7B](https://openreview.net/pdf?id=GqDntYTTbk) | RLAIF |
| [Gemini](https://arxiv.org/pdf/2312.11805) | RLHF, continuously updating Reward Model |

2024

| Model | Algorithm |
| --- | --- |
| [DeepSeekMath](https://arxiv.org/pdf/2402.03300)| proposed GRPO |
| [Claude3](https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf)| RLAIF(see [Constitutional AI](https://arxiv.org/pdf/2212.08073)) |
| [InternLM2](https://arxiv.org/pdf/2403.17297)| COOL RLHF |
| [Reka](https://arxiv.org/pdf/2404.12387) | RLHF |
| [Llama3](https://arxiv.org/pdf/2407.21783)| rejection-sampling + DPO with some tricks |
| [Phi-3](https://arxiv.org/pdf/2404.14219)| DPO |
| [Zephyr 141B-A39B](https://huggingface.co/HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1)| [ORPO](https://arxiv.org/pdf/2403.07691) |
| [DeepSeek-V2](https://arxiv.org/pdf/2405.04434)| GRPO |
| [Qwen2](https://arxiv.org/pdf/2407.10671)| DPO |
| [Nemotron-4 340B](https://arxiv.org/pdf/2406.11704)| DPO + RPO |
| [ChatGLM](https://arxiv.org/pdf/2406.12793)| [ChatGLM-RLHF](https://arxiv.org/pdf/2404.00934) |
| [Hermes 3](https://arxiv.org/pdf/2408.11857)| DPO + LoRA |
| [Gemma2](https://arxiv.org/pdf/2408.00118)| RLHF |
| [Qwen2.5](https://arxiv.org/pdf/2412.15115)| DPO for offline and GRPO for online |
| [Hunyuan-Large](https://arxiv.org/pdf/2411.02265)| DPO |
| [Phi-4](https://arxiv.org/pdf/2412.08905)| DPO |
| [DeepSeek-V3](https://arxiv.org/pdf/2412.19437)| GRPO |

2025

| Model | Algorithm |
| --- | --- |
| [MiniMax-01](https://arxiv.org/pdf/2501.08313) | DPO for offline, modified GRPO for online |
| [Kimi-k1.5](https://arxiv.org/pdf/2501.12599v1)| [MDPO](https://arxiv.org/pdf/2005.09814) for CoT |
| [DeepSeek-R1](https://arxiv.org/pdf/2501.12948)| GRPO for CoT |
| [Qwen3](https://arxiv.org/pdf/2505.09388)| GRPO for CoT |
| [Phi-4-reasoning](https://arxiv.org/pdf/2504.21318)| GRPO for CoT |
