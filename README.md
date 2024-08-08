<h1 align="center">Grokking Deep Learning</h1>

<p align="center">
    <img src="https://img.shields.io/badge/python-3.9.10-orange"
         alt="python version">
</p>

<p align="center">
    <a href=#motivation>Motivation</a> •
    <a href=#logs>Logs</a>
</p>

## Motivation

> [!NOTE]
>
> Inspired by [@moolmohino](https://www.youtube.com/@moolmohino), I have embarked on a similar journey, committing 1-2 hours daily over the next ? days to deepen my intuition in deep learning.

- Initially, I was motivated to go for a 100-day challenge. But life happens, so I am now going to continue in a disciplined manner instead.

In short, I grew tired of simply building Generative AI applications and managing LLM operations in my current role. The routine of calling an API without deeply understanding the process often frustrates me.

I started this repository for my personal growth as I [learn in public](https://www.swyx.io/learn-in-public). For context, I graduated with a Bachelor of Science in Business Analytics. Since my undergraduate days, I have known that I have a deep passion for solving problems in tech. However, solving bugs and building features never truly excited me. 

Instead, I have always been drawn to more math-related aspects of the field — despite having purposefully avoided math for the longest time due to some secondary school "trauma".

That being said, this repository will document my step-by-step journey of getting hands-on experience and building models from scratch. I'll be diving deep into the math from research papers, experiencing plenty of mind-blowing moments along the way, and likely talking to GPT more than I talk to my partner over the entire process.

## Logs

> [!WARNING]
>
> I intend to bombard this section with daily logs of my learning journey. If you're reading this from the future at the end of my challenge, be prepared to scroll through quite a bit of content.


- **Day 1, 07/08/24**: Exploring Vision Language Model [[PaliGemma]](https://huggingface.co/blog/paligemma).
    - Implemented (partial) Siglip Model [Contrastive Vision Encoder].
        - Siglip configurations
        - Siglip vision embeddings
    - Paper backlogs: 
        - [ ] [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
        - [ ] [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343)
        - [ ] [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
- **Day 2, 08/08/24**: Exploring Vision Language Model [[PaliGemma]](https://huggingface.co/blog/paligemma).
    - Implemented (completed) Siglip Model [Contrastive Vision Encoder].
        - Siglip Multi-headed Attention
        - Siglig Encoder and Layers
    - Implemented (partial) PaliGemma Model [Input Processor]
        - Image Processor
        - Text Processor
    - Paper backlogs:
        - [ ] [Layer Normalization](https://arxiv.org/pdf/1607.06450v1)