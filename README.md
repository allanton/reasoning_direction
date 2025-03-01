# Reasoning Direction

This repository contains code and notebooks related to identifying and analyzing the "reasoning direction" within Large Language Models (LLMs).

## Project Overview

The main goal of this research is to estimate the "reasoning direction" within LLM activation space. This work builds upon methodologies used to find other directional vectors in LLMs (such as the "refusal direction"), but with an important difference:

- Instead of using 1 LLM with different types of prompts, we use 2 models (original vs reasoning-tuned) with the same prompts (math problems)
- We collect activations from both models and calculate their difference to identify the reasoning direction
- We then test if adding this direction to models enhances their reasoning capabilities

## Repository Contents

- `reasoning_demo.ipynb`: Main notebook demonstrating the methodology
- `results/`: Directory containing outputs from reasoning and original models

## Methodology

The approach adapts techniques from orthogonal direction analysis to identify and manipulate the reasoning capability within LLMs.

For more detailed information, please refer to the project summary:
[Project Documentation](https://docs.google.com/document/d/1FDjSY00IC1xbJhkEBF8WeHjRqU_VQ07n2HwXwLVg3cs/edit?usp=sharing) 