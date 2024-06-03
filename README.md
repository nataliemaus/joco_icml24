# Joint Composite Latent Space Bayesian Optimization (JoCo)

Official implementation of Joint Composite Latent Space Bayesian Optimization (JoCo) (TODO: link to paper here...). This repository includes base code to run JoCo, along with full implementation for all optimization tasks and baselines needed to replicate all results in the original paper.

## Environment Setup
Use pip to install all dependencies needed to run JoCo on all tasks from the paper.

```Bash
pip install torch
pip install botorch
pip install gpytorch
pip install matplotlib
pip install pandas
pip install py-pde
pip install cma
```

### Additional dependencies for Stable Diffusion and LLM optimization tasks:

```Bash
pip install transformers
pip install sentencepiece
pip install accelerate
pip install diffusers
pip install torchvision
pip install protobuf
pip install einops
```

Additionally, for the Stable Diffusion Model Prompt optimization tasks (sportscar, dog, aircraft), download the data file at https://github.com/DebugML/adversarial_prompting/tree/master/data locally to joco/tasks/stable_diffusion_tasks/data.

## Running JoCo

Run the script run_joco.py to run JoCo and other baseline methods on each of the high-dimensional composite function optimization tasks from the paper. Run the following for a description of all optimization args:

```Bash
python3 run_joco.py -h
```

Note that the random seed will be set to 0 by default. To run with additional seeds specify a random seed using the --seed argument. All results in the paper average over runs using seeds 0 through 39.

## Tasks

The tasks set up in this repo can be found in the joco/tasks directory. Use the --task_id argument when running to provide the string id that identifies which optimization task you would like to run. The task id's are as follows:

1. langermann
Composite version of the Langermann function from the BO of Composite Functions paper (https://arxiv.org/abs/1906.01537).

2. rosenbrock
Composite version of the Rosenbrock function from the BO of Composite Functions paper (https://arxiv.org/abs/1906.01537).

3. rover
Mars Rover optimization from TuRBO paper (https://arxiv.org/pdf/1910.01739.pdf)

4. env
Environmental pollutants modeling funciton optimization task used in the BO of Composite Functions paper (https://arxiv.org/abs/1906.01537). Task was modified to be high-dimensional for the BO w/ High-Dimensional Outputs paper (https://arxiv.org/pdf/2106.12997.pdf), so this is the implementation we use to obtain a higher-dim version of this task.

5. pde
PDE optimization task from section 4.4 of the BO w/ High-Dimensional Outputs paper (https://arxiv.org/pdf/2106.12997.pdf).

6. sportscar
We seek to find prompts that cause a large stable diffusion model to generate images of sports cars without using any words related to sports cars.

7. dog
We seek to find prompts that cause a large stable diffusion model to generate images of dogs (and not mountains) without using any words related to dogs, and despite prepending the prompt to "a picture of mountain".

8. aircraft
We seek to find prompts that cause a large stable diffusion model to generate images of aircrafts (and not the ocean) without using any words related to aircrafts, and despite prepending the prompt to "a picture of the ocean".

9. falcon
We seek to find prompts that trick the Falcon LLM into generating toxic text.
