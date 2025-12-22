<div align="center">
<h1>V-DPM: 4D Video Reconstruction with Dynamic Point Maps</h1>

<a href="https://www.robots.ox.ac.uk/~vgg/research/vdpm/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<a href="https://huggingface.co/spaces/edgarsucar/vdpm"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a>

**[Visual Geometry Group, University of Oxford](https://www.robots.ox.ac.uk/~vgg/)**


[Edgar Sucar](https://edgarsucar.github.io/)\*, [Eldar Insafutdinov](https://eldar.insafutdinov.com/)\*, [Zihang Lai](https://scholar.google.com/citations?user=31eXgMYAAAAJ), [Andrea Vedaldi](https://www.robots.ox.ac.uk/~vedaldi/)
</div>

## Setup

First, clone the repository and setup a virtual environment with [uv](https://github.com/astral-sh/uv):

```bash
git clone git@github.com:eldar/vdpm.git
cd vdpm
uv venv --python 3.12
. .venv/bin/activate
uv pip install -r requirements.txt
```

## Viser demo
```bash
python visualise.py ++vis.input_video=examples/videos/camel.mp4
```

## Gradio demo
```bash
python gradio_demo.py
```
