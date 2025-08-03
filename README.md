# AI Work

## Installation

    conda create -n aiwork-env python=3.11
    conda activate aiwork-env
    export PATH=/usr/local/Caskroom/miniconda/base/envs/aiwork-env/bin:/usr/local/Caskroom/miniconda/base/condabin:$PATH
    pip install -r requirements.txt

    # Dev setup
    pip install -r requirements-dev.txt
    pre-commit
    vim .pre-commit-config.yaml
    pre-commit install
