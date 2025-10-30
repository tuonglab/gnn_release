# Utilising Machine Learning to classify cancer using Paedatric Immune Repertoire

## Installation

*NOTE: At the moment, since development is pinneed on torch==2.3.1, we will use it for now. I will explore updating torch to build agnostic version in the future

To install torch==2.3.1, follow this step:

- OSX

`pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1`

- Linux and Windows

```
# CUDA 11.8
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
# CPU only
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu
```

For non-developer

`pip install tcrgnn`

For developer

`pip install pdm`
`pdm install -E`
`pytest`
