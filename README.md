

I recommend to run the installs in a fresh env in this order:

pip install jax[cuda]

pip install netket[cuda]

pip install transformers huggingface_hub

pip install torch torchvision torchaudio

since jax is the most destructive. You may remove the [cuda] if you don't have
an nvidia gpu.


see this hf repository for a dataset example:
lprimeau/MDR-Supercon-questions


the scripts are:
```
upload_download_model.py
```
upload model with pytorchmodelhubmixin

```
upload_download_model2.py
```
custom_net/
upload model downloadable by AutoModel, model definition uploaded to repo

```
foundational_nn.py
```
just instructions for running J1J2 foundation model nqs-models/j1j2_square_fnqs

```
transformers.ipynb
```
basic llm functionality

```
train_model_upload.py
```
train a model with netket and upload the msgpack model params to a repo
