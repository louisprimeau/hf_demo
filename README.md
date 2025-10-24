

I recommend to run the installs in a fresh env in this order:

pip install jax[cuda]
pip install netket[cuda]
pip install transformers huggingface_hub
pip install torch torchvision torchaudio

since jax is the most destructive. You may remove the [cuda] if you don't have
an nvidia gpu. 