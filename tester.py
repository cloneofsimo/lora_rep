from .predict import Predictor


p = Predictor()
p.setup()

out = p.predict(
    list_of_lora_urls="https://storage.googleapis.com/replicant-misc/lora/bfirsh-2.safetensors|https://storage.googleapis.com/replicant-misc/lora/lora_illust.safetensors",
    list_of_lora_scales="0.7|0.2",
    prompt="a photo of <0> in style of <1>",
)


# extract all default values from function Predictor.predict

from inspect import signature

sig = signature(Predictor.predict)

defaults = {}
for param_name, param in sig.parameters.items():
    if param.default != param.empty:
        defaults[param_name] = param.default

print(defaults)
