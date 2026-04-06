from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.phased_deltanet.configuration_phased_deltanet import PhasedDeltaNetConfig
from fla.models.phased_deltanet.modeling_phased_deltanet import PhasedDeltaNetForCausalLM, PhasedDeltaNetModel

AutoConfig.register(PhasedDeltaNetConfig.model_type, PhasedDeltaNetConfig, exist_ok=True)
AutoModel.register(PhasedDeltaNetConfig, PhasedDeltaNetModel, exist_ok=True)
AutoModelForCausalLM.register(PhasedDeltaNetConfig, PhasedDeltaNetForCausalLM, exist_ok=True)

__all__ = ['PhasedDeltaNetConfig', 'PhasedDeltaNetForCausalLM', 'PhasedDeltaNetModel']
