"""Original complete native transformer argument boundary; no projections."""
from dataclasses import dataclass

LORA_ATTN_CLASS = "MiniMaxH3Attention"
LORA_LINEAR_NAMES = ("to_q", "to_k", "to_v", "to_out.0")

@dataclass
class PackedLayout:
    hidden_states: object
    audio_hidden_states: object
    encoder_hidden_states: object
    timestep: object
    timestep_indices: object
    token_tags: object
    position_ids: object
    video_indices: object
    audio_indices: object
    text_indices: object
