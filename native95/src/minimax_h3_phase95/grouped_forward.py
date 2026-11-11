"""Pinned native forward with ONLY the main-block dispatch loop replaced.

The exact original input packing, BF16/FP32 casts, full attention, native
refiner, final heads and return contract are preserved. This is a new scoped
execution callable, not a change to the installed Diffusers source file.
"""
from typing import Any
import torch
from diffusers.utils import apply_lora_scale
from diffusers.models.modeling_utils import get_parameter_dtype
from diffusers.models.transformers.transformer_minimax_h3 import (
    MiniMaxH3TransformerOutput, MINIMAX_H3_MODALITY_NUM,
)

@apply_lora_scale("attention_kwargs")
def grouped_native_forward(
    self,
    hidden_states: torch.Tensor,
    audio_hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    timestep_indices: torch.Tensor,
    token_tags: torch.Tensor,
    position_ids: torch.Tensor,
    video_indices: torch.Tensor,
    audio_indices: torch.Tensor,
    text_indices: torch.Tensor,
    attention_kwargs: dict[str, Any] | None = None,
    return_dict: bool = True,
) -> MiniMaxH3TransformerOutput | tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        hidden_states (`torch.Tensor` of shape `(batch_size, num_video_tokens, in_channels * prod(patch_size))`):
            Patchified video latent rows — conditioning rows and target rows — ordered as they appear in the packed
            sequence, i.e. matching `video_indices`.
        audio_hidden_states (`torch.Tensor` of shape `(batch_size, num_audio_tokens, audio_in_channels)`):
            Audio latent rows, ordered to match `audio_indices`.
        encoder_hidden_states (`torch.Tensor` of shape `(batch_size, num_text_tokens, text_dim)`):
            Text conditioning, ordered to match `text_indices`.
        timestep (`torch.Tensor` of shape `(num_timesteps,)`):
            The *distinct* timestep values present in the packed sequence, in `[0, 1]` and unscaled. One forward
            serves rows at different noise levels (target video, target audio, conditioning rows).
        timestep_indices (`torch.Tensor` of shape `(seq_len,)`):
            For every row of the packed sequence, the index of its timestep in `timestep`.
        token_tags (`torch.Tensor` of shape `(seq_len,)`):
            For every row of the packed sequence, its modality: `0` video, `1` text, `2` audio.
        position_ids (`torch.Tensor` of shape `(seq_len, 3)`):
            The `(t, h, w)` rotary coordinates of every row of the packed sequence.
        video_indices (`torch.Tensor` of shape `(num_video_tokens,)`):
            Positions of the video rows in the packed sequence.
        audio_indices (`torch.Tensor` of shape `(num_audio_tokens,)`):
            Positions of the audio rows in the packed sequence.
        text_indices (`torch.Tensor` of shape `(num_text_tokens,)`):
            Positions of the text rows in the packed sequence.
        attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that, if specified, may carry a `scale` entry which is applied to the LoRA layers.
        return_dict (`bool`, defaults to `True`):
            Whether to return a [`MiniMaxH3TransformerOutput`] instead of a plain tuple.

    Returns:
        [`MiniMaxH3TransformerOutput`] or `tuple`:
            The video velocity of shape `(batch_size, num_video_tokens, in_channels * prod(patch_size))` and the
            audio velocity of shape `(batch_size, num_audio_tokens, audio_in_channels)`, in the row order of
            `video_indices` and `audio_indices`.
    """
    # `attention_kwargs` is consumed by the `@apply_lora_scale` decorator on this method.
    if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
        raise ValueError(f"`position_ids` must be a `(seq_len, 3)` tensor, got {list(position_ids.shape)}.")
    sequence_length = position_ids.shape[0]
    if token_tags.shape != (sequence_length,) or timestep_indices.shape != (sequence_length,):
        raise ValueError(
            "`token_tags` and `timestep_indices` must both be `(seq_len,)` tensors matching `position_ids`, got "
            f"{list(token_tags.shape)} and {list(timestep_indices.shape)} for seq_len={sequence_length}."
        )

    rotary_emb = self.rope(position_ids)

    # 1. Project each modality and scatter the rows into the packed sequence buffer. The checkpoint is
    # mixed-precision (the two patch projections are float32 while `context_embedder` and the block stack are
    # bfloat16 — see `_keep_in_fp32_modules`), so every input is aligned with its projection's parameter dtype,
    # mirroring the reference's explicit casts. The text stream sets the dtype of the packed sequence.
    video_embeds = self.proj_in(hidden_states.to(get_parameter_dtype(self.proj_in)))
    audio_embeds = self.audio_proj_in(audio_hidden_states.to(get_parameter_dtype(self.audio_proj_in)))
    text_embeds = self.context_embedder(encoder_hidden_states.to(get_parameter_dtype(self.context_embedder)))
    text_embeds = self.token_refiner(text_embeds)

    hidden_states = text_embeds.new_zeros((text_embeds.shape[0], sequence_length, text_embeds.shape[-1]))
    hidden_states = hidden_states.index_copy(1, text_indices, text_embeds)
    hidden_states = hidden_states.index_copy(1, video_indices, video_embeds.to(text_embeds.dtype))
    hidden_states = hidden_states.index_copy(1, audio_indices, audio_embeds.to(text_embeds.dtype))

    # 2. One timestep embedding per distinct noise level. `temb` is shared by all AdaLN projections, which are
    # bfloat16 in the checkpoint while `time_embedder` is float32, so it stays at the time embedder's precision:
    # each AdaLN module applies its own activation to it and casts to its projection's dtype afterwards.
    temb = self.time_proj(timestep)
    temb = self.time_embedder(temb.to(get_parameter_dtype(self.time_embedder)))

    # 3. Row -> AdaLN table row.
    adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags

    hidden_states = self._h3_grouped_dispatch(
        hidden_states, temb, adaln_indices, rotary_emb, None
    )

    # 5. Both heads run over every row, then the rows of each modality are selected. The heads are listed in
    # `_keep_in_fp32_modules`, so they stay float32 while the block stack runs in the requested `torch_dtype`;
    # align the activation with their parameter dtype.
    hidden_states = self.norm_out(hidden_states, temb, timestep_indices).to(get_parameter_dtype(self.proj_out))
    video_output = self.proj_out(hidden_states).index_select(1, video_indices)
    audio_output = self.audio_proj_out(hidden_states).index_select(1, audio_indices)

    if not return_dict:
        return (video_output, audio_output)
    return MiniMaxH3TransformerOutput(sample=video_output, audio_sample=audio_output)
