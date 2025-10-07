import torch
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from einops import rearrange
from diffusers.models.attention import BasicTransformerBlock

# Based on https://github.com/fudan-generative-vision/hallo/blob/main/hallo/models/mutual_self_attention.py
class ReferenceAdapter:
    
    def __init__(
        self,
        unet,
        mode="write",
        cfg=False,
    ) -> None:
        """
       Initializes the ReferenceAdapter.

       Args:
           unet: The UNet model.
           mode: The mode of operation ("write" for Reference UNet, "read" for denoising UNet).
       """
        self.unet = unet
        assert mode in ["read", "write"]
        self.register_hooks(mode, cfg=cfg)

    def register_hooks(
        self,
        mode,
        cfg=False
    ):
        """
        Registers hooks for the model.
        """
        MODE = mode
        
        # Adapted from https://github.com/huggingface/diffusers/blob/v0.29.2/src/diffusers/models/attention.py#L407C5-L527C29
        def hacked_BasicTransformerBlock_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
            added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        ) -> torch.Tensor:
            if cross_attention_kwargs is not None:
                if cross_attention_kwargs.get("scale", None) is not None:
                    logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

            # Notice that normalization is always applied before the real computation in the following blocks.
            # 0. Self-Attention
            batch_size = hidden_states.shape[0]

            if self.norm_type == "ada_norm":
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.norm_type == "ada_norm_zero":
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = self.norm1(hidden_states)
            elif self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
            elif self.norm_type == "ada_norm_single":
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
                ).chunk(6, dim=1)
                norm_hidden_states = self.norm1(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
                norm_hidden_states = norm_hidden_states.squeeze(1)
            else:
                raise ValueError("Incorrect norm used")

            if self.pos_embed is not None:
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            # 1. Prepare GLIGEN inputs
            cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
            gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

            if MODE == "write":
                self.bank.append(norm_hidden_states.clone())
                attn_output = self.attn1(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )
            if MODE == "read":
                modified_norm_hidden_states = torch.cat(
                    [norm_hidden_states] + self.bank, dim=1
                )
                if cfg:
                    split_idx = norm_hidden_states.shape[0] // 2
                    attn_output_uc = self.attn1(
                        norm_hidden_states[:split_idx],
                        encoder_hidden_states=encoder_hidden_states[:split_idx] if self.only_cross_attention else None,
                        attention_mask=attention_mask,
                        **cross_attention_kwargs,
                    )
                    attn_output_c = self.attn1(
                        norm_hidden_states[split_idx:],
                        encoder_hidden_states=encoder_hidden_states[split_idx:] if self.only_cross_attention else modified_norm_hidden_states[split_idx:],
                        attention_mask=attention_mask,
                        **cross_attention_kwargs,
                    )
                    attn_output = torch.cat([attn_output_uc, attn_output_c], dim=0)
                else:
                    attn_output = self.attn1(
                        norm_hidden_states,
                        encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else modified_norm_hidden_states,
                        attention_mask=attention_mask,
                        **cross_attention_kwargs,
                    )

            if self.norm_type == "ada_norm_zero":
                attn_output = gate_msa.unsqueeze(1) * attn_output
            elif self.norm_type == "ada_norm_single":
                attn_output = gate_msa * attn_output

            hidden_states = attn_output + hidden_states
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)

            # 1.2 GLIGEN Control
            if gligen_kwargs is not None:
                hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

            # 3. Cross-Attention
            if self.attn2 is not None:
                if self.norm_type == "ada_norm":
                    norm_hidden_states = self.norm2(hidden_states, timestep)
                elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                    norm_hidden_states = self.norm2(hidden_states)
                elif self.norm_type == "ada_norm_single":
                    # For PixArt norm2 isn't applied here:
                    # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                    norm_hidden_states = hidden_states
                elif self.norm_type == "ada_norm_continuous":
                    norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
                else:
                    raise ValueError("Incorrect norm")

                if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                    norm_hidden_states = self.pos_embed(norm_hidden_states)

                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states

            # 4. Feed-forward
            # i2vgen doesn't have this norm 🤷‍♂️
            if self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
            elif not self.norm_type == "ada_norm_single":
                norm_hidden_states = self.norm3(hidden_states)

            if self.norm_type == "ada_norm_zero":
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            if self.norm_type == "ada_norm_single":
                norm_hidden_states = self.norm2(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
            else:
                ff_output = self.ff(norm_hidden_states)

            if self.norm_type == "ada_norm_zero":
                ff_output = gate_mlp.unsqueeze(1) * ff_output
            elif self.norm_type == "ada_norm_single":
                ff_output = gate_mlp * ff_output

            hidden_states = ff_output + hidden_states
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)

            return hidden_states
        
        attn_modules = []
        for block in self.unet.down_blocks:
            if hasattr(block, "attentions"):
                for attn in block.attentions:
                    for module in attn.transformer_blocks:
                        if isinstance(module, BasicTransformerBlock):
                            attn_modules.append(module)
        if hasattr(self.unet.mid_block, "attentions"):
            for attn in self.unet.mid_block.attentions:
                for module in attn.transformer_blocks:
                    if isinstance(module, BasicTransformerBlock):
                        attn_modules.append(module)
        for block in self.unet.up_blocks:
            if hasattr(block, "attentions"):
                for attn in block.attentions:
                    for module in attn.transformer_blocks:
                        if isinstance(module, BasicTransformerBlock):
                            attn_modules.append(module)

        for module in attn_modules:
            module.forward = hacked_BasicTransformerBlock_forward.__get__(module, BasicTransformerBlock)
            module.bank = []

    def update(self, writer):
        """
        Pass the reference attention bank from the writer model to this model.
        """

        reader_attn_modules = []
        for block in self.unet.down_blocks:
            if hasattr(block, "attentions"):
                for attn in block.attentions:
                    for module in attn.transformer_blocks:
                        if isinstance(module, BasicTransformerBlock):
                            reader_attn_modules.append(module)
        if hasattr(self.unet.mid_block, "attentions"):
            for attn in self.unet.mid_block.attentions:
                for module in attn.transformer_blocks:
                    if isinstance(module, BasicTransformerBlock):
                        reader_attn_modules.append(module)
        for block in self.unet.up_blocks:
            if hasattr(block, "attentions"):
                for attn in block.attentions:
                    for module in attn.transformer_blocks:
                        if isinstance(module, BasicTransformerBlock):
                            reader_attn_modules.append(module)

        writer_attn_modules = []
        for block in writer.unet.down_blocks:
            if hasattr(block, "attentions"):
                for attn in block.attentions:
                    for module in attn.transformer_blocks:
                        if isinstance(module, BasicTransformerBlock):
                            writer_attn_modules.append(module)
        if hasattr(writer.unet.mid_block, "attentions"):
            for attn in writer.unet.mid_block.attentions:
                for module in attn.transformer_blocks:
                    if isinstance(module, BasicTransformerBlock):
                        writer_attn_modules.append(module)
        for block in writer.unet.up_blocks:
            if hasattr(block, "attentions"):
                for attn in block.attentions:
                    for module in attn.transformer_blocks:
                        if isinstance(module, BasicTransformerBlock):
                            writer_attn_modules.append(module)

        assert len(reader_attn_modules) == len(writer_attn_modules)
        
        for r, w in zip(reader_attn_modules, writer_attn_modules):
            r.bank = [v.clone() for v in w.bank]


    def clear(self):
        """
        Clears the attention bank of all attention modules.
        """
        attn_modules = []
        for block in self.unet.down_blocks:
            if hasattr(block, "attentions"):
                for attn in block.attentions:
                    for module in attn.transformer_blocks:
                        if isinstance(module, BasicTransformerBlock):
                            attn_modules.append(module)
        if hasattr(self.unet.mid_block, "attentions"):
            for attn in self.unet.mid_block.attentions:
                for module in attn.transformer_blocks:
                    if isinstance(module, BasicTransformerBlock):
                        attn_modules.append(module)
        for block in self.unet.up_blocks:
            if hasattr(block, "attentions"):
                for attn in block.attentions:
                    for module in attn.transformer_blocks:
                        if isinstance(module, BasicTransformerBlock):
                            attn_modules.append(module)
        for r in attn_modules:
            r.bank.clear()