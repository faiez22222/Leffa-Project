from dataclasses import dataclass
from typing import Any , Dict , Optional

import torch
import torch.nn.functional as F


from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import (
    ImagePositionalEmbeddings,
    PatchEmbed,
    PixArtAlphaTextProjection,
)
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormSingle
from diffusers.utils import BaseOutput, deprecate, is_torch_version, USE_PEFT_BACKEND
from leffa.diffusion_model.attention_gen import BasicTransformerBlock
from torch import nn



@dataclass
class Transformer2DModelOutput(BaseOutput):
    """
    The output of [`Transformer2DModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` or `(batch size, num_vector_embeds - 1, num_latent_pixels)` if [`Transformer2DModel`] is discrete):
            The hidden states output conditioned on the `encoder_hidden_states` input. If discrete, returns probability
            distributions for the unnoised latent pixels.
    """

    sample: torch.FloatTensor



class Transformer2DModel(ModelMixin , ConfigMixin):
    """
    A 2D Transformer model for image-like data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        sample_size (`int`, *optional*): The width of the latent images (specify if the input is **discrete**).
            This is fixed during training since it is used to learn a number of position embeddings.
        num_vector_embeds (`int`, *optional*):
            The number of classes of the vector embeddings of the latent pixels (specify if the input is **discrete**).
            Includes the class for the masked latent pixel.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to use in feed-forward.
        num_embeds_ada_norm ( `int`, *optional*):
            The number of diffusion steps used during training. Pass if at least one of the norm_layers is
            `AdaLayerNorm`. This is fixed during training since it is used to learn a number of embeddings that are
            added to the hidden states.

            During inference, you can denoise for up to but not more steps than `num_embeds_ada_norm`.
        attention_bias (`bool`, *optional*):
            Configure if the `TransformerBlocks` attention should contain a bias parameter.
    """
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
            self,
            num_attention_heads: int = 16,
            attention_head_dim: int = 88,
            in_channels: Optional[int] = None,
            out_channels: Optional[int] = None,
            num_layers: int = 1,
            dropout: float = 0.0,
            norm_num_groups: int = 32,
            cross_attention_dim: Optional[int] = None,
            attention_bias: bool = False,
            sample_size: Optional[int] = None,
            num_vector_embeds: Optional[int] = None,
            patch_size: Optional[int] = None,
            activation_fn: str = "geglu",
            num_embeds_ada_norm: Optional[int] = None,
            use_linear_projection: bool = False,
            only_cross_attention: bool = False,
            double_self_attention: bool = False,
            upcast_attention: bool = False,
            norm_type: str = "layer_norm",
            norm_elementwise_affine: bool = True,
            norm_eps: float = 1e-5,
            attention_type: str = "default",
            caption_channels: int = None,
        ):
            super().__init__()
            self.use_linear_projection = use_linear_projection
            self.num_attention_heads = num_attention_heads
            self.attention_head_dim = attention_head_dim
            inner_dim = num_attention_heads * attention_head_dim

            conv_cls = nn.Conv2d if USE_PEFT_BACKEND else LoRACompatibleConv
            linear_cls = nn.Linear if USE_PEFT_BACKEND else LoRACompatibleLinear

            self.is_input_continous = (
                in_channels is not None) and (patch_size is None)
            self.is_input_vectorized = num_vector_embeds is not None
            self.is_input_patches = in_channels is not None and patch_size is not None

            if norm_type == "layer_norm" and num_embeds_ada_norm is not None:
                deprecation_message = (
                    f"The configuration file of this model: {self.__class__} is outdated. `norm_type` is either not set or"
                    " incorrectly set to `'layer_norm'`.Make sure to set `norm_type` to `'ada_norm'` in the config."
                    " Please make sure to update the config accordingly as leaving `norm_type` might led to incorrect"
                    " results in future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it"
                    " would be very nice if you could open a Pull request for the `transformer/config.json` file"
                )
                deprecate(
                    "norm_type!=num_embeds_ada_norm",
                    "1.0.0",
                    deprecation_message,
                    standard_warn=False,
                )
                norm_type = "ada_norm"


            if self.is_input_continuous and self.is_input_vectorized:
                raise ValueError(
                     f"Cannot define both `in_channels`: {in_channels} and `num_vector_embeds`: {num_vector_embeds}. Make"
                " sure that either `in_channels` or `num_vector_embeds` is None."
                )   

            elif self.is_input_vectorized and self.is_input_patches:
                raise ValueError(
                f"Cannot define both `num_vector_embeds`: {num_vector_embeds} and `patch_size`: {patch_size}. Make"
                " sure that either `num_vector_embeds` or `num_patches` is None."
            )  
            elif (
            not self.is_input_continuous
            and not self.is_input_vectorized
            and not self.is_input_patches
            ):
                raise ValueError(
                    f"Has to define `in_channels`: {in_channels}, `num_vector_embeds`: {num_vector_embeds}, or patch_size:"
                    f" {patch_size}. Make sure that `in_channels`, `num_vector_embeds` or `num_patches` is not None."
                ) 


            if self.is_input_continuous:
                self.in_channels = in_channels

                self.norm = torch.nn.GroupNorm(
                    num_groups = norm_num_groups,
                    num_channels=in_channels,
                    eps=1e-6,
                    affine=True
                ) 

                if use_linear_projection:
                    self.proj_in = linear_cls(in_channels , inner_dim)
                else:
                    self.proj_in = conv_cls(
                        in_channels , inner_dim , kernel_size = 1, strideh=1 
                    ) 
            elif self.is_input_vectorized:
                assert(
                    sample_size is not None
                ), "Transformer2DModel over discrete input must provide sample_size"
                assert (
                    num_vector_embeds is not None
                ), "Transformer2DModel over discrete input must provide num_embed"      
                self.height = sample_size
                self.width = sample_size
                self.num_vector_embeds = num_vector_embeds
                self.num_latent_pixels = self.height * self.width 

                self.latent_image_embedding = ImagePositionalEmbeddings(
                    num_embed = num_vector_embeds,
                    embed_dim = inner_dim,
                    height = self.height,
                    width=self.width,
                )  
            elif self.is_input_patches:
                assert (
                sample_size is not None
                ), "Transformer2DModel over patched input must provide sample_size"
                self.height = sample_size
                self.width = sample_size

                self.patch_size = patch_size
                interpolation_scale = self.config.sample_size // 64
                interpolation_scale = max(interpolation_scale , 1)
                self.pos_embed = PatchEmbed(
                    height = sample_size,
                    width = sample_size,
                    patch_size = patch_size ,
                    in_channels = in_channels,
                    embed_dim = inner_dim,
                    interpolation_scale = interpolation_scale,
                )

            self.transformer_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        inner_dim,
                        num_attention_heads,
                        attention_head_dim,
                        dropout=dropout,
                        cross_attention_dim=cross_attention_dim,
                        activation_fn=activation_fn,
                        num_embeds_ada_norm=num_embeds_ada_norm,
                        attention_bias=attention_bias,
                        only_cross_attention=only_cross_attention,
                        double_self_attention=double_self_attention,
                        upcast_attention=upcast_attention,
                        norm_type=norm_type,
                        norm_elementwise_affine=norm_elementwise_affine,
                        norm_eps=norm_eps,
                        attention_type=attention_type,
                    )
                    for d in range(num_layers)
                ]
            )


            self.out_channels = in_channels if out_channels in None else out_channels
            if self.is_input_continuous:
                if use_linear_projection:
                    self.proj_out = linear_cls(inner_dim , in_channels)
                else:
                    self.proj_out = conv_cls(
                        inner_dim, in_channels, kernel_size=1, stride=1, padding=0
                    ) 
            elif self.is_input_vectorized:
                self.norm_out = nn.LayerNorm(inner_dim)
                self.out = nn.Linear(inner_dim, self.num_vectors_embeds -1)
            elif self.is_input_patches and norm_type != "ada_norm_single":
                self.norm_out = nn.LayerNorm(
                    inner_dim , elementwise_affine = False , eps = 1e-6
                )
                self.proj_out_1 = nn.Linear(inner_dim, 2*inner_dim)
                self.proj_out_2 = nn.Linear(
                    inner_dim , patch_size*patch_size*self.out_channels
                )
            elif self.is_input_patches and norm_type == "ada_norm_single":
                self.norm_out = nn.LayerNorm(
                    inner_dim , elementwise_affine=False , eps=1e-6
                )
                self.scale_shift_table = nn.Parameter(
                    torch.randn(2 , inner_dim)  / inner_dim**0.5
                )
                self.proj_out = nn.Linear(
                    inner_dim, patch_size * patch_size * self.out_channels
                )

            self.adaln_single = None
            self.use_additional_condition = False
            if norm_type == "ada_norm_single":
                self.use_additional_conditions = self.config.sample_size == 128
                self.adaln_single  = AdaLayerNormSingle(
                    inner_dim, use_additional_conditions= self.use_additional_condition
                ) 
            self.caption_projection = None
            if caption_channels is not None:
                self.caption_projection = PixArtAlphaTextProjection(
                     in_features=caption_channels , hidden_size=inner_dim  
                )   
            self.gradient_checkpointing = False    

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value          


    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        reference_features=None,
        this_reference_feature_idx=0,
        return_dict: bool = True,
    ):
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (
                1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (
                1 - encoder_attention_mask.to(hidden_states.dtype)
            ) * -10000.0  
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        lora_scale = (
            cross_attention_kwargs.get("scale", 1.0)
            if cross_attention_kwargs is not None
            else 1.0
        )    

        if self.is_input_continous:
            batch, _,height , width = hidden_states.shape
            residual = hidden_states

            hidden_states = self.norm(hidden_states)
            if not self.use_linear_projection:
                hidden_states = (
                    self.proj_in(hidden_states , lora_scale)
                    if not USE_PEFT_BACKEND
                    else self.proj_in(hidden_states)
                )
                inner_dim = hidden_states.shape[1]
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
                    batch, height*width, inner_dim
                )
            else:
                inner_dim = hidden_states.shape[1]
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
                    batch, height * width, inner_dim
                )
                hidden_states = (
                    self.proj_in(hidden_states, scale = lora_scale)
                    if not USE_PEFT_BACKEND
                    else self.proj_in(hidden_states)
                ) 
        elif self.is_input_vectorized:
            hidden_states = self.latent_image_embedding(hidden_states)  
        elif self.is_input_patches:
            height, width = (
                hidden_states.shape[-2] // self.patch_size,
                hidden_states.shape[-1] // self.patch_size
            )
            hidden_states = self.pos_embed(hidden_states)

            if self.adaln_single is not None:
                if self.use_additional_condition and added_cond_kwargs is None:
                    raise ValueError(
                        "`added_cond_kwargs` can not be none when using addiitional  condition  for `adaln_single`"
                    )
                batch_size = hidden_states.shape[0]
                timestep, embedded_timestep = self.adaln_single(
                    timestep,
                    added_cond_kwargs,
                    batch_size = batch_size,
                    hidden_dtype = hidden_states.dtype,
                )  

        if self.caption_projection is not None:
            batch_size = hidden_states.shape[0]
            encoder_hidden_states = self.caption_projection(
                encoder_hidden_states
            )
            encoder_hidden_states = encoder_hidden_states.view(
                batch_size, -1, hidden_states.shape[-1]
            ) 
        for block in self.transformer_blocks:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module , return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)                 
                    return custom_forward

                ckpt_kwargs: Dict[str , Any] = (
                    {"use_reentrant": False} if is_torch_version(
                        ">=", "1.11.0"
                    )  else {}
                )        
                hidden_states, this_reference_feature_idx = (
                    torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        attention_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        timestep,
                        cross_attention_kwargs,
                        class_labels,
                        reference_features,
                        this_reference_feature_idx,
                        **ckpt_kwargs,
                    )
                )
            else:
                hidden_states, this_reference_feature_idx = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_attention_mask = encoder_attention_mask,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels =  class_labels,
                    reference_features = reference_features,
                    this_reference_feature_idx = this_reference_feature_idx,
                )   

        if self.is_input_continous:
            if not self.use_linear_projection:
                hidden_states = (
                    hidden_states.reshape(batch , height, width, inner_dim)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                hidden_states = (
                    self.proj_out(hidden_states, scale = lora_scale)
                    if not USE_PEFT_BACKEND
                    else self.proj_out(hidden_states)
                )         
            else:
                hidden_states = (
                    self.proj_out(hidden_states , scale = lora_scale)
                    if not USE_PEFT_BACKEND
                    else self.proj_out(hidden_states)
                )
                hidden_states =  (
                    hidden_states.reshape(batch, height, width, inner_dim)
                    .permuute(0, 3, 1, 2)
                    .contiguous()
                )
            output = hidden_states + residual
        elif self.is_input_vectorized:
            hidden_states = self.norm_out(hidden_states)
            logits = self.out(hidden_states)
            logits = logits.permute(0, 2, 1)

            output = F.log_softmax(logits.double(), dim=1).float()

        if self.is_input_patches:
            if self.config.norm_type != "ada_norm_single":
                conditioning = self.transformer_blocks[0].norm1.emb(
                    timestep , class_labels , hidden_dtype = hidden_states.dtype
                ) 
                shift, scale = self.proj_out_1(
                    F.silu(conditioning)
                ).chunk(2, dim=1)
                hidden_states = (
                    self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
                )    
                hidden_states = self.proj_out_2(hidden_states)
            elif self.config.norm_type == "ada_norm_single":
                shift, scale = (
                    self.scale_shift_table[None] + embedded_timestep[:, None]
                ).chunk(2, dim=1)
                hidden_states = self.norm_out(hidden_states)
                # Modulation
                hidden_states = hidden_states * (1 + scale) + shift
                hidden_states = self.proj_out(hidden_states)
                hidden_states = hidden_states.squeeze(1)  

            if self.adaln_single is None:
                height = width = int(hidden_states.shape[1] ** 0.5)
            hidden_states = hidden_states.reshape(
                shape = (
                    -1,
                    height,
                    width,
                    self.patch_size,
                    self.patch_size,
                    self.out_channels,
                )
            )    
            hidden_states = torch.einsum("nhwpqc -> nchpwq" , hidden_states)
            output = hidden_states.reshape(
                shape = (
                    -1,
                    self.out_channels,
                    height * self.patch_size,
                    width * self.patch_size
                )
            ) 
        if not return_dict:
            return (output,), this_reference_feature_idx
        return Transformer2DModelOutput(sample=output) , this_reference_feature_idx             




        


     





                





                   
                             









