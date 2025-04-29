from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from diffusers.models.activations import ApproximateGELU, GEGLU, GELU
from diffusers.models.attention_processor import Attention 
from diffusers.models.embeddings import SinusoidalPositionalEmbedding
from diffusers.models.lora import LoRACompatibleLinear
from diffusers.models.normalization import (
    AdaLayerNorm,
    AdaLayerNormContinuous,
    AdaLayerNormZero,
    RMSNorm
)
from diffusers.utils import USE_PEFT_BACKEND
from diffusers.utils.torch_utils import maybe_allow_in_graph
from torch import nn 


def _chunked_feed_forward(
        ff: nn.Module,
        hidden_states: torch.tensor,
        chunk_dim: int,
        chunk_size: int,
        lora_scale: Optional[float] = None,
):
    if hidden_states.shape[chunk_dim] % chunk_size !=0:
        raise ValueError(
            f"`hidden_states` dimension to be chunked: {hidden_states.shape[chunk_dim]} has to be divisible by chunk size: {chunk_size}. Make sure to set an approriate `chunk_size` when calling `unet.enable_forward_chunking`"
        )
    num_chunks = hidden_states.shape[chunk_dim] // chunk_size
    if lora_scale is None:
        ff_output = torch.cat(
            [
                ff(hid_slice)
                for hid_slice in hidden_states.chunk(num_chunks , dim = chunk_dim)
            ],
            dim = chunk_dim
        )
    else:
        ff_output = torch.cat(
            [
                ff(hid_slice, scale = lora_scale)
                for hid_slice in hidden_states.chunk(num_chunks, dim=chunk_dim)
            ],
            dim = chunk_dim
        )  
    return ff_output    


@maybe_allow_in_graph
class GatedSelfAttentionDense(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, n_heads: int, d_heads: int):
        super().__init__()

        self.linear = nn.Linear(context_dim , query_dim)
        self.attn = Attention(query_dim=query_dim , heads= n_heads, dim_head=d_heads)
        self.ff = FeedForward(query_dim , activation_fn="geglu")
        self.norm1 = nn.LAyerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter("alpha_attn", nn.Parameter(torch.tensor(0.0)))
        self.register_parameter("alpha_dense", nn.Parameter(torch.tensor(0.0)))
        self.enabled = True 

    def forward(self, x:torch.Tensors, objs: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x 

        n_visual = x.shape[1]
        objs = self.linear(objs)    
        
        x = (
            x + self.alpha_attn.tanh()
            * self.attn(self.norm1(torch.cat([x , objs], dim = 1)))[: , :n_visual, :]
        )
        return x
    

@maybe_allow_in_graph
class BasicTransformerBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_attention_heads: int,
            attention_head_dim: int,
            dropout = 0.0,
            cross_attention_dim: Optional[int] = None,
            activation_fn: str = "geglu",
            num_embeds_ada_norm: Optional[int] = None,
            attention_bias: bool = False,
            only_cross_attention: bool = False,
            double_self_attention: bool = False,
            upcast_attention: bool = False,
            norm_element_affine: bool = True,
            norm_type: str = "layer_norm",
            norm_eps: float = 1e-5,
            final_dropout: bool = False,
            attention_type: str = "default" ,
            positional_embeddings: Optional[str] = None,
            num_positional_embeddings: Optional[int] = None,
            ada_norm_continous_conditioning_embedding_dim: Optional[int] = None,
            ada_norm_bias: Optional[int] = None,
            ff_inner_dim: Optional[int] = None,
            ff_bias: bool = True,
            attention_out_bias: bool = True,
    ):
        super().init()
        self.only_cross_attentions = only_cross_attention
        self.use_ada_layer_norm_zero = (
            num_embeds_ada_norm is not None
        ) and norm_type = "ada_norm_zero"
        self.use_ada_layer_norm = (
            num_embeds_ada_norm is not None
        ) and norm_type == "ada_norm"
        self.use_ada_layer_norm_single = norm_type == "ada_norm_single"
        self.use_layer_norm = norm_type == "layer_norm"
        self.use_ada_layer_norm_continuous = norm_type == "ada_norm_continuous"

        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined.Please make sure to "
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )  
        
        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )
        
        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(
                dim , max_seq_length=num_positional_embeddings
            )
        else:
            self.pos_embed = None

        if self.use_ada_layer_norm:
            self.norm1 = AdaLayerNorm(dim , num_embeds_ada_norm)
        elif self.use_ada_layer_norm_zero:
            self.norm1 = AdaLayerNormZero(dim , num_embeds_ada_norm)
        elif self.use_ada_layer_norm_continuous:
            self.norm1 = AdaLayerNormContinuous(
                dim,
                ada_norm_continous_conditioning_embedding_dim,
                norm_element_affine,
                norm_eps,
                ada_norm_bias,
                "rms_norm",
            )   
        else:
            self.norm1 = nn.LayerNorm(
                dim, elementwise_affine=norm_element_affine, eps = norm_eps
            )

        self.attn1 = Attention(
            query_dim = dim,
            heads = num_attention_heads,
            dim_head = attention_head_dim,
            dropout=dropout,
            bias = attention_bias,
            cross_attention_dim= cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
            out_bias= attention_out_bias
        )  

        if cross_attention_dim is not None or double_self_attention:
            if self.use_ada_layer_norm:
                self.norm2 = AdaLayerNorm(dim , num_embeds_ada_norm)
            elif self.use_ada_layer_norm_continuous:
                self.norm2 = AdaLayerNormContinuous(
                    dim,
                    ada_norm_continous_conditioning_embedding_dim,
                    norm_element_affine,
                    norm_eps,
                    ada_norm_bias,
                    "rms_norm",
                )
            else:
                self.norm2 = nn.LayerNorm(
                    dim, norm_eps, norm_element_affine
                )

            self.attn2 = Attention(
                query_dim = dim,
                cross_attention_dim=(
                    cross_attention_dim if not  double_self_attention else None
                ),
                heads = num_attention_heads,
                dim_head = attention_head_dim,
                dropout=dropout,
                bias = attention_bias,
                upcast_attention = upcast_attention,
                out_bias = attention_out_bias
            )   
        else:
            self.norm2 = None
            self.attn2 = None

        if self.use_ada_layer_norm_continuous:
            self.norm3 = AdaLayerNormContinuous(
                dim,
                ada_norm_continous_conditioning_embedding_dim,
                norm_element_affine,
                norm_eps,
                ada_norm_bias,
                "layer_norm",
            )      
        elif not self.use_ada_layer_norm_single:
            self.norm3 = nn.LayerNorm(dim, norm_eps, norm_element_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn = activation_fn,
            final_dropout = final_dropout,
            inner_dim = ff_inner_dim,
            bias = ff_bias
        ) 

        #fuser
        if attention_type == "gated" or attention_type == "gated-text-image":
            self.fuser = GatedSelfAttentionDense(
                dim , cross_attention_dim, num_attention_heads, attention_head_dim
            )    

        if self.use_ada_layer_norm_single:
            self.scale_shift_table = nn.Paramter(
                torch.randn(6 , dim)/dim**0.5
            )   

        self.chunk_size = None
        self.chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim:int = 0):
        self._chunk_size = chunk_size
        self._chunk_dim = dim    

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
            reference_feature = None,
            this_reference_feature_idx = 0,
            added_cond_kwargs: Optional[Dict[str , torch.Tensor]] = None,
    )->torch.FloatTensor:
        batch_size = hidden_states.shape[0]
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states , timestep) 
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa , shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states , timestep, class_labels, hidden_dtype = hidden_states.dtype
            )  
        elif self.use_layer_norm:
            norm_hidden_states = self.norm1(hidden_states)
        elif self.use_ada_layer_norm_continuous:
            norm_hidden_states = self.norm1(
                hidden_states, added_cond_kwargs["pooled_text_emb"]
            )              
        elif self.use_ada_layer_norm_single:
            shift_msa, scale_msa, gate_msa, shift_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(batch_size , 6, -1).chunk(6, dim=1)
            )    
            norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * \
                (1+scale_msa) + shift_msa
            norm_hidden_states = norm_hidden_states.squeeze(1)
        else:
            raise ValueError("Incorrect norm used")
        

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)
                 

        # Retrieve lora scale
        lora_scale = (
            cross_attention_kwargs.get("scale" , 1.0)
            if cross_attention_kwargs is not None
            else 1.0
        )         

        cross_attention_kwargs = (
            cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        ) 
        gligen_kwargs = cross_attention_kwargs.pop("gligen" , None)

        modify_norm_hidden_states = torch.cat(
            [norm_hidden_states , reference_feature[this_reference_feature_idx]] , dim = 1
        ) 

        this_reference_feature_idx +=1
        attn_output = self.attn1(
            modify_norm_hidden_states,
            encoder_hidden_states = (
                encoder_hidden_states if self.only_cross_attention else None
            ),
            attention_mask = attention_mask
            **cross_attention_kwargs
        )
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif self.use_ada_layer_norm_single:
            attn_output = gate_msa * attn_output

        hidden_states = attn_output[: , : hidden_states[-2], :] + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        if self.attn2 is not None:
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm2(hidden_states , timestep)
            elif self.use_ada_layer_norm_zero or self.use_layer_norm:
                norm_hidden_states = self.norm2(hidden_states)
            elif self.use_ada_layer_norm_single:
                norm_hidden_states = hidden_states
            elif self.use_ada_layer_norm_continuous:
                norm_hidden_states = self.norm2(
                    hidden_states , added_cond_kwargs["pooled_text_emb"]
                )
            else:
                raise ValueError("Incorrect norm")

            if self.pose_embed is not None and self.use_ada_layer_norm_single is False:
                norm_hidden_states = self.pose_embed(norm_hidden_states)  

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states   

        if self.use_ada_layer_norm_continuous:
            norm_hidden_states =  self.norm3(
                hidden_states , added_cond_kwargs["pooled_text_emb"]
            )  
        elif not self.use_ada_layer_norm_single:
            norm_hidden_States = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
                norm_hidden_states = (
                    norm_hidden_states *
                    (1 + scale_mlp[:, None]) + shift_mlp[:, None]
                )  
        if self.use_ada_layer_norm_single:
            norm_hidden_States = self.norm2(hidden_states)
            norm_hidden_States = norm_hidden_States * \
                (1 + scale_mlp) +shift_mlp

        if self._chunk_size is not None:
            ff_output = _chunked_feed_forward(
                self.ff,
                norm_hidden_states,
                self._chunk_dim,
                self.chunk_size,
                lora_scale=lora_scale,
            )           
        else:
            ff_output = self.ff(norm_hidden_States , scale = lora_scale)


        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.use_ada_layer_norm_single:
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)                   
        return hidden_states, this_reference_feature_idx




@maybe_allow_in_graph
class TemporalBasicTransformerBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            time_mix_inne_dim: int,
            num_attentoin_heads:int,
            attention_head_dim: int,
            cross_attention_dim: Optional[int] = None,    
    ) :
        super().__init__()
        self.is_res = dim == time_mix_inne_dim
        self.norm_in = nn.LayerNorm(dim)
        

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn  
        self.norm_in = nn.LayerNorm(dim)
        self.ff_in = FeedForward(
                dim,
                dim_out=time_mix_inne_dim,
                activation_fn="geglu",
            )
        
        self.norm1 = nn.LayerNorm(time_mix_inne_dim)
        self.attm1 = Attention(
            query_dim = time_mix_inne_dim,
            heads = num_attentoin_heads,
            dim_heads = attention_head_dim,
            cross_attention_dim = None, 
        )

        if cross_attention_dim is not None:
            self.norm2 = nn.LayerNorm(time_mix_inne_dim)
            self.attn2 = Attention(
                query_dim = time_mix_inne_dim,
                cross_attention_dim=cross_attention_dim,
                heads= num_attentoin_heads,
                dim_head=attention_head_dim,
            )
        else:
            self.norm2 = None
            self.attn2 = None    


    def set_chunk_feed_forward(self, chunk_size: Optional[int], **kwargs):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        # chunk dim should be hardcoded to 1 to have better speed vs. memory trade-off
        self._chunk_dim = 1

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            num_frames: int,
            encoder_hidden_states: Optional[torch.FloatTensor] = None, 
    ) -> torch.FloatTensor:
        batch_size = hidden_states.shape[0]
        batch_frames , seq_length, channels = hidden_states.shape 
        batch_size = batch_frames // num_frames
        hidden_states = hidden_states[None,:].reshape(
            batch_size , num_frames  , seq_length , channels
        )
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        hidden_states = hidden_states.reshape(
            batch_size*seq_length , num_frames, channels
        )

        residual = hidden_states 
        hidden_states = self.norm_in(hidden_states)

        if self._chunk_size is not None:
            hidden_states = _chunked_feed_forward(
                self.ff_in, hidden_states, self._chunk_dim, self._chunk_size
            )
        else:
            hidden_states = self.ff_in(hidden_states)    

        if self.is_res:
            hidden_states = hidden_states + residual

        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn1(
            norm_hidden_states, encoder_hidden_states=None)
        hidden_states = attn_output + hidden_states    

        # 3. Cross-Attention

        if self.attn2 is not None:
            norm_hidden_states = self.norm2(hidden_states)
            attn_output = self.attn2(
                norm_hidden_states, encoder_hidden_states=encoder_hidden_states
            )
            hidden_states = attn_output + hidden_states

        norm_hidden_states = self.norm3(hidden_states)

        if self._chunk_size is not None:
            ff_output = _chunked_feed_forward(
                self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size
            )
        else:
            ff_output = self.ff(norm_hidden_states)  

        if self.is_res:
            hidden_states = ff_output + hidden_states
        else:
            hidden_states = ff_output

        hidden_states = hidden_states[None, :].reshape(
            batch_size, seq_length, num_frames, channels
        )
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        hidden_states = hidden_states.reshape(
            batch_size * num_frames, seq_length, channels
        )

        return hidden_states  


class SkipFFTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        kv_input_dim: int,
        kv_input_dim_proj_use_bias: bool,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        attention_out_bias: bool = True,
    ):
        super().__init__()
        if kv_input_dim != dim:
            self.kv_mapper = nn.Linear(
                kv_input_dim, dim, kv_input_dim_proj_use_bias)
        else:
            self.kv_mapper = None

        self.norm1 = RMSNorm(dim, 1e-06)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            out_bias=attention_out_bias,
        )

        self.norm2 = RMSNorm(dim, 1e-06)

        self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            out_bias=attention_out_bias,
        )

    def forward(self, hidden_states, encoder_hidden_states, cross_attention_kwargs):
        cross_attention_kwargs = (
            cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        )

        if self.kv_mapper is not None:
            encoder_hidden_states = self.kv_mapper(
                F.silu(encoder_hidden_states))

        norm_hidden_states = self.norm1(hidden_states)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            **cross_attention_kwargs,
        )

        hidden_states = attn_output + hidden_states

        norm_hidden_states = self.norm2(hidden_states)

        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            **cross_attention_kwargs,
        )

        hidden_states = attn_output + hidden_states

        return hidden_states    
    

class FeedForward(nn.Module):
 

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim=None,
        bias: bool = True,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        linear_cls = LoRACompatibleLinear if not USE_PEFT_BACKEND else nn.Linear

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim, bias=bias)
        if activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim, bias=bias)

        self.net = nn.ModuleList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(linear_cls(inner_dim, dim_out, bias=bias))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        compatible_cls = (GEGLU,) if USE_PEFT_BACKEND else (
            GEGLU, LoRACompatibleLinear)
        for module in self.net:
            if isinstance(module, compatible_cls):
                hidden_states = module(hidden_states, scale)
            else:
                hidden_states = module(hidden_states)
        return hidden_states




                 








    


