import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import os
import torch
import torch.distributed as dist
from torch import nn

from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
import transformers
from transformers.generation.utils import SampleOutput

import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import jensenshannon

def softmax_with_temperature(logits, temperature=1.0):
    logits_scaled = logits / temperature
    exps = np.exp(logits_scaled - np.max(logits_scaled))  # TODO 减去最大值以提高数值稳定性
    probabilities = exps / np.sum(exps)
    
    return probabilities

def cal_JS_div(p_logits, q_logits, temperature=1.0, clip_value=5e-8):
    p_probs = softmax_with_temperature(p_logits, temperature)
    q_probs = softmax_with_temperature(q_logits, temperature)

    p_probs = np.clip(p_probs, clip_value, 1.0-clip_value)
    q_probs = np.clip(q_probs, clip_value, 1.0-clip_value)

    m = 0.5 * (p_probs + q_probs)

    kl_p_m = np.sum(p_probs * np.log(p_probs / m), axis=-1)
    kl_q_m = np.sum(q_probs * np.log(q_probs / m), axis=-1)

    jsd = 0.5 * (kl_p_m + kl_q_m)
    return jsd

def sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    use_avisc : Optional[bool] = False,
    use_JS_consis : Optional[bool] = False,
    JS_PH_alpha : Optional[bool] = False,
    JS_MH_alpha : Optional[bool] = False,
    JS_beta : Optional[bool] = False,
    JS_top_attn : Optional[bool] = False,
    JS_threshold : Optional[bool] = False,
    use_m3id : Optional[bool] = False,
    **model_kwargs,
) -> Union[SampleOutput, torch.LongTensor]:
    assert use_JS_consis + use_avisc + use_m3id + use_cd <= 1
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id

    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )

    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
    this_peer_finished = False

    model_kwargs_avisc = copy.deepcopy(model_kwargs)
    model_kwargs_cd = copy.deepcopy(model_kwargs)
    model_kwargs_m3id = copy.deepcopy(model_kwargs)
    model_kwargs_JS = copy.deepcopy(model_kwargs)
    m3id_t = 1
    JS_mode = 0
    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tesnsor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        output_attentions = (use_avisc or use_JS_consis) and not (model_kwargs.get("use_cache") and model_kwargs.get("past_key_values") is not None)
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions, 
            output_hidden_states=output_hidden_states, 
        )

        if synced_gpus and this_peer_finished:
            continue

        next_token_logits = outputs.logits[:, -1, :]
        
        ## For contrastive decoding initial
        use_cd = model_kwargs.get("images_cd") != None
        output_attentions_wo_img = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states_wo_img = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )

        if use_JS_consis:
            masking_scheme = model_kwargs.get("masking_scheme")
            model_name = model_kwargs.get("model_name") if model_kwargs.get("model_name") is not None else "llava" 
            
            model_inputs_JS = self.prepare_inputs_for_generation_method(input_ids, **model_kwargs_JS)
            
            if model_inputs_JS.get("past_key_values") is None:
                attention = outputs.attentions             
                if model_name.lower() == "llava-1.5" or model_name.lower() == "llava-next":
                    img_idx = list(range(35, 35+576))
                elif model_name.lower() == "qwen-vl":
                    img_idx = list(range(1, 1+256))
                    
                layer_img_att_portion = []
                sum_layer_img_att = torch.zeros([1, len(img_idx)]).cuda()
                for logit in outputs.attentions:
                    img_logit = logit.mean(dim=1)[:,-1, img_idx]
                    sum_layer_img_att += img_logit
                    
                img_att_logits = sum_layer_img_att
                
                top_mask = int(img_att_logits.size()[1] * (1-JS_top_attn))
                _, mask_low_idx = torch.topk(img_att_logits, top_mask, largest=False)

            output_attentions_JS = False
            model_inputs_JS.update(
                {
                    "mask_idx": mask_low_idx,
                    "masking_scheme": masking_scheme
                }
            )
            
            outputs_JS = self(
                **model_inputs_JS,
                return_dict=True,
                output_attentions=output_attentions_JS,
                output_hidden_states=output_hidden_states,
            )
            
            next_token_logits_JS = outputs_JS.logits[:, -1, :]
            
            if torch.isnan(next_token_logits_JS).any():
                next_token_logits_JS = next_token_logits   
                
            P = next_token_logits[0].float().cpu().numpy()
            Q = next_token_logits_JS[0].float().cpu().numpy()

            cutoff = torch.log(torch.tensor(JS_beta)) + next_token_logits.max(dim=-1, keepdim=True).values  # 自适应合理性约束
            
            if JS_mode == 0:
                JS = cal_JS_div(P, Q)
                if JS < JS_threshold:
                    JS_mode = 1
                else:
                    JS_mode = -1
            
            if JS_mode == 1:
                diffs = next_token_logits + JS_PH_alpha * next_token_logits_JS
            elif JS_mode == -1:
                diffs = (1+JS_MH_alpha)*next_token_logits - JS_MH_alpha * next_token_logits_JS

            MoD_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))
            MoD_logits = logits_processor(input_ids, MoD_logits)
            MoD_logits = logits_warper(input_ids, MoD_logits)

            next_token_scores = MoD_logits
            MoD_probs = nn.functional.softmax(MoD_logits, dim=-1)
            next_tokens = torch.multinomial(MoD_probs, num_samples=1).squeeze(1)

        elif use_avisc:
            layer_gamma = model_kwargs.get("layer_gamma")
            masking_scheme = model_kwargs.get("masking_scheme")
            lamb = model_kwargs.get("lamb")
            model_name = model_kwargs.get("model_name") if model_kwargs.get("model_name") is not None else "qwen-vl" 
            
            def count_top_p(img_att_logit, top_p=0.8):
                """
                img_att_logit: torch.Tensor, shape (1, N)
                """
                norm_img_att_logit = img_att_logit / img_att_logit.sum()
                sorted_img_att_logit = torch.sort(norm_img_att_logit, descending=True)[0]
                
                return (torch.cumsum(sorted_img_att_logit, dim=1) < top_p).sum() + 1
            
            model_inputs_avisc = self.prepare_inputs_for_generation_method(input_ids, **model_kwargs_avisc)
            
            mask_idx = None
            if model_inputs_avisc.get("past_key_values") is None:
                attention = outputs.attentions             
                if model_name.lower() == "llava-1.5" or model_name.lower() == "llava-next":
                    img_idx = list(range(35, 35+576))
                elif model_name.lower() == "qwen-vl":
                    img_idx = list(range(1, 1+256))
                    
                layer_img_att_portion = []
                for logit in outputs.attentions:
                    img_logit = logit.mean(dim=1)[:,-1, img_idx]
                    layer_img_att_portion.append(img_logit.sum())
                    
                layer_img_att_portion = torch.stack(layer_img_att_portion, dim=0)
                total_img_att_portion = layer_img_att_portion.sum()
                layer_img_att_portion = layer_img_att_portion / total_img_att_portion
                k = count_top_p(layer_img_att_portion.unsqueeze(0), top_p=float(layer_gamma))
                
                _, top_k_lay_idx = torch.topk(layer_img_att_portion.float(), k, dim=0)
                
                att_logits = torch.stack([attention[i].mean(dim=1)[:,-1,img_idx] for i in top_k_lay_idx], dim=1)  # [batch_size, num_layer, seq_len]
                img_att_logits = att_logits.mean(dim=1)
                
                mask_idx = torch.where(img_att_logits < img_att_logits.mean() + img_att_logits.std() * lamb)[1].unsqueeze(0)
                
            output_attentions_method = False
            model_inputs_avisc.update(
                {
                    "mask_idx": mask_idx,
                    "masking_scheme": masking_scheme
                }
            )

            outputs_method = self(
                **model_inputs_avisc,
                return_dict=True,
                output_attentions=output_attentions_method,
                output_hidden_states=output_hidden_states,
            )
            
            next_token_logits_method = outputs_method.logits[:, -1, :]
            
            if torch.isnan(next_token_logits_method).any():
                next_token_logits_method = next_token_logits   
            
            avisc_alpha = model_kwargs.get("cd_alpha") if model_kwargs.get("cd_alpha") is not None else 1.0
            avisc_beta = model_kwargs.get("cd_beta") if model_kwargs.get("cd_beta") is not None else 0.1
            
            cutoff = torch.log(torch.tensor(avisc_beta)) + next_token_logits.max(dim=-1, keepdim=True).values
            diffs = (1+avisc_alpha)*next_token_logits - avisc_alpha*next_token_logits_method
            avisc_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))
            avisc_logits = logits_processor(input_ids, avisc_logits)
            avisc_logits = logits_warper(input_ids, avisc_logits)

            next_token_scores = avisc_logits
            avisc_probs = nn.functional.softmax(avisc_logits, dim=-1)
            next_tokens = torch.multinomial(avisc_probs, num_samples=1).squeeze(1)
            
        elif use_cd:
            ## cd_comments: forward pass of the model with distorted image input
            model_inputs_cd = self.prepare_inputs_for_generation_cd(input_ids, **model_kwargs_cd)
            outputs_cd = self(
                **model_inputs_cd,
                return_dict=True,
                output_attentions=output_attentions_wo_img,
                output_hidden_states=output_hidden_states_wo_img,
            )
            next_token_logits_cd = outputs_cd.logits[:, -1, :]
            
            ## cd_comments: pre-process logits from contrastive inputs
            cd_alpha = model_kwargs.get("cd_alpha") if model_kwargs.get("cd_alpha") is not None else 1.0
            cd_beta = model_kwargs.get("cd_beta") if model_kwargs.get("cd_beta") is not None else 0.1

            # version 2 set cutoff for Adaptive Plausibility Constraints
            cutoff = torch.log(torch.tensor(cd_beta)) + next_token_logits.max(dim=-1, keepdim=True).values
            
            diffs = (1+cd_alpha)*next_token_logits - cd_alpha*next_token_logits_cd
            cd_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))

            ## cd_comments: apply temperature warping and top-k filtering in contrastive decoding
            cd_logits = logits_processor(input_ids, cd_logits)
            cd_logits = logits_warper(input_ids, cd_logits)

            next_token_scores = cd_logits
            cd_probs = nn.functional.softmax(cd_logits, dim=-1)
            next_tokens = torch.multinomial(cd_probs, num_samples=1).squeeze(1)

        elif use_m3id:
            import math
            lamda = 0.02
            gamma_t = math.exp(-lamda * m3id_t)
            m3id_t += 1
            
            model_inputs_m3id = self.prepare_inputs_for_generation_m3id(input_ids, **model_kwargs_m3id)
            outputs_m3id = self(
                **model_inputs_m3id,
                return_dict=True,
                output_attentions=output_attentions_wo_img,
                output_hidden_states=output_hidden_states_wo_img,
            )
            next_token_logits_m3id = outputs_m3id.logits[:, -1, :]
            
            cd_beta = model_kwargs.get("cd_beta") if model_kwargs.get("cd_beta") is not None else 0.1
            cutoff = torch.log(torch.tensor(cd_beta)) + next_token_logits.max(dim=-1, keepdim=True).values
            
            lc = torch.log_softmax(next_token_logits, dim=-1)
            lu = torch.log_softmax(next_token_logits_m3id, dim=-1)
            m3id_logit = lc + ((1-gamma_t)/gamma_t)*(lc - lu)
            m3id_logit = m3id_logit.masked_fill(next_token_logits < cutoff, -float("inf"))
            
            m3id_logit = logits_processor(input_ids, m3id_logit)
            m3id_logit = logits_warper(input_ids, m3id_logit)
            
            next_token_scores = m3id_logit
            m3id_probs = nn.functional.softmax(m3id_logit, dim=-1)
            next_tokens = torch.multinomial(m3id_probs, num_samples=1).squeeze(1)

        else:
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)
            next_token_scores = next_token_scores
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )


        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        if use_JS_consis:
            model_kwargs_JS = self._update_model_kwargs_for_generation(
                outputs_JS, model_kwargs_JS, is_encoder_decoder=self.config.is_encoder_decoder
            )
        elif use_avisc:
            model_kwargs_avisc = self._update_model_kwargs_for_generation(
                outputs_method, model_kwargs_avisc, is_encoder_decoder=self.config.is_encoder_decoder
            )
        elif use_cd:
            model_kwargs_cd = self._update_model_kwargs_for_generation(
                outputs_cd, model_kwargs_cd, is_encoder_decoder=self.config.is_encoder_decoder
            )
        elif use_m3id:
            model_kwargs_cd = self._update_model_kwargs_for_generation(
                outputs_m3id, model_kwargs_m3id, is_encoder_decoder=self.config.is_encoder_decoder
            )

        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return SampleEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return SampleDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return input_ids

def evolve_MoD_sampling():
    transformers.generation.utils.GenerationMixin.sample = sample

