import re
import time
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import Tensor


class PromptCompressor:
    """
    DAC prompt compressor adapted from the original DAC compressor.

    This version can reuse the already-loaded evaluation model/tokenizer and can
    compress token IDs directly, which avoids depending on the external DAC repo.
    """

    def __init__(self, model, tokenizer, device: Optional[Union[str, torch.device]] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device if device is not None else getattr(model, "device", "cuda" if torch.cuda.is_available() else "cpu")

    def normalize(self, tensor: Tensor) -> Tensor:
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
        min_val = tensor.min()
        max_val = tensor.max()
        if max_val - min_val > 1e-8:
            return (tensor - min_val) / (max_val - min_val)
        return torch.zeros_like(tensor)

    def get_ppl(
        self,
        context: str = "",
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple:
        with torch.no_grad():
            if input_ids is None:
                inputs = self.tokenizer(context, return_tensors="pt", add_special_tokens=False)
                input_ids = inputs["input_ids"].to(self.device)
                attention_mask = inputs["attention_mask"].to(self.device)
            else:
                input_ids = input_ids.to(self.device)
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids)
                attention_mask = attention_mask.to(self.device)

            self.model.eval()
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=return_attn,
                return_dict=True,
            )
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            token_losses = token_losses.view(shift_labels.size())

            if return_attn:
                all_heads_sum = None
                matrix_num = 0
                for layer_attn in outputs.attentions:
                    for head in range(layer_attn.shape[1]):
                        head_attn = layer_attn[0][head].squeeze()
                        all_heads_sum = head_attn if all_heads_sum is None else all_heads_sum + head_attn
                        matrix_num += 1
                column_sum = torch.sum(all_heads_sum, dim=0)[1:] / max(matrix_num, 1)
                return token_losses, input_ids, attention_mask, column_sum

            return token_losses, input_ids, attention_mask

    def _fuse_attn_ppl_additive(self, ppl: Tensor, attn: Tensor, alpha: float = 0.8) -> Tensor:
        ppl_norm = self.normalize(ppl)
        attn_norm = self.normalize(attn)
        return alpha * attn_norm + (1 - alpha) * ppl_norm

    def _fuse_attn_ppl_multiplicative(self, ppl: Tensor, attn: Tensor) -> Tensor:
        return torch.mul(ppl, attn)

    def _preserve_punctuation_mask(self, input_ids: Tensor, device: Union[str, torch.device]) -> Tensor:
        ids = input_ids[0].detach().cpu().tolist()
        decoded_tokens = [self.tokenizer.decode([token_id]) for token_id in ids]
        punct_pattern = re.compile(r"^\s*[^\w\s]+\s*$")
        special_tokens = {"<s>", "</s>", "[CLS]", "[SEP]", "<pad>"}
        preserve = [
            bool(punct_pattern.match(token)) or token in special_tokens
            for token in decoded_tokens
        ]
        return torch.tensor(preserve, dtype=torch.bool, device=device)

    def direct_compress(
        self,
        ppl: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        compress_ratio: float,
        preserve_punct: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if compress_ratio <= 0:
            return input_ids, attention_mask, torch.arange(input_ids.size(1), device=input_ids.device)

        total_tokens = ppl.numel()
        k = max(1, int(total_tokens * (1 - compress_ratio)))

        if preserve_punct:
            punct_mask = self._preserve_punctuation_mask(input_ids, ppl.device)
            ppl = ppl + 1e5 * punct_mask

        _, indices = torch.topk(ppl.view(-1), k=k, largest=True)
        sorted_indices = torch.sort(indices)[0]

        selected_input_ids = input_ids[:, sorted_indices]
        selected_attention_mask = attention_mask[:, sorted_indices]
        return selected_input_ids, selected_attention_mask, sorted_indices

    def direct_compress_attn(
        self,
        ppl: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        compress_ratio: float,
        attn_sum: Tensor,
        fusion: str = "additive",
        alpha: float = 0.8,
        preserve_punct: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if compress_ratio <= 0:
            indices = torch.arange(input_ids.size(1), device=input_ids.device)
            return input_ids, attention_mask, indices, attn_sum

        if fusion == "additive":
            score = self._fuse_attn_ppl_additive(ppl, attn_sum, alpha)
        elif fusion == "multiplicative":
            score = self._fuse_attn_ppl_multiplicative(ppl, attn_sum)
        else:
            raise ValueError("Fusion must be 'additive' or 'multiplicative'")

        total_tokens = score.numel()
        k = max(1, int(total_tokens * (1 - compress_ratio)))

        if preserve_punct:
            punct_mask = self._preserve_punctuation_mask(input_ids, score.device)
            score = score + 1e5 * punct_mask

        _, indices = torch.topk(score.view(-1), k=k, largest=True)
        sorted_indices = torch.sort(indices)[0]

        selected_input_ids = input_ids[:, sorted_indices]
        selected_attention_mask = attention_mask[:, sorted_indices]
        new_attn_sum = attn_sum[sorted_indices]
        return selected_input_ids, selected_attention_mask, sorted_indices, new_attn_sum

    def direct_compress_attn_wosucce(
        self,
        ppl: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        compress_ratio: float,
        attn_sum: Tensor,
        fusion: str = "additive",
        alpha: float = 0.8,
        preserve_punct: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if compress_ratio <= 0:
            indices = torch.arange(input_ids.size(1), device=input_ids.device)
            return input_ids, attention_mask, indices, attn_sum

        if fusion == "additive":
            score = self._fuse_attn_ppl_additive(ppl, attn_sum, alpha)
        elif fusion == "multiplicative":
            score = self._fuse_attn_ppl_multiplicative(ppl, attn_sum)
        else:
            raise ValueError("Fusion must be 'additive' or 'multiplicative'")

        total_tokens = score.numel()
        k = max(1, int(total_tokens * (1 - compress_ratio)))

        if preserve_punct:
            punct_mask = self._preserve_punctuation_mask(input_ids, score.device)
            score = score - 1e5 * punct_mask

        _, indices = torch.topk(score.view(-1), k=k, largest=True)
        sorted_indices = torch.sort(indices)[0]

        all_values = torch.arange(ppl.numel(), device=self.device)
        del_indices = all_values[~torch.isin(all_values, sorted_indices)]
        if del_indices.numel() > 1:
            differences = del_indices[1:] - del_indices[:-1]
            mask = torch.ones_like(del_indices, dtype=torch.bool)
            mask[1:] = differences == 1
            mask[0] = False
            for idx in range(1, len(mask)):
                if mask[idx - 1]:
                    mask[idx] = False
            filtered_indices = del_indices[mask]
            all_indices, _ = torch.sort(torch.cat((indices, filtered_indices)))
        else:
            all_indices = sorted_indices

        selected_input_ids = input_ids[:, all_indices]
        selected_attention_mask = attention_mask[:, all_indices]
        new_attn_sum = attn_sum[all_indices]
        return selected_input_ids, selected_attention_mask, sorted_indices, new_attn_sum

    def _compress_encoded(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor],
        compress_ratio: float,
        method: str,
        fusion: str,
        alpha: float,
        dyn_time: Optional[int],
        preserve_punct: bool,
    ) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Any]]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        seq_len = input_ids.size(1)
        if seq_len <= 1 or compress_ratio <= 0:
            kept = torch.arange(seq_len, device=input_ids.device)
            info = {"original_tokens": seq_len, "compressed_tokens": seq_len, "actual_ratio": 0.0, "kept_indices": kept.tolist()}
            return input_ids, attention_mask, kept, info

        if dyn_time is None:
            dyn_time = min(max(1, seq_len // 100), 15)

        if method == "ppl":
            ppl, input_ids, attention_mask = self.get_ppl(input_ids=input_ids, attention_mask=attention_mask)
            selected_input_ids, selected_attention_mask, kept_indices = self.direct_compress(
                ppl, input_ids[:, 1:], attention_mask[:, 1:], compress_ratio, preserve_punct
            )
        elif method == "attn_ppl":
            ppl, input_ids, attention_mask, attn_sum = self.get_ppl(input_ids=input_ids, attention_mask=attention_mask, return_attn=True)
            selected_input_ids, selected_attention_mask, kept_indices, _ = self.direct_compress_attn(
                ppl, input_ids[:, 1:], attention_mask[:, 1:], compress_ratio, attn_sum,
                fusion=fusion, alpha=alpha, preserve_punct=preserve_punct,
            )
        elif method == "dynamic_ppl":
            ppl, input_ids, attention_mask = self.get_ppl(input_ids=input_ids, attention_mask=attention_mask)
            real_ratio = 1 - (1 - compress_ratio) ** (1.0 / dyn_time)
            for _ in range(dyn_time):
                selected_input_ids, selected_attention_mask, kept_indices = self.direct_compress(
                    ppl, input_ids[:, 1:], attention_mask[:, 1:], real_ratio, preserve_punct
                )
                input_ids = torch.cat((input_ids[:, :1], selected_input_ids), dim=1)
                attention_mask = torch.cat((attention_mask[:, :1], selected_attention_mask), dim=1)
                ppl, input_ids, attention_mask = self.get_ppl(input_ids=input_ids, attention_mask=attention_mask)
            selected_input_ids = input_ids[:, 1:]
            selected_attention_mask = attention_mask[:, 1:]
        elif method in {"dynamic_attn_ppl", "dynamic_attn_ppl_wosucce"}:
            ppl, input_ids, attention_mask, attn_sum = self.get_ppl(input_ids=input_ids, attention_mask=attention_mask, return_attn=True)
            real_ratio = 1 - (1 - compress_ratio) ** (1.0 / dyn_time)
            compress_fn = self.direct_compress_attn_wosucce if method == "dynamic_attn_ppl_wosucce" else self.direct_compress_attn
            for _ in range(dyn_time):
                selected_input_ids, selected_attention_mask, kept_indices, attn_sum = compress_fn(
                    ppl, input_ids[:, 1:], attention_mask[:, 1:], real_ratio, attn_sum,
                    fusion=fusion, alpha=alpha, preserve_punct=preserve_punct,
                )
                input_ids = torch.cat((input_ids[:, :1], selected_input_ids), dim=1)
                attention_mask = torch.cat((attention_mask[:, :1], selected_attention_mask), dim=1)
                ppl, input_ids, attention_mask, attn_sum = self.get_ppl(
                    input_ids=input_ids, attention_mask=attention_mask, return_attn=True
                )
            selected_input_ids = input_ids[:, 1:]
            selected_attention_mask = attention_mask[:, 1:]
        else:
            raise ValueError(f"Unknown method: {method}")

        compressed_input_ids = torch.cat((input_ids[:, :1], selected_input_ids), dim=1)
        compressed_attention_mask = torch.cat((attention_mask[:, :1], selected_attention_mask), dim=1)
        compressed_tokens = compressed_input_ids.size(1)
        info = {
            "original_tokens": seq_len,
            "compressed_tokens": compressed_tokens,
            "actual_ratio": round(1 - (compressed_tokens / seq_len), 3),
            "kept_indices": kept_indices.detach().cpu().tolist(),
        }
        return compressed_input_ids, compressed_attention_mask, kept_indices, info

    def compress_input_ids(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        compress_ratio: float = 0.5,
        method: str = "dynamic_attn_ppl",
        fusion: str = "additive",
        alpha: float = 0.8,
        dyn_time: Optional[int] = None,
        preserve_punct: bool = False,
        return_info: bool = True,
    ) -> Union[Tensor, Dict[str, Any]]:
        start_time = time.time()
        assert 0 <= compress_ratio < 1, "compress_ratio must be in [0, 1)"
        compressed_input_ids, compressed_attention_mask, _, info = self._compress_encoded(
            input_ids=input_ids,
            attention_mask=attention_mask,
            compress_ratio=compress_ratio,
            method=method,
            fusion=fusion,
            alpha=alpha,
            dyn_time=dyn_time,
            preserve_punct=preserve_punct,
        )
        info.update({
            "compress_ratio": compress_ratio,
            "method": method,
            "fusion": fusion,
            "alpha": alpha,
            "dyn_time": dyn_time,
            "processing_time": round(time.time() - start_time, 3),
        })
        if return_info:
            return {
                "compressed_input_ids": compressed_input_ids,
                "compressed_attention_mask": compressed_attention_mask,
                **info,
            }
        return compressed_input_ids

    def compress_text(
        self,
        context: str,
        compress_ratio: float = 0.5,
        method: str = "dynamic_attn_ppl",
        fusion: str = "additive",
        alpha: float = 0.8,
        dyn_time: Optional[int] = None,
        preserve_punct: bool = False,
        return_info: bool = True,
    ) -> Union[str, Dict[str, Any]]:
        if not context.strip():
            context = " "

        inputs = self.tokenizer(context, return_tensors="pt", add_special_tokens=False)
        result = self.compress_input_ids(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            compress_ratio=compress_ratio,
            method=method,
            fusion=fusion,
            alpha=alpha,
            dyn_time=dyn_time,
            preserve_punct=preserve_punct,
            return_info=True,
        )
        compressed_text = self.tokenizer.decode(
            result["compressed_input_ids"][0].tolist(),
            skip_special_tokens=False,
        )
        if return_info:
            return {"compressed_text": compressed_text, **result}
        return compressed_text
