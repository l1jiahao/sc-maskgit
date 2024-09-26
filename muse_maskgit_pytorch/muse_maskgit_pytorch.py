import math
from random import random
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn, einsum
import pathlib
from pathlib import Path
import torchvision.transforms as T

from typing import Callable, Optional, List

from einops import rearrange, repeat

from beartype import beartype

from muse_maskgit_pytorch.vqgan_vae import VQGanVAE
from muse_maskgit_pytorch.t5 import t5_encode_text, get_encoded_dim, DEFAULT_T5_NAME
from muse_maskgit_pytorch.attend import Attend

from tqdm.auto import tqdm

# helpers


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


def l2norm(t):
    return F.normalize(t, dim=-1)


# tensor helpers


def get_mask_subset_prob(mask, prob, min_mask=0):
    batch, seq, device = *mask.shape, mask.device
    num_to_mask = (mask.sum(dim=-1, keepdim=True) * prob).clamp(min=min_mask)
    logits = torch.rand((batch, seq), device=device)
    logits = logits.masked_fill(~mask, -1)

    randperm = logits.argsort(dim=-1).argsort(dim=-1).float()

    num_padding = (~mask).sum(dim=-1, keepdim=True)
    randperm -= num_padding

    subset_mask = randperm < num_to_mask
    subset_mask.masked_fill_(~mask, False)
    return subset_mask


# classes


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class GEGLU(nn.Module):
    """https://arxiv.org/abs/2002.05202"""

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return gate * F.gelu(x)


def FeedForward(dim, mult=4):
    """https://arxiv.org/abs/2110.09456"""

    inner_dim = int(dim * mult * 2 / 3)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        GEGLU(),
        LayerNorm(inner_dim),
        nn.Linear(inner_dim, dim, bias=False),
    )


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    将 time emb 成 frequency_embedding_size 维，再投影到 hidden_size
    """

    def __init__(self, hidden_size=512, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

        # 将输入 emb 成 frequency_embedding_size 维
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        # 最后一维加 None 相当于拓展一维
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        # 采用 pos emb 之后过 mlp
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        cross_attend=False,
        scale=8,
        flash=True,
        dropout=0.0,
    ):
        super().__init__()
        self.scale = scale
        self.heads = heads
        inner_dim = dim_head * heads

        self.cross_attend = cross_attend
        self.norm = LayerNorm(dim)

        self.attend = Attend(flash=flash, dropout=dropout, scale=scale)

        self.null_kv = nn.Parameter(torch.randn(2, heads, 1, dim_head))

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, context=None, context_mask=None):
        assert not (exists(context) ^ self.cross_attend)

        n = x.shape[-2]
        h, is_cross_attn = self.heads, exists(context)

        x = self.norm(x)

        kv_input = context if self.cross_attend else x

        q, k, v = (self.to_q(x), *self.to_kv(kv_input).chunk(2, dim=-1))

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        nk, nv = self.null_kv
        nk, nv = map(lambda t: repeat(t, "h 1 d -> b h 1 d", b=x.shape[0]), (nk, nv))

        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        if exists(context_mask):
            context_mask = repeat(context_mask, "b j -> b h i j", h=h, i=n)
            context_mask = F.pad(context_mask, (1, 0), value=True)

        out = self.attend(q, k, v, mask=context_mask)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class TransformerBlocks(nn.Module):
    def __init__(self, *, dim, depth, dim_head=64, heads=8, ff_mult=4, flash=True):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim=dim, dim_head=dim_head, heads=heads, flash=flash),
                        Attention(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            cross_attend=True,
                            flash=flash,
                        ),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = LayerNorm(dim)

    def forward(self, x, context=None, context_mask=None):
        for attn, cross_attn, ff in self.layers:
            x = attn(x) + x

            x = cross_attn(x, context=context, context_mask=context_mask) + x

            x = ff(x) + x

        return self.norm(x)


# transformer - it's all we need


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        seq_len,
        dim_out=None,
        t5_name=DEFAULT_T5_NAME,
        self_cond=False,
        add_mask_id=False,
        num_condition=9,
        focal_gamma=2,
        focal_alpha=0.01,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.mask_id = num_tokens if add_mask_id else None

        self.num_tokens = num_tokens
        # self.token_emb = nn.Embedding(num_tokens + int(add_mask_id), dim)
        self.token_emb = TimestepEmbedder(hidden_size=dim)
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.seq_len = seq_len

        self.transformer_blocks = TransformerBlocks(dim=dim, **kwargs)
        self.norm = LayerNorm(dim)

        self.dim_out = default(dim_out, num_tokens)
        self.focal_loss = torch.hub.load(
            "adeelh/pytorch-multi-class-focal-loss",
            model="FocalLoss",
            alpha=torch.tensor([focal_alpha] + [1] * (num_tokens - 1)),
            gamma=focal_gamma,
            reduction="mean",
            force_reload=False,
            ignore_index=-1,
        )
        print("use focal loss!")
        self.to_logits = nn.Linear(dim, self.dim_out, bias=False)

        # text conditioning

        # NOTE: 更改 encode text 函数为 nn.embedding
        # self.encode_text = partial(t5_encode_text, name=t5_name)
        self.encode_text = nn.Embedding(num_condition, dim)

        text_embed_dim = get_encoded_dim(t5_name)

        self.text_embed_proj = (
            nn.Linear(text_embed_dim, dim, bias=False)
            if text_embed_dim != dim
            else nn.Identity()
        )

        # optional self conditioning

        self.self_cond = self_cond
        self.self_cond_to_init_embed = FeedForward(dim)

    def forward_with_cond_scale(
        self, *args, cond_scale=3.0, return_embed=False, **kwargs
    ):
        if cond_scale == 1:
            return self.forward(
                *args, return_embed=return_embed, cond_drop_prob=0.0, **kwargs
            )

        logits, embed = self.forward(
            *args, return_embed=True, cond_drop_prob=0.0, **kwargs
        )

        null_logits = self.forward(*args, cond_drop_prob=1.0, **kwargs)

        scaled_logits = null_logits + (logits - null_logits) * cond_scale

        if return_embed:
            return scaled_logits, embed

        return scaled_logits

    def forward_with_neg_prompt(
        self,
        text_embed: torch.Tensor,
        neg_text_embed: torch.Tensor,
        cond_scale=3.0,
        return_embed=False,
        **kwargs,
    ):
        neg_logits = self.forward(
            *args, neg_text_embed=neg_text_embed, cond_drop_prob=0.0, **kwargs
        )
        pos_logits, embed = self.forward(
            *args,
            return_embed=True,
            text_embed=text_embed,
            cond_drop_prob=0.0,
            **kwargs,
        )

        logits = neg_logits + (pos_logits - neg_logits) * cond_scale

        if return_embed:
            return scaled_logits, embed

        return scaled_logits

    def forward(
        self,
        x,
        return_embed=False,
        return_logits=False,
        labels=None,
        ignore_index=0,
        self_cond_embed=None,
        cond_drop_prob=0.0,
        conditioning_token_ids: Optional[torch.Tensor] = None,
        texts: Optional[List[str]] = None,
        text_embeds: Optional[torch.Tensor] = None,
    ):
        device, b, n = x.device, *x.shape
        assert n <= self.seq_len

        # prepare texts

        assert exists(texts) ^ exists(text_embeds)

        if exists(texts):
            text_embeds = self.encode_text(texts)

        # QST: 简单的线性层?
        context = self.text_embed_proj(text_embeds)

        context_mask = (text_embeds != 0).any(dim=-1)

        # classifier free guidance

        if cond_drop_prob > 0.0:
            mask = prob_mask_like((b, 1), 1.0 - cond_drop_prob, device)
            context_mask = context_mask & mask

        # concat conditioning image token ids if needed

        if exists(conditioning_token_ids):
            conditioning_token_ids = rearrange(
                conditioning_token_ids, "b ... -> b (...)"
            )
            cond_token_emb = self.token_emb(conditioning_token_ids)
            context = torch.cat((context, cond_token_emb), dim=-2)
            context_mask = F.pad(
                context_mask, (0, conditioning_token_ids.shape[-1]), value=True
            )

        # embed tokens

        if isinstance(self.token_emb, TimestepEmbedder):
            seq_len = x.shape[-1]
            x = self.token_emb(rearrange(x, "b n -> (b n)"))
            x = rearrange(x, "(b n) d -> b n d", n=seq_len, d=self.dim)
        else:
            x = self.token_emb(x)
        x = x + self.pos_emb(torch.arange(n, device=device))

        if self.self_cond:
            if not exists(self_cond_embed):
                self_cond_embed = torch.zeros_like(x)
            x = x + self.self_cond_to_init_embed(self_cond_embed)

        embed = self.transformer_blocks(x, context=context, context_mask=context_mask)

        # NOTE: 线性层
        logits = self.to_logits(embed)

        if return_embed:
            return logits, embed

        if not exists(labels):
            return logits

        if self.dim_out == 1:
            loss = F.binary_cross_entropy_with_logits(
                rearrange(logits, "... 1 -> ..."), labels
            )
        else:
            # NOTE: 真要做一个 65536 的分类
            # loss = F.cross_entropy(
            #     rearrange(logits, "b n c -> b c n"), labels, ignore_index=ignore_index
            # )
            loss = self.focal_loss(rearrange(logits, "b n c -> b c n"), labels)

        if not return_logits:
            return loss

        return loss, logits


# self critic wrapper


class SelfCritic(nn.Module):
    def __init__(self, net):
        super().__init__()
        # NOTE: 这里输入的是 transformer, 和 maskgit 用的 transformer 一样
        self.net = net
        self.to_pred = nn.Linear(net.dim, 1)

    def forward_with_cond_scale(self, x, *args, **kwargs):
        _, embeds = self.net.forward_with_cond_scale(
            x, *args, return_embed=True, **kwargs
        )
        return self.to_pred(embeds)

    def forward_with_neg_prompt(self, x, *args, **kwargs):
        _, embeds = self.net.forward_with_neg_prompt(
            x, *args, return_embed=True, **kwargs
        )
        return self.to_pred(embeds)

    def forward(self, x, *args, labels=None, **kwargs):
        # 过 transformer 得到 logits
        _, embeds = self.net(x, *args, return_embed=True, **kwargs)
        logits = self.to_pred(embeds)

        # NOTE: 调用会带 critic label, 但 critic 是一个 0,1 mask, 1 表示 x 和原始的 ids 不同
        if not exists(labels):
            return logits

        # 相当于 unsequeeze
        logits = rearrange(logits, "... 1 -> ...")
        # NOTE: 相当于要 logits 和 labels 接近, 需要自己明白哪里预测的不一样
        return F.binary_cross_entropy_with_logits(logits, labels)


# specialized transformers


class MaskGitTransformer(Transformer):
    def __init__(self, *args, **kwargs):
        assert "add_mask_id" not in kwargs
        super().__init__(*args, add_mask_id=True, **kwargs)


class TokenCritic(Transformer):
    def __init__(self, *args, **kwargs):
        assert "dim_out" not in kwargs
        super().__init__(*args, dim_out=1, **kwargs)


# classifier free guidance functions


def uniform(shape, min=0, max=1, device=None):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)


def prob_mask_like(shape, prob, device=None):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return uniform(shape, device=device) < prob


# sampling helpers


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)


def top_k(logits, thres=0.9):
    # 取 logits 的最后一维也就是 vocab size, 相当于每次取 (1-thres)*vocab size 个最大值
    k = math.ceil((1 - thres) * logits.shape[-1])
    # logits 中最后一维取最大的 k 个元素
    val, ind = logits.topk(k, dim=-1)
    probs = torch.full_like(logits, float("-inf"))
    # 在第二维, ind 位置 插入 val, 其他置为 -inf
    probs.scatter_(2, ind, val)
    return probs


# noise schedules


def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)


# main maskgit classes


@beartype
class MaskGit(nn.Module):
    def __init__(
        self,
        image_size,
        transformer: MaskGitTransformer,
        noise_schedule: Callable = cosine_schedule,
        token_critic: Optional[TokenCritic] = None,
        self_token_critic=False,
        vae: Optional[VQGanVAE] = None,
        cond_vae: Optional[VQGanVAE] = None,
        cond_image_size=None,
        cond_drop_prob=0.5,
        self_cond_prob=0.9,
        no_mask_token_prob=0.0,
        critic_loss_weight=1.0,
    ):
        super().__init__()
        # NOTE: vae 只做 eval
        self.vae = vae.copy_for_eval() if exists(vae) else None

        if exists(cond_vae):
            self.cond_vae = cond_vae.eval()
        else:
            self.cond_vae = self.vae

        # QST: cond vae 用来做超分辨率?
        assert not (
            exists(cond_vae) and not exists(cond_image_size)
        ), "cond_image_size must be specified if conditioning"

        self.image_size = image_size
        # NOTE: cond_image_size 用来做超分辨率
        self.cond_image_size = cond_image_size
        self.resize_image_for_cond_image = exists(cond_image_size)

        self.cond_drop_prob = cond_drop_prob

        self.transformer = transformer
        self.self_cond = transformer.self_cond

        # QST: transformer.num_tokens 是什么
        if exists(self.vae):
            assert (
                self.vae.codebook_size
                == self.cond_vae.codebook_size
                == transformer.num_tokens
            ), "transformer num_tokens must be set to be equal to the vae codebook size"

        # QST: mask id 什么作用?
        self.mask_id = transformer.mask_id
        self.noise_schedule = noise_schedule

        # QST: 什么是 critic token
        assert not (self_token_critic and exists(token_critic))
        self.token_critic = token_critic

        if self_token_critic:
            self.token_critic = SelfCritic(transformer)

        self.critic_loss_weight = critic_loss_weight

        # self conditioning
        # QST: self_cond_prob 是什么, 默认 0.9
        self.self_cond_prob = self_cond_prob

        # percentage of tokens to be [mask]ed to remain the same token, so that transformer produces better embeddings across all tokens as done in original BERT paper
        # may be needed for self conditioning
        # QST: 默认是 0
        self.no_mask_token_prob = no_mask_token_prob

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        state_dict = torch.load(str(path))
        self.load_state_dict(state_dict)

    # NOTE: 直接开启 eval 模式
    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        texts: List[str],
        seq_len: int,
        negative_texts: Optional[List[str]] = None,
        cond_images: Optional[torch.Tensor] = None,
        fmap_size=None,
        temperature=1.0,
        topk_filter_thres=0.9,
        can_remask_prev_masked=False,
        force_not_use_token_critic=False,
        timesteps=18,  # ideal number of steps is 18 in maskgit paper
        cond_scale=3,  # cfg scale
        critic_noise_scale=1,
    ):
        # QST: 这里是 feature map 的 size?
        # fmap_size = default(fmap_size, self.vae.get_encoded_fmap_size(self.image_size))

        # begin with all image token ids masked

        device = next(self.parameters()).device

        # NOTE: 这里 seq 指的是 image token
        # seq_len = fmap_size**2

        batch_size = len(texts)

        shape = (batch_size, seq_len)

        # 一开始所有图片都设置成 mask_id
        ids = torch.full(shape, self.mask_id, dtype=torch.long, device=device)
        # QST: 这里是置信度? 越大代表置信度越低?
        scores = torch.zeros(shape, dtype=torch.float32, device=device)

        starting_temperature = temperature

        cond_ids = None

        text_embeds = self.transformer.encode_text(texts)

        # QST:这里是预测函数?
        demask_fn = self.transformer.forward_with_cond_scale

        # whether to use token critic for scores

        use_token_critic = exists(self.token_critic) and not force_not_use_token_critic

        # NOTE: 默认不使用 token critic
        if use_token_critic:
            token_critic_fn = self.token_critic.forward_with_cond_scale

        # negative prompting, as in paper

        neg_text_embeds = None
        # NOTE: 默认不使用 negative_texts
        if exists(negative_texts):
            assert len(texts) == len(negative_texts)

            neg_text_embeds = self.transformer.encode_text(negative_texts)
            demask_fn = partial(
                self.transformer.forward_with_neg_prompt,
                neg_text_embeds=neg_text_embeds,
            )

            if use_token_critic:
                token_critic_fn = partial(
                    self.token_critic.forward_with_neg_prompt,
                    neg_text_embeds=neg_text_embeds,
                )

        if self.resize_image_for_cond_image:
            assert exists(
                cond_images
            ), "conditioning image must be passed in to generate for super res maskgit"
            with torch.no_grad():
                _, cond_ids, _ = self.cond_vae.encode(cond_images)

        self_cond_embed = None

        # NOTE: 主体的采样函数, timesteps 默认为 18
        for timestep, steps_until_x0 in tqdm(
            zip(
                torch.linspace(0, 1, timesteps, device=device),
                reversed(range(timesteps)),
            ),
            total=timesteps,
        ):
            # NOTE: 简单的 cos 函数, timestep 从 0 到 1
            rand_mask_prob = self.noise_schedule(timestep)
            # NOTE: 比如 0.5 * 256 = 128
            num_token_masked = max(int((rand_mask_prob * seq_len).item()), 1)

            # NOTE: scores 越大表示置信度越低
            # NOTE: 从 scores 中取出 num_token_masked 个最大值的索引
            masked_indices = scores.topk(num_token_masked, dim=-1).indices

            # NOTE: ids 一开始全是 mask_id, 表示在第一维, mask_indices 上插入 mask_id
            ids = ids.scatter(1, masked_indices, self.mask_id)

            # 计算出新的 logits
            # NOTE: embed 是 cross attn 的输出, 并且这个 logits 是 CFG 过的
            logits, embed = demask_fn(
                ids,
                text_embeds=text_embeds,
                self_cond_embed=self_cond_embed,
                conditioning_token_ids=cond_ids,
                cond_scale=cond_scale,
                return_embed=True,
            )

            self_cond_embed = embed if self.self_cond else None

            # NOTE: 每个 logits 取出 (1-topk_filter_thres) 的分类概率, 其他为 -inf
            filtered_logits = top_k(logits, topk_filter_thres)

            # NOTE: steps_until_x0 是一个从 timesteps 到 0 的序列, 表示温度越来越低
            temperature = starting_temperature * (
                steps_until_x0 / timesteps
            )  # temperature is annealed

            # 用 gumbel sample 采出新的 ids
            # NOTE: logits 形状应该是 batch * seq_len * vocab size
            # NOTE: pred_ids 形状是 batch * seq_len, 即取了最大概率的那个预测值
            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)

            # NOTE: 在 mask_id 的位置填充 pred_ids
            is_mask = ids == self.mask_id
            ids = torch.where(is_mask, pred_ids, ids)

            # NOTE: 默认不使用
            if use_token_critic:
                scores = token_critic_fn(
                    ids,
                    text_embeds=text_embeds,
                    conditioning_token_ids=cond_ids,
                    cond_scale=cond_scale,
                )

                scores = rearrange(scores, "... 1 -> ...")

                scores = scores + (
                    uniform(scores.shape, device=device) - 0.5
                ) * critic_noise_scale * (steps_until_x0 / timesteps)

            else:
                # 简单通过 softmax 来得到置信度?
                probs_without_temperature = logits.softmax(dim=-1)

                # 通过调用 probs_without_temperature.gather(2, pred_ids[..., None])，我们从 probs_without_temperature 张量中提取了 pred_ids 所指定位置的概率值。
                # probs_without_temperature.gather(2, pred_ids[..., None]) 相当于是置信度了
                scores = 1 - probs_without_temperature.gather(2, pred_ids[..., None])
                # NOTE: 这里其实是 1 - 置信度了, scores 越大, 置信度越低
                scores = rearrange(scores, "... 1 -> ...")

                if not can_remask_prev_masked:
                    scores = scores.masked_fill(~is_mask, -1e5)
                else:
                    assert (
                        self.no_mask_token_prob > 0.0
                    ), "without training with some of the non-masked tokens forced to predict, not sure if the logits will be meaningful for these token"

        # get ids, 完成采样

        # ids = rearrange(ids, "b (i j) -> b i j", i=fmap_size, j=fmap_size)

        if not exists(self.vae):
            return ids

        images = self.vae.decode_from_ids(ids)
        return images

    def forward(
        self,
        images_or_ids: torch.Tensor,
        ignore_index=-1,
        cond_images: Optional[torch.Tensor] = None,
        cond_token_ids: Optional[torch.Tensor] = None,
        texts: Optional[torch.Tensor] = None,
        text_embeds: Optional[torch.Tensor] = None,
        cond_drop_prob=None,
        train_only_generator=False,
        sample_temperature=None,
    ):
        # tokenize if needed

        if images_or_ids.dtype == torch.float:
            assert exists(
                self.vae
            ), "vqgan vae must be passed in if training from raw images"

            assert all(
                [
                    height_or_width == self.image_size
                    for height_or_width in images_or_ids.shape[-2:]
                ]
            ), "the image you passed in is not of the correct dimensions"

            with torch.no_grad():
                _, ids, _ = self.vae.encode(images_or_ids)
        else:
            assert not self.resize_image_for_cond_image, "you cannot pass in raw image token ids if you want the framework to autoresize image for conditioning super res transformer"
            ids = images_or_ids

        # take care of conditioning image if specified

        if self.resize_image_for_cond_image:
            # NOTE: 插值
            cond_images_or_ids = F.interpolate(
                images_or_ids, self.cond_image_size, mode="nearest"
            )

        # get some basic variables

        # 相当于把 hw 展开
        # NOTE: 默认输入是 (4,16,16) 的 feature map
        ids = rearrange(ids, "b ... -> b (...)")

        batch, seq_len, device, cond_drop_prob = (
            *ids.shape,
            ids.device,
            default(cond_drop_prob, self.cond_drop_prob),
        )

        # tokenize conditional images if needed

        assert not (
            exists(cond_images) and exists(cond_token_ids)
        ), "if conditioning on low resolution, cannot pass in both images and token ids"

        if exists(cond_images):
            assert exists(self.cond_vae), "cond vqgan vae must be passed in"
            assert all(
                [
                    height_or_width == self.cond_image_size
                    for height_or_width in cond_images.shape[-2:]
                ]
            )

            with torch.no_grad():
                _, cond_token_ids, _ = self.cond_vae.encode(cond_images)

        # prepare mask

        rand_time = uniform((batch,), device=device)
        # 每条 sequence 有一个 mask prob
        # NOTE: mask prob 是一个 cosine schedule
        rand_mask_probs = self.noise_schedule(rand_time)
        # QST: 这不是相当于一条序列都会被 mask 吗? 形状是 b, 这是 threshold 相当于
        num_token_masked = (seq_len * rand_mask_probs).round().clamp(min=1)

        mask_id = self.mask_id
        # 生成 batch * seq_len 的随机数, 用于 mask
        batch_randperm = torch.rand((batch, seq_len), device=device).argsort(dim=-1)
        # NOTE: batch randperm 中小于 num_token_masked 是真正的 mask token
        mask = batch_randperm < rearrange(num_token_masked, "b -> b 1")

        # QST: transformer 的 mask_id 是什么
        mask_id = self.transformer.mask_id
        # 当mask中的某个位置为真时，torch.where会选择ids中对应位置的值；当mask中的某个位置为假时，torch.where会选择ignore_index。最终，labels张量将包含根据掩码条件选择的值。
        # ignore_index 是 -1
        # mask=0 为 ignore_index, mask=1 为 ids
        labels = torch.where(mask, ids, ignore_index)

        if self.no_mask_token_prob > 0.0:
            # NOTE: 默认是 0, 可以跳过
            no_mask_mask = get_mask_subset_prob(mask, self.no_mask_token_prob)
            mask &= ~no_mask_mask

        # mask 为真的时候, 选择 mask_id, 否则选择 ids
        # mask=0 为 ids, mask=1 为 mask_id
        x = torch.where(mask, mask_id, ids)
        # NOTE: 相当于用 mask_id 预测 ids, 也就是前面的 label, 其他的是正常的 ids

        # get text embeddings
        if exists(texts):
            text_embeds = self.transformer.encode_text(texts)
            texts = None

        # self conditioning

        self_cond_embed = None

        # NOTE: 这里是 CFG 的训练策略
        if self.transformer.self_cond and random() < self.self_cond_prob:
            with torch.no_grad():
                _, self_cond_embed = self.transformer(
                    x,
                    text_embeds=text_embeds,
                    # NOTE: cond_token_ids 是做超分时的输入
                    conditioning_token_ids=cond_token_ids,
                    cond_drop_prob=0.0,
                    return_embed=True,
                )

                self_cond_embed.detach_()

        # get loss

        # QST: 这里 ce loss 干嘛的?
        ce_loss, logits = self.transformer(
            x,
            text_embeds=text_embeds,
            self_cond_embed=self_cond_embed,
            conditioning_token_ids=cond_token_ids,
            labels=labels,
            cond_drop_prob=cond_drop_prob,
            ignore_index=ignore_index,
            return_logits=True,
        )

        # NOTE: 如果不存在 token critic 直接返回 ce loss
        if not exists(self.token_critic) or train_only_generator:
            return ce_loss

        # token critic loss

        # NOTE: 要使用 logits 来 sample
        sampled_ids = gumbel_sample(
            logits, temperature=default(sample_temperature, random())
        )

        # 还是根据 mask 来分配 sample id, mask=1 的地方用 sampled_id
        critic_input = torch.where(mask, sampled_ids, x)
        # critic labels 也是一个 mask, 比较 critic_input 和 ids
        critic_labels = (ids != critic_input).float()

        # NOTE: 相当于再过一次 transformer, 让 sample 的结果和原来的 ids 尽量接近
        bce_loss = self.token_critic(
            critic_input,
            text_embeds=text_embeds,
            conditioning_token_ids=cond_token_ids,
            labels=critic_labels,
            cond_drop_prob=cond_drop_prob,
        )

        return ce_loss + self.critic_loss_weight * bce_loss


# final Muse class


@beartype
class Muse(nn.Module):
    def __init__(self, base: MaskGit, superres: MaskGit):
        super().__init__()
        self.base_maskgit = base.eval()

        assert superres.resize_image_for_cond_image
        self.superres_maskgit = superres.eval()

    @torch.no_grad()
    def forward(
        self,
        texts: List[str],
        cond_scale=3.0,
        temperature=1.0,
        timesteps=18,
        superres_timesteps=None,
        return_lowres=False,
        return_pil_images=True,
    ):
        lowres_image = self.base_maskgit.generate(
            texts=texts,
            cond_scale=cond_scale,
            temperature=temperature,
            timesteps=timesteps,
        )

        superres_image = self.superres_maskgit.generate(
            texts=texts,
            cond_scale=cond_scale,
            cond_images=lowres_image,
            temperature=temperature,
            timesteps=default(superres_timesteps, timesteps),
        )

        if return_pil_images:
            lowres_image = list(map(T.ToPILImage(), lowres_image))
            superres_image = list(map(T.ToPILImage(), superres_image))

        if not return_lowres:
            return superres_image

        return superres_image, lowres_image
