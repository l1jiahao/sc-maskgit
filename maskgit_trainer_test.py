# train MaskGit
import torch
from muse_maskgit_pytorch import VQGanVAE, MaskGit, MaskGitTransformer
# (1) create your transformer / attention network

transformer = MaskGitTransformer(
    num_tokens=51,  # must be same as codebook size above
    seq_len=1200,  # must be equivalent to fmap_size ** 2 in vae
    dim=512,  # model dimension
    depth=3,  # depth
    dim_head=64,  # attention head dimension
    heads=8,  # attention heads,
    ff_mult=4,  # feedforward expansion factor
    t5_name="t5-small",  # name of your T5, 如果没有 text 可以不用
)

# (2) pass your trained VAE and the base transformer to MaskGit

# dtype != float 可以不用 vae
base_maskgit = MaskGit(
    vae=None,  # vqgan vae
    transformer=transformer,  # transformer
    image_size=256,  # image size, 如果不做图像超分可以不用
    cond_drop_prob=0.25,  # conditional dropout, for classifier free guidance
).cuda()

# ready your training text and images

# texts = [
#     "a child screaming at finding a worm within a half-eaten apple",
#     "lizard running across the desert on two feet",
#     "waking up to a psychedelic landscape",
#     "seashells sparkling in the shallow waters",
# ]

images = torch.randint(0, 51, (4, 1200)).cuda()
texts = torch.randint(0, 12, (4, 1)).cuda()

# feed it into your maskgit instance, with return_loss set to True

# QST: 这里 text 如何起作用
loss = base_maskgit(images, texts=texts)

loss.backward()

# 生成这一步一定要 text 了
pred_ids = base_maskgit.generate(texts, seq_len=1200)
