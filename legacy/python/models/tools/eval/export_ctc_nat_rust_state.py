from __future__ import annotations

import argparse
from pathlib import Path

import torch


def split_qkv(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden = weight.shape[0] // 3
    return weight[:hidden], weight[hidden : 2 * hidden], weight[2 * hidden :]


def split_qkv_bias(bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden = bias.shape[0] // 3
    return bias[:hidden], bias[hidden : 2 * hidden], bias[2 * hidden :]


def map_encoder(remapped: dict[str, torch.Tensor], state: dict[str, torch.Tensor]) -> None:
    remapped["token_embed.weight"] = state["encoder.token_embedding.weight"]
    remapped["pos_embed.weight"] = state["encoder.pos_embedding.weight"]
    remapped["enc_final_norm.weight"] = state["encoder.final_norm.weight"]
    remapped["enc_final_norm.bias"] = state["encoder.final_norm.bias"]

    layer = 0
    while f"encoder.layers.layers.{layer}.self_attn.in_proj_weight" in state:
        prefix = f"encoder.layers.layers.{layer}"
        out = f"enc_{layer}"
        q_w, k_w, v_w = split_qkv(state[f"{prefix}.self_attn.in_proj_weight"])
        q_b, k_b, v_b = split_qkv_bias(state[f"{prefix}.self_attn.in_proj_bias"])
        remapped[f"{out}.self_attn.q_proj.weight"] = q_w
        remapped[f"{out}.self_attn.q_proj.bias"] = q_b
        remapped[f"{out}.self_attn.k_proj.weight"] = k_w
        remapped[f"{out}.self_attn.k_proj.bias"] = k_b
        remapped[f"{out}.self_attn.v_proj.weight"] = v_w
        remapped[f"{out}.self_attn.v_proj.bias"] = v_b
        remapped[f"{out}.self_attn.out_proj.weight"] = state[f"{prefix}.self_attn.out_proj.weight"]
        remapped[f"{out}.self_attn.out_proj.bias"] = state[f"{prefix}.self_attn.out_proj.bias"]
        remapped[f"{out}.ffn_in.weight"] = state[f"{prefix}.linear1.weight"]
        remapped[f"{out}.ffn_in.bias"] = state[f"{prefix}.linear1.bias"]
        remapped[f"{out}.ffn_out.weight"] = state[f"{prefix}.linear2.weight"]
        remapped[f"{out}.ffn_out.bias"] = state[f"{prefix}.linear2.bias"]
        remapped[f"{out}.norm1.weight"] = state[f"{prefix}.norm1.weight"]
        remapped[f"{out}.norm1.bias"] = state[f"{prefix}.norm1.bias"]
        remapped[f"{out}.norm2.weight"] = state[f"{prefix}.norm2.weight"]
        remapped[f"{out}.norm2.bias"] = state[f"{prefix}.norm2.bias"]
        layer += 1


def map_decoder(remapped: dict[str, torch.Tensor], state: dict[str, torch.Tensor]) -> None:
    remapped["proposal_pos_embed.weight"] = state["decoder.pos_embed.weight"]
    remapped["prop_final_norm.weight"] = state["decoder.final_norm.weight"]
    remapped["prop_final_norm.bias"] = state["decoder.final_norm.bias"]

    layer = 0
    while f"decoder.layers.{layer}.self_attn.in_proj_weight" in state:
        prefix = f"decoder.layers.{layer}"
        out = f"prop_{layer}"

        q_w, k_w, v_w = split_qkv(state[f"{prefix}.self_attn.in_proj_weight"])
        q_b, k_b, v_b = split_qkv_bias(state[f"{prefix}.self_attn.in_proj_bias"])
        remapped[f"{out}.self_attn.q_proj.weight"] = q_w
        remapped[f"{out}.self_attn.q_proj.bias"] = q_b
        remapped[f"{out}.self_attn.k_proj.weight"] = k_w
        remapped[f"{out}.self_attn.k_proj.bias"] = k_b
        remapped[f"{out}.self_attn.v_proj.weight"] = v_w
        remapped[f"{out}.self_attn.v_proj.bias"] = v_b
        remapped[f"{out}.self_attn.out_proj.weight"] = state[f"{prefix}.self_attn.out_proj.weight"]
        remapped[f"{out}.self_attn.out_proj.bias"] = state[f"{prefix}.self_attn.out_proj.bias"]
        remapped[f"{out}.self_attn_norm.weight"] = state[f"{prefix}.self_attn_norm.weight"]
        remapped[f"{out}.self_attn_norm.bias"] = state[f"{prefix}.self_attn_norm.bias"]

        q_w, k_w, v_w = split_qkv(state[f"{prefix}.cross_attn.in_proj_weight"])
        q_b, k_b, v_b = split_qkv_bias(state[f"{prefix}.cross_attn.in_proj_bias"])
        remapped[f"{out}.cross_attn.q_proj.weight"] = q_w
        remapped[f"{out}.cross_attn.q_proj.bias"] = q_b
        remapped[f"{out}.cross_attn.k_proj.weight"] = k_w
        remapped[f"{out}.cross_attn.k_proj.bias"] = k_b
        remapped[f"{out}.cross_attn.v_proj.weight"] = v_w
        remapped[f"{out}.cross_attn.v_proj.bias"] = v_b
        remapped[f"{out}.cross_attn.out_proj.weight"] = state[f"{prefix}.cross_attn.out_proj.weight"]
        remapped[f"{out}.cross_attn.out_proj.bias"] = state[f"{prefix}.cross_attn.out_proj.bias"]
        remapped[f"{out}.cross_attn_norm.weight"] = state[f"{prefix}.cross_attn_norm.weight"]
        remapped[f"{out}.cross_attn_norm.bias"] = state[f"{prefix}.cross_attn_norm.bias"]

        remapped[f"{out}.ffn_in.weight"] = state[f"{prefix}.ffn.0.weight"]
        remapped[f"{out}.ffn_in.bias"] = state[f"{prefix}.ffn.0.bias"]
        remapped[f"{out}.ffn_out.weight"] = state[f"{prefix}.ffn.3.weight"]
        remapped[f"{out}.ffn_out.bias"] = state[f"{prefix}.ffn.3.bias"]
        remapped[f"{out}.ffn_norm.weight"] = state[f"{prefix}.ffn_norm.weight"]
        remapped[f"{out}.ffn_norm.bias"] = state[f"{prefix}.ffn_norm.bias"]
        layer += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]

    remapped: dict[str, torch.Tensor] = {}
    map_encoder(remapped, state)
    map_decoder(remapped, state)
    torch.save(remapped, args.output)
    print(Path(args.output))


if __name__ == "__main__":
    main()
