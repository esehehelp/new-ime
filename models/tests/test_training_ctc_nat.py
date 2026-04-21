import torch

from models.src.data.tokenizer import MASK_ID, PAD_ID
from models.src.model.ctc_nat import CTCAlignmentToken
from models.src.training.train_ctc_nat import (
    build_refinement_batch,
    build_refinement_batch_from_proposal,
    resolve_refine_loss_weight,
    resolve_refine_mask_ratio,
)


def test_build_refinement_batch_masks_only_valid_tokens():
    target_ids = torch.tensor(
        [
            [10, 11, 12, PAD_ID, PAD_ID],
            [20, 21, 22, 23, PAD_ID],
        ],
        dtype=torch.long,
    )
    target_lengths = torch.tensor([3, 4], dtype=torch.long)

    hyp_ids, hyp_attn, mask_positions = build_refinement_batch(
        target_ids,
        target_lengths,
        mask_ratio=1.0,
    )

    assert hyp_attn.tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]]
    assert torch.equal(mask_positions, hyp_attn.bool())
    assert torch.equal(hyp_ids[mask_positions], torch.full((7,), MASK_ID, dtype=torch.long))
    assert torch.equal(hyp_ids[~mask_positions], target_ids[~mask_positions])


def test_build_refinement_batch_forces_at_least_one_mask_per_nonempty_row():
    torch.manual_seed(0)
    target_ids = torch.tensor(
        [
            [10, 11, PAD_ID],
            [20, 21, 22],
        ],
        dtype=torch.long,
    )
    target_lengths = torch.tensor([2, 3], dtype=torch.long)

    _hyp_ids, _hyp_attn, mask_positions = build_refinement_batch(
        target_ids,
        target_lengths,
        mask_ratio=0.0,
    )

    assert mask_positions[0].sum().item() == 1
    assert mask_positions[1].sum().item() == 1


def test_build_refinement_batch_from_proposal_uses_matching_rows_and_falls_back():
    torch.manual_seed(0)
    target_ids = torch.tensor(
        [
            [10, 11, 12, PAD_ID],
            [20, 21, PAD_ID, PAD_ID],
        ],
        dtype=torch.long,
    )
    target_lengths = torch.tensor([3, 2], dtype=torch.long)
    alignments = [
        [
            CTCAlignmentToken(30, 0, 0, -0.2, -0.2, 0.5, 0.5),
            CTCAlignmentToken(31, 1, 1, -1.2, -1.2, 0.1, 0.1),
            CTCAlignmentToken(32, 2, 2, -0.4, -0.4, 0.4, 0.4),
        ],
        [CTCAlignmentToken(40, 0, 0, -0.3, -0.3, 0.2, 0.2)],
    ]

    hyp_ids, hyp_attn, mask_positions, used_rows = build_refinement_batch_from_proposal(
        target_ids,
        target_lengths,
        alignments,
        mask_ratio=1 / 3,
    )

    assert used_rows.tolist() == [True, False]
    assert hyp_attn.tolist() == [[1, 1, 1, 0], [1, 1, 0, 0]]
    assert hyp_ids[0].tolist().count(MASK_ID) == 1
    assert hyp_ids[0, :3].tolist().count(30) + hyp_ids[0, :3].tolist().count(31) + hyp_ids[0, :3].tolist().count(32) >= 2
    assert mask_positions[1].sum().item() == 1


def test_resolve_refine_loss_weight_warmup():
    class Args:
        refine_loss_weight = 0.4
        refine_warmup_steps = 10

    assert resolve_refine_loss_weight(Args, 0) == 0.0
    assert abs(resolve_refine_loss_weight(Args, 5) - 0.2) < 1e-6
    assert abs(resolve_refine_loss_weight(Args, 10) - 0.4) < 1e-6


def test_resolve_refine_mask_ratio_fixed_and_sampled():
    class FixedArgs:
        refine_mask_ratio = 0.3
        refine_mask_ratio_min = None
        refine_mask_ratio_max = None

    assert abs(resolve_refine_mask_ratio(FixedArgs) - 0.3) < 1e-6

    class SampleArgs:
        refine_mask_ratio = 0.3
        refine_mask_ratio_min = 0.1
        refine_mask_ratio_max = 0.2

    sampled = resolve_refine_mask_ratio(SampleArgs)
    assert 0.1 <= sampled <= 0.2
