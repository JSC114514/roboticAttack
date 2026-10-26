import torch

from sites import Site, nonselected_equal, replace_selected, selected, site_registry


def test_registry_has_all_mechanism_families():
    names = ["q_spatial_b08", "k_spatial_b16", "v_spatial_b22", "attn_spatial_b08",
             "mlp_spatial_b16", "block_spatial_b22", "block_cls_b08", "block_register_b16",
             "dino_wrist_output", "siglip_wrist_output", "projector_wrist_output"]
    values = site_registry(names)
    assert set(values) == set(names)
    assert values["q_spatial_b08"].headwise
    assert not values["block_register_b16"].headwise


def test_q_replacement_preserves_prefix():
    site = Site("q_spatial_b08", "q", 8, True)
    value = torch.arange(1 * 3 * 261 * 2).reshape(1, 3, 261, 2).float()
    replacement = torch.full_like(value[:, :, 5:], -7)
    changed = replace_selected(value, replacement, site)
    assert nonselected_equal(value, changed, site)
    assert torch.equal(selected(changed, site), replacement)


def test_v_replacement_preserves_q_k_and_prefix_v():
    heads, head_dim = 4, 3
    site = Site("v_spatial_b08", "v", 8, True)
    value = torch.arange(261 * 3 * heads * head_dim).reshape(1, 261, 3 * heads * head_dim).float()
    replacement = torch.full((1, heads, 256, head_dim), -9.)
    changed = replace_selected(value, replacement, site, heads=heads)
    assert nonselected_equal(value, changed, site, heads=heads)
    shaped = changed.reshape(1, 261, 3, heads, head_dim)
    actual = shaped[:, 5:, 2].permute(0, 2, 1, 3)
    assert torch.equal(actual, replacement)


def test_block_component_replacements_are_disjoint():
    value = torch.arange(261 * 6).reshape(1, 261, 6).float()
    cases = [
        (Site("block_spatial_b08", "block_spatial", 8, False), torch.full((1, 256, 6), -1.)),
        (Site("block_cls_b08", "block_cls", 8, False), torch.full((1, 1, 6), -2.)),
        (Site("block_register_b08", "block_register", 8, False), torch.full((1, 4, 6), -3.)),
    ]
    for site, replacement in cases:
        changed = replace_selected(value, replacement, site)
        assert nonselected_equal(value, changed, site)
        assert torch.equal(selected(changed, site), replacement)


def test_consumed_and_projector_replacements():
    dino = Site("dino_wrist_output", "dino", None, False)
    value = torch.zeros(1, 256, 5)
    replacement = torch.ones_like(value)
    assert torch.equal(replace_selected(value, replacement, dino), replacement)
    projector = Site("projector_wrist_output", "projector", None, False)
    fused = torch.arange(512 * 5).reshape(1, 512, 5).float()
    changed = replace_selected(fused, torch.full((1, 256, 5), -4.), projector)
    assert nonselected_equal(fused, changed, projector)
    assert torch.equal(selected(changed, projector), torch.full((1, 256, 5), -4.))
