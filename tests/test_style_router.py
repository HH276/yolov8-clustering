import unittest

import torch
from torch import nn

from utils.style_assignment import compute_soft_assignment
from utils.style_router import build_style_routing
from utils.style_adv_trainer import (
    compute_weighted_discriminator_loss,
    compute_weighted_generator_adv_loss,
)


class TinyDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Conv2d(4, 1, 1)

    def forward(self, x):
        return self.layer(x)


class StyleRouterTest(unittest.TestCase):
    def setUp(self):
        self.distance = torch.tensor([
            [0.1, 2.0, 3.0], [4.0, 0.2, 2.0], [3.0, 2.0, 0.3]
        ])
        self.hard = self.distance.argmin(dim=1)

    def route(self, mode, topk, floor=0.2):
        return build_style_routing(
            self.hard, self.distance, mode, topk, temperature=1.0,
            distance_scale=1.0, confidence_floor=floor,
            confidence_power=1.0
        )

    def test_argmax_is_argmin(self):
        assignment = compute_soft_assignment(self.distance, 1.0, 1.0)
        self.assertTrue(torch.equal(assignment.hard_from_prob, self.hard))

    def test_hard_is_one_hot(self):
        route = self.route("hard", 1)
        expected = torch.nn.functional.one_hot(self.hard, 3).float()
        self.assertTrue(torch.equal(route.final_weight, expected))

    def test_top2_has_two_normalized_entries(self):
        route = self.route("soft_top2", 2)
        self.assertTrue(torch.equal((route.routing_prob > 0).sum(1), torch.full((3,), 2)))
        self.assertTrue(torch.allclose(route.routing_prob.sum(1), torch.ones(3)))

    def test_confidence_range_and_order(self):
        probability_like_distance = -torch.log(torch.tensor([
            [0.90, 0.07, 0.03], [0.34, 0.33, 0.33]
        ]))
        assignment = compute_soft_assignment(
            probability_like_distance, 1.0, 1.0, confidence_floor=0.2
        )
        self.assertGreaterEqual(float(assignment.confidence.min()), 0.2)
        self.assertLessEqual(float(assignment.confidence.max()), 1.0)
        self.assertGreater(float(assignment.confidence[0]), float(assignment.confidence[1]))

    def test_global_normalization_is_branch_count_invariant(self):
        student = torch.zeros(3, 4, 2, 2)
        discriminators = [TinyDiscriminator() for _ in range(3)]
        for discriminator in discriminators:
            nn.init.zeros_(discriminator.layer.weight)
            nn.init.zeros_(discriminator.layer.bias)
        losses = []
        for mode, topk in (("hard", 1), ("soft_top2", 2), ("soft_dense", 3)):
            route = self.route(mode, topk)
            loss, _ = compute_weighted_generator_adv_loss(
                student, discriminators, route.final_weight
            )
            losses.append(loss)
        self.assertTrue(torch.allclose(torch.stack(losses), torch.ones(3)))

    def test_generator_and_discriminator_backward(self):
        student = torch.randn(3, 4, 2, 2, requires_grad=True)
        experts = [torch.randn_like(student) for _ in range(3)]
        discriminators = [TinyDiscriminator() for _ in range(3)]
        route = self.route("soft_top2_conf", 2)
        generator_loss, _ = compute_weighted_generator_adv_loss(
            student, discriminators, route.final_weight
        )
        generator_loss.backward(retain_graph=True)
        self.assertIsNotNone(student.grad)
        for discriminator in discriminators:
            discriminator.zero_grad(set_to_none=True)
        discriminator_loss, _ = compute_weighted_discriminator_loss(
            student.detach(), experts, discriminators, route.final_weight
        )
        discriminator_loss.backward()
        self.assertTrue(all(d.layer.weight.grad is not None for d in discriminators))


if __name__ == "__main__":
    unittest.main()
