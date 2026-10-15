import unittest
import torch
from benchmarks.v5.phase_6_mask_decoder.continue_dense_boundary import DenseContextModel
from benchmarks.v5.phase_6_mask_decoder.multiscale_dense_decoder import MultiscaleDenseModel


class MultiscaleDenseTests(unittest.TestCase):
    def inputs(self):
        return {'deep': torch.randn(1, 256, 4, 4), 'shallow': torch.randn(1, 32, 8, 8),
                'semantic': torch.randn(1, 12, 4, 4), 'task': torch.randn(1, 256),
                'base': torch.rand(1, 32, 32)*.8+.1, 'prompt_maps': torch.rand(1, 2, 8, 8),
                'native_shallow': torch.randn(1, 256, 8, 8), 'middle': torch.randn(1, 512, 4, 4),
                'stem': torch.randn(1, 64, 16, 16)}

    def test_all_arms_reproduce_trained_coarse_at_initialization(self):
        coarse = DenseContextModel()
        torch.nn.init.normal_(coarse.head.output.weight, std=.1)
        sample = self.inputs()
        expected = coarse(sample)['pixel_probability']
        for mode in ['control', 'native', 'middle', 'full256']:
            model = MultiscaleDenseModel(coarse.state_dict(), mode)
            self.assertTrue(torch.equal(model(sample)['pixel_probability'], expected))

    def test_native_middle_and_stem_paths_receive_gradients(self):
        model = MultiscaleDenseModel(DenseContextModel().state_dict(), 'full256')
        torch.nn.init.normal_(model.fusion.output128.weight, std=.1)
        torch.nn.init.normal_(model.fusion.output256.weight, std=.1)
        model(self.inputs())['pixel_logits'].square().mean().backward()
        for projection in [model.fusion.native, model.fusion.middle, model.fusion.stem]:
            self.assertGreater(float(projection[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.coarse.head.output.weight.grad.abs().sum()), 0)

    def test_gt_fields_do_not_change_trained_inference(self):
        model = MultiscaleDenseModel(DenseContextModel().state_dict(), 'full256').eval()
        torch.nn.init.normal_(model.fusion.output128.weight, std=.1)
        torch.nn.init.normal_(model.fusion.output256.weight, std=.1)
        sample = self.inputs()
        expected = model(sample)['pixel_probability']
        altered = {**sample, 'truth': torch.zeros(1, 32, 32, dtype=torch.bool),
                   'valid': torch.zeros(1, 32, 32, dtype=torch.bool), 'target_class': torch.tensor([11])}
        self.assertTrue(torch.equal(expected, model(altered)['pixel_probability']))

    def test_progressive_arms_share_seeded_common_initialization(self):
        state = DenseContextModel().state_dict()
        torch.manual_seed(17)
        first = MultiscaleDenseModel(state, 'native')
        torch.manual_seed(17)
        last = MultiscaleDenseModel(state, 'full256')
        for key, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, last.state_dict()[key]), key)


if __name__ == '__main__':
    unittest.main()
