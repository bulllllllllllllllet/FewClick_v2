import copy
import unittest
import torch
from benchmarks.v5.phase_6_mask_decoder.run_segnext import validate_resume, training_state, restore_training_state


class SegNextContinuationTests(unittest.TestCase):
    def test_legacy_weights_require_explicit_optimizer_reset(self):
        payload={'run':{'cache':'/tmp/segnext_test_cache'},'epoch':7}
        self.assertEqual(validate_resume(payload,'/tmp/segnext_test_cache','reset'),8)
        with self.assertRaises(ValueError):validate_resume(payload,'/tmp/segnext_test_cache','restore')
        with self.assertRaises(ValueError):validate_resume(payload,'/tmp/segnext_test_cache',None)

    def test_wrong_feature_cache_is_rejected(self):
        payload={'run':{'cache':'/tmp/encoder_cache_A'},'epoch':7}
        with self.assertRaises(ValueError):validate_resume(payload,'/tmp/encoder_cache_B','reset')

    def test_optimizer_and_dropout_rng_restore_matches_uninterrupted_updates(self):
        torch.manual_seed(103)
        model=torch.nn.Sequential(torch.nn.Linear(4,8),torch.nn.Dropout(.3),torch.nn.Linear(8,1))
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
        x=torch.randn(3,4);y=torch.randn(3,1)
        def update(network,opt):
            opt.zero_grad();loss=(network(x)-y).square().mean();loss.backward();opt.step()
        for _ in range(3):update(model,optimizer)
        payload=copy.deepcopy({'model':model.state_dict(),**training_state(optimizer,'cpu')})
        for _ in range(2):update(model,optimizer)
        uninterrupted=copy.deepcopy(model.state_dict())
        restarted=torch.nn.Sequential(torch.nn.Linear(4,8),torch.nn.Dropout(.3),torch.nn.Linear(8,1))
        restarted.load_state_dict(payload['model'])
        restored_optimizer=torch.optim.AdamW(restarted.parameters(),lr=1e-3)
        restore_training_state(restored_optimizer,payload,'cpu')
        for _ in range(2):update(restarted,restored_optimizer)
        for key,value in restarted.state_dict().items():
            torch.testing.assert_close(value,uninterrupted[key],atol=0,rtol=0)


if __name__=='__main__':unittest.main()
