import torch
import pytest

from vq.quant.grid import quantize_rtn, NF4_CODEBOOK, quantize_nf4
from vq.quant.gptq import gptq_quantize, build_hessian
from vq.quant.awq import awq_quantize

torch.manual_seed(0)


def _unique_per_group(w, group):
    out, cols = [], w.shape[1]
    for g in range(0, cols, group):
        for row in w[:, g : g + group]:
            out.append(len(torch.unique(row)))
    return max(out)


class TestRTN:
    def test_values_on_grid_per_group(self):
        w = torch.randn(16, 256)
        q = quantize_rtn(w, bits=4, group_size=128)
        assert _unique_per_group(q, 128) <= 16

    def test_error_bounded_by_half_step(self):
        w = torch.randn(8, 128)
        q = quantize_rtn(w, bits=4, group_size=128)
        step = (2 * w.abs().amax(dim=1, keepdim=True)) / 15
        # scales rounded to fp16 (rel. error <= 2^-11) and the extreme level is
        # 7.5 steps from zero -> slack of 15 * 2^-11 relative to the half step
        assert torch.all((w - q).abs() <= step / 2 * (1 + 15 * 2**-11) + 1e-6)

    def test_8bit_nearly_lossless(self):
        w = torch.randn(32, 64)
        q = quantize_rtn(w, bits=8, group_size=None)
        assert (w - q).abs().max() < 0.02 * w.abs().max()

    def test_zero_row_stays_zero(self):
        w = torch.zeros(2, 128)
        assert torch.equal(quantize_rtn(w, 4, 128), w)

    def test_preserves_dtype_and_shape(self):
        w = torch.randn(4, 128, dtype=torch.bfloat16)
        q = quantize_rtn(w, 4, 128)
        assert q.dtype == torch.bfloat16 and q.shape == w.shape


class TestNF4:
    def test_codebook_matches_bitsandbytes(self):
        assert len(NF4_CODEBOOK) == 16
        assert NF4_CODEBOOK[0] == -1.0 and NF4_CODEBOOK[-1] == 1.0
        assert 0.0 in NF4_CODEBOOK.tolist()
        assert NF4_CODEBOOK[7] == 0.0  # bnb NF4: 7 negative, zero, 8 positive levels
        assert NF4_CODEBOOK[8] == pytest.approx(0.07958029955625534)

    def test_values_are_scaled_codebook(self):
        w = torch.randn(4, 128)
        q = quantize_nf4(w, block_size=64)
        blocks_w = w.reshape(-1, 64)
        blocks_q = q.reshape(-1, 64)
        for bw, bq in zip(blocks_w, blocks_q):
            absmax = bw.abs().max()
            ratios = bq / absmax
            dist = (ratios[:, None] - NF4_CODEBOOK[None, :]).abs().min(dim=1).values
            assert dist.max() < 1e-5

    def test_nearest_codebook_entry(self):
        w = torch.tensor([[1.0, 0.0, -1.0, 0.5] + [0.0] * 60])
        q = quantize_nf4(w, block_size=64)
        assert q[0, 0] == 1.0 and q[0, 2] == -1.0 and q[0, 1] == 0.0


class TestGPTQ:
    def _problem(self, n=512, d_in=128, d_out=32, outlier=True):
        x = torch.randn(n, d_in)
        if outlier:
            x[:, :4] *= 20.0
        x = x @ torch.randn(d_in, d_in) * 0.3 + x  # correlated features
        w = torch.randn(d_out, d_in)
        return x, w

    def test_identity_hessian_equals_rtn(self):
        w = torch.randn(16, 256)
        h = torch.eye(256)
        q = gptq_quantize(w, h, bits=4, group_size=128, damp=0.0)
        torch.testing.assert_close(q, quantize_rtn(w, 4, 128))

    def test_beats_rtn_on_output_error(self):
        x, w = self._problem()
        h = build_hessian(x)
        q_gptq = gptq_quantize(w, h, bits=4, group_size=128)
        q_rtn = quantize_rtn(w, 4, 128)
        err = lambda q: ((x @ (w - q).T) ** 2).mean()
        assert err(q_gptq) < 0.8 * err(q_rtn)

    def test_values_on_grid(self):
        x, w = self._problem()
        q = gptq_quantize(w, build_hessian(x), bits=4, group_size=128)
        assert _unique_per_group(q, 128) <= 16

    def test_3bit_grid(self):
        x, w = self._problem()
        q = gptq_quantize(w, build_hessian(x), bits=3, group_size=128)
        assert _unique_per_group(q, 128) <= 8

    def test_dead_inputs_handled(self):
        x, w = self._problem()
        x[:, 5] = 0.0
        q = gptq_quantize(w, build_hessian(x), bits=4, group_size=128)
        assert torch.isfinite(q).all()


class TestContrastiveHessian:
    def test_lambda_zero_is_plain(self):
        x = torch.randn(100, 32)
        d = torch.randn(10, 32)
        torch.testing.assert_close(build_hessian(x, d, lam=0.0), build_hessian(x))

    def test_normalised_by_counts(self):
        x = torch.randn(100, 8)
        torch.testing.assert_close(build_hessian(x), 2.0 * x.T @ x / 100)

    def test_contrastive_term_reduces_error_on_difference_directions(self):
        torch.manual_seed(1)
        n, d_in, d_out = 1024, 128, 32
        x = torch.randn(n, d_in)
        # decision directions live in a subspace poorly covered by generic data
        basis = torch.linalg.qr(torch.randn(d_in, 4)).Q
        x = x - (x @ basis) @ basis.T * 0.95
        diffs = torch.randn(64, 4) @ basis.T
        w = torch.randn(d_out, d_in)
        err_d = lambda q: ((diffs @ (w - q).T) ** 2).mean()
        q0 = gptq_quantize(w, build_hessian(x), bits=4, group_size=128)
        q1 = gptq_quantize(w, build_hessian(x, diffs, lam=1.0), bits=4, group_size=128)
        assert err_d(q1) < err_d(q0)


class TestAWQ:
    def test_beats_rtn_with_activation_outliers(self):
        torch.manual_seed(2)
        x = torch.randn(512, 128)
        x[:, :4] *= 30.0
        w = torch.randn(32, 128)
        q_awq = awq_quantize(w, x, bits=4, group_size=128)
        q_rtn = quantize_rtn(w, 4, 128)
        err = lambda q: ((x @ (w - q).T) ** 2).mean()
        assert err(q_awq) <= err(q_rtn)

    def test_alpha_zero_candidate_means_never_worse_than_rtn(self):
        x = torch.randn(256, 128)
        w = torch.randn(16, 128)
        err = lambda q: ((x @ (w - q).T) ** 2).mean()
        assert err(awq_quantize(w, x, 4, 128)) <= err(quantize_rtn(w, 4, 128)) + 1e-6
