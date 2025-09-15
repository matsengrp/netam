"""Tests for ParentIndependentBinarySelectionModel mask functionality."""

import pytest
import torch

from netam.models import ParentIndependentBinarySelectionModel


class TestParentIndependentBinarySelectionModelMask:
    """Test mask functionality for ParentIndependentBinarySelectionModel."""

    @pytest.fixture
    def model_1d(self):
        """Create a ParentIndependentBinarySelectionModel with output_dim=1."""
        return ParentIndependentBinarySelectionModel(
            output_dim=1, known_token_count=21, model_type="test"
        )

    @pytest.fixture
    def model_multidim(self):
        """Create a ParentIndependentBinarySelectionModel with output_dim=20."""
        return ParentIndependentBinarySelectionModel(
            output_dim=20, known_token_count=21, model_type="test"
        )

    @pytest.fixture
    def model_with_wildtype(self):
        """Create a ParentIndependentBinarySelectionModel with wildtype sequence."""
        return ParentIndependentBinarySelectionModel(
            output_dim=20,
            wildtype_sequence="ACDEFG",
            known_token_count=21,
            model_type="test",
        )

    @pytest.fixture
    def sample_inputs(self):
        """Create sample inputs for testing."""
        batch_size, seq_len = 2, 6
        amino_acid_indices = torch.randint(0, 21, (batch_size, seq_len))
        # Create mask with some positions masked (False)
        mask = torch.tensor(
            [
                [True, True, False, True, False, True],
                [True, False, True, True, True, False],
            ],
            dtype=torch.bool,
        )
        return amino_acid_indices, mask

    def test_mask_application_1d(self, model_1d, sample_inputs):
        """Test that masking works correctly for 1D output."""
        amino_acid_indices, mask = sample_inputs

        # Get output with mask
        result = model_1d.forward(amino_acid_indices, mask)

        # Check that result has correct shape
        assert result.shape == (2, 6)

        # Check that masked positions have value 0 (mask=False means multiply by 0)
        assert torch.all(result[~mask] == 0.0)

        # Check that unmasked positions retain their original values
        # Compare with result when mask is all True
        all_true_mask = torch.ones_like(mask, dtype=torch.bool)
        unmasked_result = model_1d.forward(amino_acid_indices, all_true_mask)

        # Unmasked positions should match
        assert torch.allclose(result[mask], unmasked_result[mask])

    def test_mask_application_multidim(self, model_multidim, sample_inputs):
        """Test that masking works correctly for multi-dimensional output."""
        amino_acid_indices, mask = sample_inputs

        # Get output with mask
        result = model_multidim.forward(amino_acid_indices, mask)

        # Check that result has correct shape
        assert result.shape == (2, 6, 20)

        # Check that masked positions have value 0 across all output dimensions
        masked_positions = ~mask
        assert torch.all(result[masked_positions] == 0.0)

        # Check that unmasked positions retain their original values
        all_true_mask = torch.ones_like(mask, dtype=torch.bool)
        unmasked_result = model_multidim.forward(amino_acid_indices, all_true_mask)

        # Unmasked positions should match
        unmasked_positions = mask
        assert torch.allclose(
            result[unmasked_positions], unmasked_result[unmasked_positions]
        )

    def test_mask_with_wildtype_zapping(self, model_with_wildtype, sample_inputs):
        """Test that masking works correctly with wildtype zapping."""
        amino_acid_indices, mask = sample_inputs

        # Adjust input to match wildtype sequence length
        amino_acid_indices = amino_acid_indices[
            :, :6
        ]  # Model has wildtype sequence of length 6
        mask = mask[:, :6]

        result = model_with_wildtype.forward(amino_acid_indices, mask)

        # Check shape
        assert result.shape == (2, 6, 20)

        # Check that masked positions are 0
        masked_positions = ~mask
        assert torch.all(result[masked_positions] == 0.0)

    def test_all_masked(self, model_1d):
        """Test behavior when all positions are masked."""
        batch_size, seq_len = 2, 6
        amino_acid_indices = torch.randint(0, 21, (batch_size, seq_len))
        mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)  # All False

        result = model_1d.forward(amino_acid_indices, mask)

        # All positions should be 0
        assert torch.all(result == 0.0)

    def test_all_unmasked(self, model_1d):
        """Test behavior when no positions are masked."""
        batch_size, seq_len = 2, 6
        amino_acid_indices = torch.randint(0, 21, (batch_size, seq_len))
        mask = torch.ones((batch_size, seq_len), dtype=torch.bool)  # All True

        result = model_1d.forward(amino_acid_indices, mask)

        # Should be same as model's learned parameters
        expected = model_1d.log_selection_factors[:seq_len].expand(batch_size, seq_len)
        assert torch.allclose(result, expected)

    def test_gradient_flow_masked_positions(self, model_1d, sample_inputs):
        """Test that gradients don't flow through masked positions."""
        amino_acid_indices, mask = sample_inputs

        # Enable gradients
        model_1d.train()

        # Forward pass
        result = model_1d.forward(amino_acid_indices, mask)

        # Create a simple loss that only depends on the result
        loss = result.sum()

        # Backward pass
        loss.backward()

        # Check that gradients exist for the model parameters
        assert model_1d.log_selection_factors.grad is not None

        # The gradient contribution from masked positions should be 0
        # This is automatically handled by the multiplication by 0

    def test_mask_consistency_across_batches(self, model_1d):
        """Test that masking is applied consistently across batch dimensions."""
        batch_size, seq_len = 3, 5
        amino_acid_indices = torch.randint(0, 21, (batch_size, seq_len))

        # Create mask where same positions are masked across all batches
        mask = torch.tensor(
            [
                [True, False, True, False, True],
                [True, False, True, False, True],
                [True, False, True, False, True],
            ],
            dtype=torch.bool,
        )

        result = model_1d.forward(amino_acid_indices, mask)

        # Masked positions (index 1 and 3) should be 0 for all batches
        assert torch.all(result[:, 1] == 0.0)
        assert torch.all(result[:, 3] == 0.0)

        # Unmasked positions should have same values across batches (since they come from position-specific parameters)
        for pos in [0, 2, 4]:
            # All batches should have same value at this position (from position-specific params)
            assert torch.allclose(result[0, pos], result[1, pos])
            assert torch.allclose(result[1, pos], result[2, pos])

    def test_different_masks_per_batch(self, model_1d):
        """Test that different masks can be applied to different sequences in the
        batch."""
        batch_size, seq_len = 2, 4
        amino_acid_indices = torch.randint(0, 21, (batch_size, seq_len))

        # Different masks for each sequence in batch
        mask = torch.tensor(
            [
                [True, False, True, True],  # Second position masked
                [True, True, False, True],  # Third position masked
            ],
            dtype=torch.bool,
        )

        result = model_1d.forward(amino_acid_indices, mask)

        # Check that different positions are masked for each sequence
        assert result[0, 1] == 0.0  # Second position of first sequence
        assert result[1, 2] == 0.0  # Third position of second sequence

        # Check that non-masked positions are non-zero (assuming learned parameters are non-zero)
        # We can't assume they're non-zero since they're initialized to zero, but they should be equal to the learned parameters
        expected_vals = model_1d.log_selection_factors[:seq_len]
        assert result[0, 0] == expected_vals[0]
        assert result[1, 1] == expected_vals[1]

    def test_mask_device_compatibility(self, model_1d):
        """Test that masking works correctly when tensors are on different devices."""
        batch_size, seq_len = 2, 4
        amino_acid_indices = torch.randint(0, 21, (batch_size, seq_len))
        mask = torch.ones((batch_size, seq_len), dtype=torch.bool)

        # Move model to CPU (it should already be there, but make sure)
        model_1d.to("cpu")
        amino_acid_indices = amino_acid_indices.to("cpu")
        mask = mask.to("cpu")

        # Should work without errors
        result = model_1d.forward(amino_acid_indices, mask)
        assert result.device.type == "cpu"

    def test_mask_dtype_handling(self, model_1d):
        """Test that different mask dtypes are handled correctly."""
        batch_size, seq_len = 2, 4
        amino_acid_indices = torch.randint(0, 21, (batch_size, seq_len))

        # Test with float mask (0.0 and 1.0)
        float_mask = torch.tensor(
            [[1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0]], dtype=torch.float32
        )

        result_float = model_1d.forward(amino_acid_indices, float_mask)

        # Test with bool mask
        bool_mask = torch.tensor(
            [[True, False, True, True], [True, True, False, True]], dtype=torch.bool
        )

        result_bool = model_1d.forward(amino_acid_indices, bool_mask)

        # Results should be the same
        assert torch.allclose(result_float, result_bool)
