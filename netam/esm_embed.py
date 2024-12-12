import torch
from esm import pretrained


class ESMEmbedder:
    def __init__(self, model_name: str, max_aa_seq_len: int, device: str = "cpu"):
        """
        Initializes the ESMEmbedder object.

        Args:
            model_name (str): Name of the pretrained ESM model (e.g., "esm2_t6_8M_UR50D").
            max_aa_seq_len (int): Maximum sequence length allowed.
            device (str): Device to run the model on.
        """
        self.device = device
        self.max_aa_seq_len = max_aa_seq_len
        self.model, self.alphabet = pretrained.load_model_and_alphabet(model_name)
        self.model = self.model.to(device)
        self.batch_converter = self.alphabet.get_batch_converter()

    @property
    def num_heads(self):
        return self.model.layers[0].self_attn.num_heads

    @property
    def d_model(self):
        return self.model.layers[0].self_attn.hidden_size

    @property
    def d_model_per_head(self):
        return self.d_model // self.num_heads

    @property
    def num_layers(self):
        return self.model.num_layers

    def embed_batch(self, sequences: list[str]) -> torch.Tensor:
        """
        Embeds a batch of sequences.

        Args:
            sequences (list[str]): List of amino acid sequences.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, max_aa_seq_len, embedding_dim).

        Raises:
            ValueError: If any sequence exceeds max_aa_seq_len.
        """
        for seq in sequences:
            if len(seq) > self.max_aa_seq_len:
                raise ValueError(
                    f"Sequence length {len(seq)} exceeds max_aa_seq_len {self.max_aa_seq_len}"
                )

        named_sequences = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(named_sequences)
        batch_tokens = batch_tokens.to(self.device)
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[self.num_layers])
        embeddings = results["representations"][self.num_layers]

        batch_size, seq_len, embedding_dim = embeddings.size()
        padded_embeddings = torch.zeros(
            (batch_size, self.max_aa_seq_len, embedding_dim), device=self.device
        )
        padded_embeddings[:, :seq_len, :] = embeddings

        return padded_embeddings
