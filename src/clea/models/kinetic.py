"""
Kinetic (motion sequence) representation models.

All models accept (B, T, 3) tensors representing joint-velocity trajectories.
"""

import torch
import torch.nn as nn


class RawSequenceEncoder(nn.Module):
    """Bidirectional GRU encoder mapping a motion trajectory to a fixed-size embedding."""

    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2, device='cpu'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device

        self.gru = nn.GRU(
            input_size, hidden_size // (num_layers * 2),
            num_layers, batch_first=True, dropout=dropout, bidirectional=True,
        )

    def forward(self, x):
        self.init_hidden(x.shape[0], device=x.device)
        output, final_hidden = self.gru(x, self.h0)
        return final_hidden.transpose(0, 1).reshape(-1, self.hidden_size)

    def forward_for_seq2seq(self, x):
        self.init_hidden(x.shape[0], device=x.device)
        return self.gru(x, self.h0)

    def init_hidden(self, batch_size, device=None):
        # Derive the hidden-state device from the input tensor (rather than the
        # possibly-stale self.device) so encode()/forward() still work after the
        # model is moved with .to(device) post-construction (e.g. trained on one
        # device, then used to generate embeddings on another).
        self.h0 = torch.randn(
            self.num_layers * 2, batch_size,
            self.hidden_size // (self.num_layers * 2),
            device=device if device is not None else self.device,
        )

    def encode(self, x):
        return self.forward(x)


class RawSequenceDecoder(nn.Module):
    """GRU-based sequence decoder."""

    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2, device='cpu'):
        super().__init__()
        self.device = device
        self.gru = nn.GRU(
            input_size, hidden_size // (num_layers * 2),
            num_layers, batch_first=True, dropout=dropout, bidirectional=True,
        )
        self.linear = nn.Linear(hidden_size // 2, input_size)

    def forward(self, x, hidden):
        output, hidden = self.gru(x, hidden.contiguous())
        return self.linear(output), hidden


class Seq2Seq(nn.Module):
    """
    Sequence-to-sequence autoencoder for motion trajectories.
    Encodes a trajectory with a GRU encoder and reconstructs it autoregressively.
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2, device='cpu'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device

        self.encoder = RawSequenceEncoder(input_size, hidden_size, num_layers, dropout, device)
        self.decoder = RawSequenceDecoder(input_size, hidden_size, num_layers, dropout, device)

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        self.encoder.init_hidden(batch_size)
        _, hidden = self.encoder.forward_for_seq2seq(x)

        inp = torch.zeros((batch_size, 1, self.input_size), device=x.device)
        output_seq = []
        for _ in range(seq_len):
            out, hidden = self.decoder(inp, hidden.contiguous())
            output_seq.append(out)
            inp = out
        return torch.cat(output_seq, dim=1)

    def encode(self, x):
        self.batch_size = x.size(0)
        self.seq_len = x.size(1)
        return self.encoder(x)

    def decode(self, x):
        inp = torch.zeros((self.batch_size, 1, self.input_size), device=x.device)
        hidden = x.reshape([self.batch_size, self.num_layers * 2, -1]).transpose(1, 0)
        output_seq = []
        for _ in range(self.seq_len):
            out, hidden = self.decoder(inp, hidden.contiguous())
            output_seq.append(out)
            inp = out
        return torch.cat(output_seq, dim=1)


class Seq2SeqVAE(nn.Module):
    """
    Variational sequence-to-sequence autoencoder for motion trajectories.

    The encoder outputs 2*hidden_size values: the first half are means, the
    second half are log-standard-deviations.
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2, beta=1.0, device='cpu'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
        self.nz = hidden_size
        self.beta = beta

        self.encoder = RawSequenceEncoder(input_size, 2 * hidden_size, num_layers, dropout, device)
        self.decoder = RawSequenceDecoder(input_size, hidden_size, num_layers, dropout, device)

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        self.encoder.init_hidden(batch_size)
        q = self.encoder(x)
        hidden = (q[:, :self.nz]
                  + torch.exp(q[:, self.nz:]) * torch.randn([q.shape[0], self.nz], device=self.device))
        hidden = hidden.reshape(2 * self.num_layers, batch_size, self.hidden_size // (2 * self.num_layers))

        inp = torch.zeros((batch_size, 1, self.input_size), device=x.device)
        output_seq = []
        for _ in range(seq_len):
            out, hidden = self.decoder(inp, hidden.contiguous())
            output_seq.append(out)
            inp = out
        return {'q': q, 'rec': torch.cat(output_seq, dim=1)}

    def encode(self, x):
        return self.encoder(x)[:, :self.nz]

    def kl_divergence(self, mu1, log_sigma1, mu2, log_sigma2):
        """KL[p||q] between two Gaussians parameterised by (mu, log_sigma)."""
        return (log_sigma2 - log_sigma1) + (
            torch.exp(log_sigma1) ** 2 + (mu1 - mu2) ** 2
        ) / (2 * torch.exp(log_sigma2) ** 2) - 0.5

    def vae_loss(self, a_output, p_output, n_output, a_label, p_label, n_label):
        a_q, a_recon = a_output['q'], a_output['rec']
        p_q, p_recon = p_output['q'], p_output['rec']
        n_q, n_recon = n_output['q'], n_output['rec']

        rec_loss = (nn.MSELoss()(a_recon, a_label)
                    + nn.MSELoss()(p_recon, p_label)
                    + nn.MSELoss()(n_recon, n_label))

        a_m, a_dev = a_q[:, :self.nz], a_q[:, self.nz:]
        p_m, p_dev = p_q[:, :self.nz], p_q[:, self.nz:]
        n_m, n_dev = n_q[:, :self.nz], n_q[:, self.nz:]
        zeros_m = torch.zeros((a_q.shape[0], self.nz), device=self.device)
        zeros_dev = torch.zeros((a_q.shape[0], self.nz), device=self.device)

        kl_loss = (self.kl_divergence(a_m, a_dev, zeros_m, zeros_dev).mean()
                   + self.kl_divergence(p_m, p_dev, zeros_m, zeros_dev).mean()
                   + self.kl_divergence(n_m, n_dev, zeros_m, zeros_dev).mean())

        return rec_loss + self.beta * kl_loss
