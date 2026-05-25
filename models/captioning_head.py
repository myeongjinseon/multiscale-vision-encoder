"""
Image Captioning Head for COCO Captions.

A lightweight Transformer decoder that generates captions conditioned on
multi-scale visual features. This head evaluates whether the encoder's
global semantic representation is effective for language generation.

Architecture:
    Multi-scale features (B, N, D) → Cross-attention decoder → Caption tokens

The decoder uses causal (autoregressive) self-attention on the text side
and cross-attention to the visual features. During training, teacher
forcing is used. During inference, beam search generates the caption.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for the text decoder."""

    def __init__(self, dim: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class CaptioningHead(nn.Module):
    """
    Transformer decoder for image captioning.
    
    This is intentionally lightweight — the research focus is on the encoder,
    not the decoder. A simple 3-layer Transformer decoder is sufficient to
    evaluate the quality of the visual representation.
    
    Args:
        visual_dim: Dimension of input visual features (encoder output_dim)
        vocab_size: Vocabulary size (BERT tokenizer: 30522)
        decoder_dim: Internal decoder dimension
        num_layers: Number of Transformer decoder layers
        num_heads: Number of attention heads
        max_length: Maximum caption length
        dropout: Dropout probability
        label_smoothing: Label smoothing for cross-entropy loss
    """

    def __init__(
        self,
        visual_dim: int = 512,
        vocab_size: int = 30522,
        decoder_dim: int = 512,
        num_layers: int = 3,
        num_heads: int = 8,
        max_length: int = 50,
        dropout: float = 0.1,
        label_smoothing: float = 0.1,
        pad_token_id: int = 0,
        bos_token_id: int = 101,
        eos_token_id: int = 102,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        # Visual projection (if dims differ)
        self.visual_proj = nn.Linear(visual_dim, decoder_dim) if visual_dim != decoder_dim else nn.Identity()

        # Token embedding + positional encoding
        self.token_embedding = nn.Embedding(vocab_size, decoder_dim, padding_idx=pad_token_id)
        self.pos_encoding = PositionalEncoding(decoder_dim, max_length, dropout)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # Output head
        self.output_norm = nn.LayerNorm(decoder_dim)
        self.output_proj = nn.Linear(decoder_dim, vocab_size, bias=False)

        # Tie input and output embeddings (reduces parameters)
        self.output_proj.weight = self.token_embedding.weight

        # Loss
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=pad_token_id,
            label_smoothing=label_smoothing,
        )

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask for autoregressive decoding."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.bool()
        return mask

    def forward(
        self,
        visual_features: torch.Tensor,
        caption_ids: torch.Tensor,
        caption_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass with teacher forcing.
        
        Args:
            visual_features: (B, N_v, D) encoder output
            caption_ids: (B, L) target caption token IDs
            caption_mask: (B, L) padding mask (1 = valid, 0 = pad)
            
        Returns:
            dict with 'loss' and 'logits'
        """
        B, L = caption_ids.shape

        # Prepare visual memory
        memory = self.visual_proj(visual_features)  # (B, N_v, D_dec)

        # Prepare text input (shift right: input = [BOS, t1, t2, ...], target = [t1, t2, ..., EOS])
        input_ids = caption_ids[:, :-1]  # (B, L-1)
        target_ids = caption_ids[:, 1:]  # (B, L-1)

        # Embed tokens
        token_embeds = self.token_embedding(input_ids)
        token_embeds = self.pos_encoding(token_embeds)

        # Causal mask for autoregressive attention
        causal_mask = self._generate_causal_mask(input_ids.size(1), input_ids.device)

        # Padding mask for memory (visual features don't need masking)
        # Padding mask for target
        if caption_mask is not None:
            tgt_key_padding_mask = ~caption_mask[:, :-1].bool()
        else:
            tgt_key_padding_mask = None

        # Decode
        decoded = self.decoder(
            tgt=token_embeds,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        # Project to vocabulary
        decoded = self.output_norm(decoded)
        logits = self.output_proj(decoded)  # (B, L-1, vocab_size)

        # Compute loss
        loss = self.criterion(
            logits.reshape(-1, self.vocab_size),
            target_ids.reshape(-1),
        )

        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def generate(
        self,
        visual_features: torch.Tensor,
        beam_size: int = 5,
        max_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate captions using greedy or beam search.
        
        Args:
            visual_features: (B, N_v, D) encoder output
            beam_size: Beam width (1 = greedy)
            max_length: Maximum generation length
            
        Returns:
            generated_ids: (B, L) generated token IDs
        """
        max_length = max_length or self.max_length
        B = visual_features.shape[0]
        device = visual_features.device

        memory = self.visual_proj(visual_features)

        if beam_size == 1:
            return self._greedy_decode(memory, max_length, device)
        else:
            return self._beam_search(memory, beam_size, max_length, device)

    def _greedy_decode(
        self, memory: torch.Tensor, max_length: int, device: torch.device
    ) -> torch.Tensor:
        """Simple greedy decoding."""
        B = memory.shape[0]
        generated = torch.full((B, 1), self.bos_token_id, dtype=torch.long, device=device)

        for _ in range(max_length - 1):
            token_embeds = self.token_embedding(generated)
            token_embeds = self.pos_encoding(token_embeds)
            causal_mask = self._generate_causal_mask(generated.size(1), device)

            decoded = self.decoder(tgt=token_embeds, memory=memory, tgt_mask=causal_mask)
            decoded = self.output_norm(decoded)
            logits = self.output_proj(decoded[:, -1:])  # Last position
            next_token = logits.argmax(dim=-1)  # (B, 1)

            generated = torch.cat([generated, next_token], dim=1)

            # Stop if all sequences have EOS
            if (next_token == self.eos_token_id).all():
                break

        return generated

    def _beam_search(
        self, memory: torch.Tensor, beam_size: int, max_length: int, device: torch.device
    ) -> torch.Tensor:
        """Beam search decoding."""
        B = memory.shape[0]
        results = []

        for b in range(B):
            mem = memory[b:b+1]  # (1, N_v, D)
            mem = mem.expand(beam_size, -1, -1)  # (beam, N_v, D)

            # Initialize beams
            beams = torch.full((beam_size, 1), self.bos_token_id, dtype=torch.long, device=device)
            beam_scores = torch.zeros(beam_size, device=device)
            beam_scores[1:] = -1e9  # Only first beam is active initially

            completed = []

            for step in range(max_length - 1):
                token_embeds = self.token_embedding(beams)
                token_embeds = self.pos_encoding(token_embeds)
                causal_mask = self._generate_causal_mask(beams.size(1), device)

                decoded = self.decoder(tgt=token_embeds, memory=mem, tgt_mask=causal_mask)
                decoded = self.output_norm(decoded)
                logits = self.output_proj(decoded[:, -1])  # (beam, vocab)
                log_probs = F.log_softmax(logits, dim=-1)

                # Expand beams
                next_scores = beam_scores.unsqueeze(-1) + log_probs  # (beam, vocab)
                next_scores = next_scores.view(-1)  # (beam * vocab)

                # Select top-k
                topk_scores, topk_indices = next_scores.topk(beam_size, dim=-1)
                beam_indices = topk_indices // self.vocab_size
                token_indices = topk_indices % self.vocab_size

                # Update beams
                beams = torch.cat([
                    beams[beam_indices],
                    token_indices.unsqueeze(-1),
                ], dim=1)
                beam_scores = topk_scores

                # Check for completed beams
                eos_mask = token_indices == self.eos_token_id
                for i in range(beam_size):
                    if eos_mask[i]:
                        completed.append((beam_scores[i].item(), beams[i]))

                if len(completed) >= beam_size:
                    break

            # Select best beam
            if completed:
                completed.sort(key=lambda x: x[0], reverse=True)
                results.append(completed[0][1])
            else:
                results.append(beams[0])

        # Pad to same length
        max_len = max(r.size(0) for r in results)
        padded = torch.full((B, max_len), self.pad_token_id, dtype=torch.long, device=device)
        for i, r in enumerate(results):
            padded[i, :r.size(0)] = r

        return padded
