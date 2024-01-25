import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# Attention with mask and RoPE
class MultiheadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiheadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(embed_size, self.head_dim * heads, bias=False)
        self.keys = nn.Linear(embed_size, self.head_dim * heads, bias=False)
        self.queries = nn.Linear(embed_size, self.head_dim * heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def sinusoidal_position_embedding(self, batch_size, nums_head, max_len, output_dim, device):
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(-1)
        ids = torch.arange(0, output_dim // 2, dtype=torch.float)
        theta = torch.pow(10000, -2 * ids / output_dim)

        embeddings = position * theta

        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        embeddings = embeddings.repeat((batch_size, nums_head, *([1] * len(embeddings.shape))))

        embeddings = torch.reshape(embeddings, (batch_size, nums_head, max_len, output_dim))
        embeddings = embeddings.to(device)
        return embeddings

    def RoPE(self, q, k):
        # q,k: (B, H, L, D)
        batch_size = q.shape[0]
        nums_head = q.shape[1]
        max_len = q.shape[2]
        output_dim = q.shape[-1]

        pos_emb = self.sinusoidal_position_embedding(batch_size, nums_head, max_len, output_dim, q.device)

        cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
        sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)

        # q,k: (B, H, L, D)
        q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1)
        q2 = q2.reshape(q.shape)
        q = q * cos_pos + q2 * sin_pos

        k2 = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1)
        k2 = k2.reshape(k.shape)
        k = k * cos_pos + k2 * sin_pos

        return q, k

    def forward(self, x, mask, use_rope=True):
        B = x.shape[0]

        len = x.shape[1]

        values = self.values(x).view(B, len, self.heads, self.head_dim)
        keys = self.keys(x).view(B, len, self.heads, self.head_dim)
        queries = self.queries(x).view(B, len, self.heads, self.head_dim)

        values = values.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        queries = queries.permute(0, 2, 1, 3)
        # [B, H, L, D]

        if use_rope:
            queries, keys = self.RoPE(queries, keys)

        energy = torch.matmul(queries, keys.permute(0, 1, 3, 2))

        if mask is not None:
            energy = energy.masked_fill(mask == 1, float("-1e20"))

        attention = F.softmax(energy / (self.head_dim ** (1 / 2)), dim=-1)

        out = torch.matmul(attention, values)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, len, self.heads * self.head_dim)

        out = self.fc_out(out)
        return out

    def predict(self, x, mask, k_cache, v_cache, use_rope=True):
        B = x.shape[0]
        len = x.shape[1]

        # prompting process
        values = self.values(x).view(B, len, self.heads, self.head_dim)
        keys = self.keys(x).view(B, len, self.heads, self.head_dim)
        # [B, L/1, H, D]

        queries = self.queries(x).view(B, len, self.heads, self.head_dim)
        # [B, L/1, H, D]

        values = values.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        queries = queries.permute(0, 2, 1, 3)
        # [B, H, L/1, D]

        if k_cache is not None:
            keys = torch.cat([k_cache, keys], dim=2)
            values = torch.cat([v_cache, values], dim=2)
            # [B, H, L+1, D]

        if use_rope:
            queries, keys = self.RoPE(queries, keys)

        energy = torch.matmul(queries, keys.permute(0, 1, 3, 2))

        if mask is not None:
            energy = energy.masked_fill(mask == 1, float("-1e20"))

        attention = F.softmax(energy / (self.head_dim ** (1 / 2)), dim=-1)

        out = torch.matmul(attention, values)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, len, self.heads * self.head_dim)

        out = self.fc_out(out)
        return out, keys, values


# SwiGLU
class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_hidden_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, ff_hidden_dim)
        self.fc2 = nn.Linear(embed_size, ff_hidden_dim)
        self.fc3 = nn.Linear(ff_hidden_dim, embed_size)

    def forward(self, x):
        gate = torch.sigmoid(self.fc1(x))
        transformed = torch.relu(self.fc2(x))
        output = gate * transformed
        return self.fc3(output)

# class FeedForward(nn.Module):
#     def __init__(self, embed_size, ff_hidden_dim):
#         super(FeedForward, self).__init__()
#         self.fc1 = nn.Linear(embed_size, ff_hidden_dim)
#         self.fc2 = nn.Linear(ff_hidden_dim, embed_size)
#
#     def forward(self, x):
#         return self.fc2(torch.relu(self.fc1(x)))

class DecoderLayer(nn.Module):
    def __init__(self, embed_size, heads, ff_hidden_dim, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.attention = MultiheadAttention(embed_size, heads)
        self.feed_forward = FeedForward(embed_size, ff_hidden_dim)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, use_rope=True):
        attention = self.attention(self.norm1(x), mask, use_rope=use_rope)

        x = self.dropout(attention) + x
        forward = self.feed_forward(self.norm2(x))

        out = self.dropout(forward) + x
        return out

    def predict(self, x, mask, kcache, vcache, use_rope=True):
        attention, kcache, vcache = self.attention.predict(x, mask, kcache, vcache, use_rope=use_rope)

        x = self.dropout(self.norm1(attention)) + x
        forward = self.feed_forward(x)

        out = self.dropout(self.norm2(forward)) + x
        return out, kcache, vcache

class TransformerDecoderPredictor(nn.Module):
    def __init__(self, embed_size, heads, ff_hidden_dim, num_layers, dropout=0.):
        super(TransformerDecoderPredictor, self).__init__()
        self.embedding = nn.Linear(1, embed_size)
        self.embed_size = embed_size
        self.decoder_blocks = nn.ModuleList(
            [DecoderLayer(embed_size, heads, ff_hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embed_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, use_rope=True):
        x = x.unsqueeze(2)
        x = self.embedding(x)
        for decoder in self.decoder_blocks:
            x = decoder(x, mask, use_rope=use_rope)
        x = self.fc_out(x)
        x = x.squeeze(2)
        return x

    def predict(self, prompt, predict_length, use_rope=True):
        """
        This function was originally intended to predict using kvcache
        In the current code, due to the presence of RoPE (Rotary Positional Encoding),
        the positional encoding after using KV_cache differs from the step-by-step prediction.
        While referencing llamas' code (https://github.com/facebookresearch/llama/blob/main/llama/model.py#280),
        I did not identify a solution to address this issue.
        Therefore, I do not recommend use this function.
        By the way, if you can solve this problem, please let me know. (github:@liaoyanqing666, email:1793706453@qq.com)

        :param prompt: prompt data
        :param predict_length: length of prediction
        :param use_rope: whether to use RoPE
        :return: prediction
        """
        with torch.no_grad():
            k_cache = [None for _ in range(len(self.decoder_blocks))]
            v_cache = [None for _ in range(len(self.decoder_blocks))]

            # prompt process in kv_cache
            mask = torch.triu(torch.ones(prompt.shape[1], prompt.shape[1]), diagonal=1).to(prompt.device)
            x = prompt.unsqueeze(2)
            # [B, L, 1]
            x = self.embedding(x)
            # [B, L, D]
            for i, decoder in enumerate(self.decoder_blocks):
                x, k_cache[i], v_cache[i] = decoder.predict(x, mask, k_cache[i], v_cache[i], use_rope=use_rope)
            # [B, L, D]
            x = self.fc_out(x[:, -1:, :])
            # [B, 1, 1]
            output = x

            # decode process
            for i in range(predict_length-1):
                x = self.embedding(x)
                # [B, 1, D]
                for j, decoder in enumerate(self.decoder_blocks):
                    x, k_cache[j], v_cache[j] = decoder.predict(x, None, k_cache[j], v_cache[j], use_rope=use_rope)
                # [B, 1, D]
                x = self.fc_out(x)
                # [B, 1, 1]
                output = torch.cat([output, x], dim=1)

            output = output.squeeze(2)

            return output


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, output_dim)
        self.activate = nn.LeakyReLU()

    def forward(self, x):
        return self.fc3(self.activate(self.fc2(self.activate(self.fc1(x)))))


if __name__ == '__main__':
    embed_size = 64
    heads = 4
    ff_hidden_dim = 256
    num_layers = 3

    transformer_decoder = TransformerDecoderPredictor(embed_size, heads, ff_hidden_dim, num_layers)
    input_data = torch.randn((3, 10)) # B, L

    outputs = input_data
    for j in range(20):
        mask = torch.triu(torch.ones(outputs.shape[1], outputs.shape[1]), diagonal=1)
        test_outputs = transformer_decoder(outputs, mask, use_rope=True)
        outputs = torch.cat((outputs, test_outputs[:, -1:]), dim=1)
    print(outputs.shape)
