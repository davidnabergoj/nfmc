import torch
import torch.optim as optim

from normalizing_flows import Flow
from potentials.base import Potential


class Buffer:
    def __init__(self, event_shape, size: int):
        self.data = torch.zeros(size=(size, *event_shape))
        self.index = 0

    def add(self, x):
        assert len(x) < len(self.data)

        slots_left = len(self.data) - self.index
        if slots_left > len(x):
            # No cycling
            self.data[self.index:self.index + len(x)] = x
            self.index += len(x)
        else:
            # Cycling
            self.data[self.index:self.index + len(x)] = x[:slots_left]
            self.data[:len(x) - slots_left] = x[slots_left:]
            self.index = len(x) - slots_left


def fab_ais(x, target_log_prob):
    raise NotImplementedError


def fab(target_potential: Potential,
        flow: Flow,
        K: int,
        L: int,
        M: int,
        N: int,
        buffer_size: int = 10_000):
    assert M < buffer_size

    buffer_x = Buffer(event_shape=target_potential.event_shape, size=buffer_size)
    buffer_log_w = Buffer(event_shape=(), size=buffer_size)
    buffer_log_flow = Buffer(event_shape=(), size=buffer_size)

    optimizer = optim.AdamW(flow.parameters())

    for i in range(K):
        x, log_prob_flow = flow.sample(M, return_log_prob=True)
        x, log_w = fab_ais(x, target_log_prob=lambda v: -2 * target_potential(v) - flow.log_prob(v))

        buffer_x.add(x)
        buffer_log_w.add(log_w)
        buffer_log_flow.add(log_prob_flow)

        for j in range(L):
            optimizer.zero_grad()

            indices = torch.distributions.Categorical(logits=buffer_log_w.data).sample(sample_shape=torch.Size(N))
            # TODO: some indices may be repeated during resampling. We can avoid recomputing the flow log prob then.

            x = buffer_x.data[indices]
            log_flow_old = buffer_log_flow.data[indices]
            log_w_corr = log_flow_old - flow.log_prob(x).detach()

            buffer_log_w.data[indices] = buffer_log_w.data[indices] + log_w_corr
            buffer_log_flow.data[indices] = buffer_log_flow.data[indices] + flow.log_prob(x)

            w_corr = log_w_corr.exp()

            loss = -torch.mean(w_corr * buffer_log_flow.data[indices])
            loss.backward()
            optimizer.step()
