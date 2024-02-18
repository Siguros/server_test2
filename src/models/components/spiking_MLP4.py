import torch
from torch import nn


class sMLP4(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 1000,
        lin2_size: int = 1000,
        lin3_size: int = 1000,
        output_size: int = 10,
    ) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.scale = scale
        self.subthresh = subthresh

        self.pPTRACE0 = pPTRACE_Node(
            parametric_tau=init_parametric_tau,
            version=init_version,
            surrogate_function1=surrogate.SpkTraceBinaryModifiedCall,
            surrogate_function2=surrogate.SpkTraceModifiedCall,
            surrogate_function3=surrogate.SpkTraceSwitchCall,
        )
        self.fc1 = nn.Linear(784, scale, bias=False)
        self.pPLIF1 = pPLIF_Node(
            surrogate_function=surrogate.HeavisideBoxcarCall(
                thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True
            )
        )

        self.pPTRACE1 = pPTRACE_Node(
            parametric_tau=init_parametric_tau,
            version=init_version,
            surrogate_function1=surrogate.SpkTraceBinaryModifiedCall,
            surrogate_function2=surrogate.SpkTraceModifiedCall,
            surrogate_function3=surrogate.SpkTraceSwitchCall,
        )
        self.fc2 = nn.Linear(scale, 100, bias=False)
        self.pPLIF2 = pPLIF_Node(
            surrogate_function=surrogate.HeavisideBoxcarCall(
                thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True
            )
        )

        self.boost1 = nn.AvgPool1d(10, 10)

        self.pPLI = pPLI_Node(
            decay_acc=init_decay_acc, surrogate_function=surrogate.AccAlwaysGradCall()
        )

        self.tau_0 = nn.Parameter(torch.ones(1, dtype=torch.float) * init_tau)
        # self.tau_0 = nn.Parameter(torch.tensor([1.9870]))
        self.tau_0.to("cuda" if torch.cuda.is_available() else "cpu")

        self.tau_vector = nn.Parameter(torch.ones(1, dtype=torch.float) * init_tau)
        # self.tau_vector = nn.Parameter(torch.tensor([1.6541, 0.9080, 0.9243, 0.6587, 1.0764, 0.9103, 0.9568, 0.9435, 1.0807]))
        self.tau_vector.to("cuda" if torch.cuda.is_available() else "cpu")

        self.spk_trace_tau_vector = nn.Parameter(
            torch.ones(2, dtype=torch.float) * init_spk_trace_tau
        )  # 1.0
        # self.spk_trace_tau_vector = nn.Parameter(torch.tensor([1.6541, 0.9080, 0.9243, 0.6587, 1.0764, 0.9103, 0.9568, 0.9435, 1.0807]))
        self.spk_trace_tau_vector.to("cuda" if torch.cuda.is_available() else "cpu")

        self.spk_threshold_vector = nn.Parameter(
            torch.ones(2, dtype=torch.float) * init_spk_trace_th
        )  # -0.35
        # self.spk_threshold_vector = nn.Parameter(torch.tensor([ 0.0060, -0.0637, -0.8105, -0.0786, -1.4248, -0.3059,  0.2116, -0.0589,-0.7510]))
        self.spk_threshold_vector.to("cuda" if torch.cuda.is_available() else "cpu")

        self.spk_a_vector = nn.Parameter(
            torch.ones(2, dtype=torch.float) * init_spk_trace_a
        )  # -0.8
        # self.spk_a_vector = nn.Parameter(torch.tensor([-0.3246, -0.6443, -0.6722, -0.9817, -0.9581, -0.9971, -1.1768, -0.9332, -0.7385]))
        self.spk_a_vector.to("cuda" if torch.cuda.is_available() else "cpu")

        self.acc_tau = nn.Parameter(torch.ones(1, dtype=torch.float) * init_acc_tau)
        # self.acc_tau = nn.Parameter(torch.tensor([5.6898]))
        self.acc_tau.to("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        self.device = x.device
        # spike_recording = []
        batch_size = x.size(0)
        x = x.flatten(1)
        x_spk = snn.spikegen.rate(x, self.num_steps)
        x_spk.to(x.device)

        h1_mem = h1_spike = h1_spike_trace = h1_spike_trace_b = torch.zeros(
            batch_size, self.scale, device=self.device
        )
        h2_mem = h2_spike = h2_spike_trace = h2_spike_trace_b = torch.zeros(
            batch_size, 100, device=self.device
        )
        boost1 = torch.zeros(batch_size, 10, device=self.device)
        acc_mem = torch.zeros(batch_size, 10, device=self.device)

        h1_mem.fill_(0.5)
        h2_mem.fill_(0.5)

        decay_0 = torch.sigmoid(self.tau_0)
        decay_vector = torch.sigmoid(self.tau_vector)
        acc_decay = torch.sigmoid(self.acc_tau)

        for step in range(self.num_steps - 1):
            with torch.no_grad():
                h1_mem, h1_spike = self.pPLIF1(
                    h1_mem.detach(), h1_spike.detach(), decay_0, self.fc1(x_spk[step])
                )

                h2_mem, h2_spike = self.pPLIF2(
                    h2_mem.detach(), h2_spike.detach(), decay_vector[0], self.fc2(h1_spike)
                )

                boost1 = self.boost1(h2_spike.unsqueeze(1)).squeeze(1)

                acc_mem = self.pPLI(acc_mem.detach(), acc_decay, boost1)

        h1_mem, h1_spike = self.pPLIF1(
            h1_mem.detach(), h1_spike.detach(), decay_0, self.fc1(x_spk[step + 1])
        )

        h2_mem, h2_spike = self.pPLIF2(
            h2_mem.detach(), h2_spike.detach(), decay_vector[0], self.fc2(h1_spike)
        )

        boost1 = self.boost1(h2_spike.unsqueeze(1)).squeeze(1)

        acc_mem = self.pPLI(acc_mem.detach(), acc_decay, boost1)

        return acc_mem, self.num_steps
        # return next - softmax and cross-entropy loss



if __name__ == "__main__":
    _ = SimpleDenseNet()
