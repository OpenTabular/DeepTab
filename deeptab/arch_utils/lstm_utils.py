import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer_utils.block_diagonal import BlockDiagonal


class mLSTMblock(nn.Module):
    """MLSTM block with convolutions, gated mechanisms, and projection layers.

    Parameters
    ----------
    x_example : torch.Tensor
        Example input tensor for defining input dimensions.
    factor : float
        Factor to scale hidden size relative to input size.
    depth : int
        Depth of block diagonal layers.
    dropout : float, optional
        Dropout probability (default is 0.2).
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        bidirectional=None,
        batch_first=None,
        nonlinearity=F.silu,
        dropout=0.2,
        bias=True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = nonlinearity

        self.ln = nn.LayerNorm(self.input_size)

        self.left = nn.Linear(self.input_size, self.hidden_size)
        self.right = nn.Linear(self.input_size, self.hidden_size)

        self.conv = nn.Conv1d(
            in_channels=self.hidden_size,  # Hidden size for subsequent layers
            out_channels=self.hidden_size,  # Output channels
            kernel_size=3,
            padding="same",  # Padding to maintain sequence length
            bias=True,
            groups=self.hidden_size,
        )
        self.drop = nn.Dropout(dropout + 0.1)

        self.lskip = nn.Linear(self.hidden_size, self.hidden_size)

        self.wq = BlockDiagonal(
            in_features=self.hidden_size,
            out_features=self.hidden_size,
            num_blocks=num_layers,
            bias=bias,
        )
        self.wk = BlockDiagonal(
            in_features=self.hidden_size,
            out_features=self.hidden_size,
            num_blocks=num_layers,
            bias=bias,
        )
        self.wv = BlockDiagonal(
            in_features=self.hidden_size,
            out_features=self.hidden_size,
            num_blocks=num_layers,
            bias=bias,
        )
        self.dropq = nn.Dropout(dropout / 2)
        self.dropk = nn.Dropout(dropout / 2)
        self.dropv = nn.Dropout(dropout / 2)

        self.i_gate = nn.Linear(self.hidden_size, self.hidden_size)
        self.f_gate = nn.Linear(self.hidden_size, self.hidden_size)
        self.o_gate = nn.Linear(self.hidden_size, self.hidden_size)

        self.ln_c = nn.LayerNorm(self.hidden_size)
        self.ln_n = nn.LayerNorm(self.hidden_size)

        self.lnf = nn.LayerNorm(self.hidden_size)
        self.lno = nn.LayerNorm(self.hidden_size)
        self.lni = nn.LayerNorm(self.hidden_size)

        self.GN = nn.LayerNorm(self.hidden_size)
        self.ln_out = nn.LayerNorm(self.hidden_size)

        self.drop2 = nn.Dropout(dropout)

        self.proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.ln_proj = nn.LayerNorm(self.hidden_size)

        # Remove fixed-size initializations for dynamic state initialization
        self.ct_1 = None
        self.nt_1 = None

    def init_states(self, batch_size, seq_length, device):
        """Initialize the state tensors with the correct batch and sequence dimensions.

        Parameters
        ----------
        batch_size : int
            The batch size.
        seq_length : int
            The sequence length.
        device : torch.device
            The device to place the tensors on.
        """
        self.ct_1 = torch.zeros(batch_size, seq_length, self.hidden_size, device=device)
        self.nt_1 = torch.zeros(batch_size, seq_length, self.hidden_size, device=device)

    def forward(self, x):
        """Forward pass through mLSTM block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, sequence_length, input_size).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, sequence_length, input_size).
        """
        if x.ndim != 3:
            raise ValueError("Input tensor must have 3 dimensions (batch, sequence_length, input_size)")
        B, N, D = x.shape
        device = x.device

        # Initialize states dynamically based on input shape
        if self.ct_1 is None or self.ct_1.shape[0] != B or self.ct_1.shape[1] != N:
            self.init_states(B, N, device)

        x = self.ln(x)  # layer norm on x

        left = self.left(x)  # part left
        # part right with just swish (silu) function
        right = self.activation(self.right(x))

        left_left = left.transpose(1, 2)
        left_left = self.activation(self.drop(self.conv(left_left).transpose(1, 2)))
        l_skip = self.lskip(left_left)

        # start mLSTM
        q = self.dropq(self.wq(left_left))
        k = self.dropk(self.wk(left_left))
        v = self.dropv(self.wv(left))

        i = torch.exp(self.lni(self.i_gate(left_left)))
        f = torch.exp(self.lnf(self.f_gate(left_left)))
        o = torch.sigmoid(self.lno(self.o_gate(left_left)))

        ct_1 = self.ct_1

        ct = f * ct_1 + i * v * k  # type: ignore[operator]
        ct = torch.mean(self.ln_c(ct), [0, 1], keepdim=True)
        self.ct_1 = ct.detach()

        nt_1 = self.nt_1
        nt = f * nt_1 + i * k  # type: ignore[operator]
        nt = torch.mean(self.ln_n(nt), [0, 1], keepdim=True)
        self.nt_1 = nt.detach()

        ht = o * ((ct * q) / torch.max(nt * q))
        # end mLSTM
        ht = ht

        left = self.drop2(self.GN(ht + l_skip))

        out = self.ln_out(left * right)
        out = self.ln_proj(self.proj(out))

        return out, None


class sLSTMblock(nn.Module):
    """SLSTM block with convolutions, gated mechanisms, and projection layers.

    Parameters
    ----------
    input_size : int
        Size of the input features.
    hidden_size : int
        Size of the hidden state.
    num_layers : int
        Depth of block diagonal layers.
    dropout : float, optional
        Dropout probability (default is 0.2).
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        bidirectional=None,
        batch_first=None,
        nonlinearity=F.silu,
        dropout=0.2,
        bias=True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = nonlinearity

        self.drop = nn.Dropout(dropout)

        self.i_gate = BlockDiagonal(
            in_features=self.input_size,
            out_features=self.input_size,
            num_blocks=num_layers,
            bias=bias,
        )
        self.f_gate = BlockDiagonal(
            in_features=self.input_size,
            out_features=self.input_size,
            num_blocks=num_layers,
            bias=bias,
        )
        self.o_gate = BlockDiagonal(
            in_features=self.input_size,
            out_features=self.input_size,
            num_blocks=num_layers,
            bias=bias,
        )
        self.z_gate = BlockDiagonal(
            in_features=self.input_size,
            out_features=self.input_size,
            num_blocks=num_layers,
            bias=bias,
        )

        self.ri_gate = BlockDiagonal(self.input_size, self.input_size, num_layers, bias=False)
        self.rf_gate = BlockDiagonal(self.input_size, self.input_size, num_layers, bias=False)
        self.ro_gate = BlockDiagonal(self.input_size, self.input_size, num_layers, bias=False)
        self.rz_gate = BlockDiagonal(self.input_size, self.input_size, num_layers, bias=False)

        self.ln_i = nn.LayerNorm(self.input_size)
        self.ln_f = nn.LayerNorm(self.input_size)
        self.ln_o = nn.LayerNorm(self.input_size)
        self.ln_z = nn.LayerNorm(self.input_size)

        self.GN = nn.LayerNorm(self.input_size)
        self.ln_c = nn.LayerNorm(self.input_size)
        self.ln_n = nn.LayerNorm(self.input_size)
        self.ln_h = nn.LayerNorm(self.input_size)

        self.left_linear = nn.Linear(self.input_size, int(self.input_size * (4 / 3)))
        self.right_linear = nn.Linear(self.input_size, int(self.input_size * (4 / 3)))

        self.ln_out = nn.LayerNorm(int(self.input_size * (4 / 3)))

        self.proj = nn.Linear(int(self.input_size * (4 / 3)), self.hidden_size)

        # Remove initial fixed-size states
        self.ct_1 = None
        self.nt_1 = None
        self.ht_1 = None
        self.mt_1 = None

    def init_states(self, batch_size, seq_length, device):
        """Initialize the state tensors with the correct batch and sequence dimensions.

        Parameters
        ----------
        batch_size : int
            The batch size.
        seq_length : int
            The sequence length.
        device : torch.device
            The device to place the tensors on.
        """
        self.nt_1 = torch.zeros(batch_size, seq_length, self.input_size, device=device)
        self.ct_1 = torch.zeros(batch_size, seq_length, self.input_size, device=device)
        self.ht_1 = torch.zeros(batch_size, seq_length, self.input_size, device=device)
        self.mt_1 = torch.zeros(batch_size, seq_length, self.input_size, device=device)

    def forward(self, x):
        """Forward pass through sLSTM block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, sequence_length, input_size).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, sequence_length, input_size).
        """
        B, N, D = x.shape
        device = x.device

        # Initialize states dynamically based on input shape
        if self.ct_1 is None or self.nt_1 is None or self.nt_1.shape[0] != B or self.nt_1.shape[1] != N:
            self.init_states(B, N, device)

        x = self.activation(x)

        # Start sLSTM operations
        ht_1 = self.ht_1

        i = torch.exp(self.ln_i(self.i_gate(x) + self.ri_gate(ht_1)))
        f = torch.exp(self.ln_f(self.f_gate(x) + self.rf_gate(ht_1)))

        # Use expand_as to match the shapes of f and i for element-wise operations
        m = torch.max(
            torch.log(f) + self.mt_1.expand_as(f),  # type: ignore
            torch.log(i),  # type: ignore
        )
        i = torch.exp(torch.log(i) - m)
        f = torch.exp(torch.log(f) + self.mt_1.expand_as(f) - m)  # type: ignore
        self.mt_1 = m.detach()

        o = torch.sigmoid(self.ln_o(self.o_gate(x) + self.ro_gate(ht_1)))
        z = torch.tanh(self.ln_z(self.z_gate(x) + self.rz_gate(ht_1)))

        ct_1 = self.ct_1
        ct = f * ct_1 + i * z  # type: ignore[operator]
        ct = torch.mean(self.ln_c(ct), [0, 1], keepdim=True)
        self.ct_1 = ct.detach()

        nt_1 = self.nt_1
        nt = f * nt_1 + i  # type: ignore[operator]
        nt = torch.mean(self.ln_n(nt), [0, 1], keepdim=True)
        self.nt_1 = nt.detach()

        ht = o * (ct / nt)
        ht = torch.mean(self.ln_h(ht), [0, 1], keepdim=True)
        self.ht_1 = ht.detach()

        slstm_out = self.GN(ht)

        left = self.left_linear(slstm_out)
        right = F.gelu(self.right_linear(slstm_out))

        out = self.ln_out(left * right)
        out = self.proj(out)
        return out, None
