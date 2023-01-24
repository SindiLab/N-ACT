"""Implementation of transfer learning for NACT's projection model."""

from .base._constants_and_types import _DEVICE_TYPES
from .nact_projection_attention import NACTProjectionAttention
import torch


class NACTFineTuning(NACTProjectionAttention):
    """Pretrained NACT model to be finetuned for a new data.

    Attributes:
        *Note: The attributes are the same as the passed on pre-trained model.
        device: A string (either "cuda" or "cpu") determining which device
          computations should be performed on.
        masking_frac: The fraction of each input cell that should be masked.
          This value should be set to zero except for when training the
          unsupervised contrastive learning mode.
        num_features: The number of inputted genes (input dimension).
        attention_module: The attention module of NACT.
        attention_dim: The dimension of the attention layer (the same as the
          input feature unless an encoding is applied at the start).
        proj_block1: The first projection block of NACT.
        proj_block2: The second and last projection block of NACT.
        pwff: The pointwise Feedforward neural network that is used as the
          activation of of the projection blocks.
        task_module: The task module of NACT. In this implementation, the task
          module is defined to be the number of cell types.
    """

    def __init__(self,
                 pretrained_nact_model: NACTProjectionAttention,
                 task_module_output_dimension: int,
                 input_dimension: int = 5000,
                 device: _DEVICE_TYPES = 'cpu'):
        """Initializer of the NACTProjectionAttention class."""

        super().__init__()
        self.device = device
        self.num_features = input_dimension
        self.out_dim = task_module_output_dimension
        # Here we transferring attributes from the pre-trained model.

        self.attention = pretrained_nact_model.attention
        self.projection_block1 = pretrained_nact_model.proj_block
        self.projectopm_block2 = pretrained_nact_model.proj_block2
        self.pwff = pretrained_nact_model.pwff
        self.attention_dim = pretrained_nact_model.attention_dim
        # Component (3)
        self.task_module = torch.nn.Sequential(
            torch.nn.Linear(self.attention_dim, task_module_output_dimension),
            torch.nn.LeakyReLU())

    def forward(self,
                input_tensor: torch.Tensor,
                training: bool = True,
                device='cpu'):
        """Forward pass of the finetuning model of NACT (for projection models).

        Args:
            input_tensor: A tensor containing input data for training.
            training: The mode that we are calling the forward function. True
              indicates that we are training the model
            device: A string ("cuda" or "cpu") indicating which device we want
              to use for performing computaions.

        Returns:
            The forward method returns three tensors:
            (1) logits: the actual logits for predictions.

            (2) alphas: The attention tensor containing the weights for all
                genes.

            (3) gamma: A tensor containing the gene scores.

        Raises:
            None.
        """
        self.training = training

        if not self.training:
            self.device = device
        # This is our way of freezing these core layers
        # TODO: Find a more efficient way of freezing these layers
        with torch.no_grad():
            x_masked = self._parallel_eval(self.masking_layer, input_tensor)
            alphas = self._softmax(self.attention_module(input_tensor))
            gamma = self._gene_scores(alphas, x_masked)

            # The abbrevation "gse" stands for gene stacked event (gse), which
            # is the output of the a projection block (with all branches).
            gse = self._parallel_eval(self.projection_block1, gamma)
            x_activated = self._parallel_eval(self.pwff, gse)

            gse2 = self._parallel_eval(self.projection_block2,
                                       x_activated + gamma)
            x_activated2 = self._parallel_eval(self.pwff, gse2)

        # This is the only layer we want to train (finetiune)!
        logits = self._parallel_eval(self.task_module, x_activated2 + gamma)

        return logits, alphas, gamma
