import math
import time
import torch
import torch.nn as nn
from .functional import cutoff_function
from .modules import *
from typing import Tuple, Optional
import pytorch_lightning as pl
from torch.optim import lr_scheduler
import torchmetrics

# backwards compatibility with old versions of pytorch
try:
    from torch.linalg import norm
except:
    from torch import norm


class SpookyNetLightning(pl.LightningModule):
    """
    Neural network for PES construction augmented with optional explicit terms
    for short-range repulsion, electrostatics and dispersion and explicit nonlocal
    interactions.
    IMPORTANT: Angstrom and electron volts are assumed to be the units for
    length and energy (charge is measured in elementary charge). When other
    units are used, some constants for computing short-range repulsion,
    electrostatics and dispersion need to be changed accordingly. If these terms
    are not used, no changes are necessary. It is recommended to work with units
    of Angstrom and electron volts to prevent the need to change the code.

    Arguments:
        activation (str):
            Kind of activation function. Possible values:
            'swish': Swish activation function.
            'ssp': Shifted softplus activation function.
        num_features (int):
            Dimensions of feature space.
        num_basis_functions (int):
            Number of radial basis functions.
        num_modules (int):
            Number of modules (iterations) for constructing atomic features.
        num_residual_electron (int):
            Number of residual blocks applied to features encoding the electronic 
            state.
        num_residual_pre (int):
            Number of residual blocks applied to atomic features in each module
            (before other transformations).
        num_residual_post (int):
            Number of residual blocks applied to atomic features after
            interaction with neighbouring atoms (per module).
        num_residual_pre_local_x (int):
            Number of residual blocks (per module) applied to atomic features in 
            local interaction.
        num_residual_pre_local_s (int):
            Number of residual blocks (per module) applied to s-type interaction features 
            in local interaction.
        num_residual_pre_local_p (int):
            Number of residual blocks (per module) applied to p-type interaction features 
            in local interaction.
        num_residual_pre_local_d (int):
            Number of residual blocks (per module) applied to d-type interaction features 
            in local interaction.
        num_residual_post (int):
            Number of residual blocks applied to atomic features in each module
            (after other transformations).
        num_residual_output (int):
            Number of residual blocks applied to atomic features in output
            branch (per module).
        basis_functions (str):
            Kind of radial basis functions. Possible values:
            'exp-bernstein': Exponential Bernstein polynomials.
            'exp-gaussian': Exponential Gaussian functions.
            'bernstein': Bernstein polynomials.
            'gaussian': Gaussian functions.
        exp_weighting (bool):
            Apply exponentially decaying weights to radial basis functions. Only
            used when 'basis_functions' argument is 'exp-bernstein' or
            'exp-gaussian'. Probably has almost no effect unless the weights of
            radial functions are regularized.
        cutoff (float):
            Cutoff radius for (neural network) interactions.
        lr_cutoff (float or None):
            Cutoff radius for long-range interactions (no cutoff is applied when
            this argument is None).
        use_zbl_repulsion (bool):
            If True, short-range repulsion inspired by the ZBL repulsive
            potential is applied to the energy prediction.
        use_electrostatics (bool):
            If True, point-charge electrostatics for the predicted atomic
            partial charges is applied to the energy prediction.
        use_d4_dispersion (bool):
            If True, Grimme's D4 dispersion correction is applied to the energy
            prediction.
        use_irreps (bool):
            For compatibility with older versions of the code.
        use_nonlinear_embedding (bool):
            For compatibility with older versions of the code.
        compute_d4_atomic (bool):
            If True, atomic polarizabilities and C6 coefficients in Grimme's D4
            dispersion correction are computed.
        module_keep_prob (float):
            Probability of keeping the last module during training. Module
            dropout can be a useful regularization that encourages
            hierarchicacally decaying contributions to the atomic features.
            Earlier modules are dropped with an automatically determined lower
            probability. Should be between 0.0 and 1.0.
        load_from (str or None):
            Load saved parameters from the given path instead of using random
            initialization (when 'load_from' is None).
        Zmax (int):
            Maximum nuclear charge +1 of atoms. The default is 87, so all
            elements up to Rn (Z=86) are supported. Can be kept at the default
            value (has minimal memory impact). Note that Grimme's D4 dispersion
            can only handle elements up to Rn (Z=86).
        zero_init (bool): Initialize parameters with zero whenever possible?
    """

    def __init__(
        self,
        activation="swish",
        num_features=64,
        num_basis_functions=16,
        num_modules=3,
        num_residual_electron=1,
        num_residual_pre=1,
        num_residual_local_x=1,
        num_residual_local_s=1,
        num_residual_local_p=1,
        num_residual_local_d=1,
        num_residual_local=1,
        num_residual_nonlocal_q=1,
        num_residual_nonlocal_k=1,
        num_residual_nonlocal_v=1,
        num_residual_post=1,
        num_residual_output=1,
        basis_functions="exp-bernstein",
        exp_weighting=True,
        cutoff=5.291772105638412,  # 10 a0 in A
        use_zbl_repulsion=True,
        use_electrostatics=True,
        use_d4_dispersion=True,
        use_irreps=True,
        use_nonlinear_embedding=False,
        compute_d4_atomic=False,
        module_keep_prob=1.0,
        load_from=None,
        Zmax=87,
        zero_init=True,
        learning_rate=1e-2,
        weight_decay=1e-10,
        lr_patience=50,
        lr_cutoff=None,
        lr_factor=0.5,
        dipole=False, 
        scheduler_name="reduce_on_plateau",
        loss_weights = {"E": 1, "F": 10, "D": 1}
    ) -> None:
        """ Initializes the SpookyNet class. """
        super(SpookyNetLightning, self).__init__()
        #super().__init__()
        self.learning_rate = learning_rate

        # load state from a file (if load_from is not None) and overwrite
        # the given arguments.
        if load_from is not None:
            saved_state = torch.load(load_from, map_location="gpu")
            activation = saved_state["activation"]
            num_features = saved_state["num_features"]
            num_basis_functions = saved_state["num_basis_functions"]
            num_modules = saved_state["num_modules"]
            num_residual_electron = saved_state["num_residual_electron"]
            num_residual_pre = saved_state["num_residual_pre"]
            num_residual_local_x = saved_state["num_residual_local_x"]
            num_residual_local_s = saved_state["num_residual_local_s"]
            num_residual_local_p = saved_state["num_residual_local_p"]
            num_residual_local_d = saved_state["num_residual_local_d"]
            num_residual_local = saved_state["num_residual_local"]
            num_residual_nonlocal_q = saved_state["num_residual_nonlocal_q"]
            num_residual_nonlocal_k = saved_state["num_residual_nonlocal_k"]
            num_residual_nonlocal_v = saved_state["num_residual_nonlocal_v"]
            num_residual_post = saved_state["num_residual_post"]
            num_residual_output = saved_state["num_residual_output"]
            basis_functions = saved_state["basis_functions"]
            exp_weighting = saved_state["exp_weighting"]
            cutoff = saved_state["cutoff"]
            lr_cutoff = saved_state["lr_cutoff"]
            use_zbl_repulsion = saved_state["use_zbl_repulsion"]
            use_electrostatics = saved_state["use_electrostatics"]
            use_d4_dispersion = saved_state["use_d4_dispersion"]
            compute_d4_atomic = saved_state["compute_d4_atomic"]
            module_keep_prob = saved_state["module_keep_prob"]
            Zmax = saved_state["Zmax"]
            lr_patience = saved_state["lr_patience"]
            lr_factor = saved_state["lr_factor"]
            weight_decay = saved_state["weight_decay"]
            dipole = saved_state["dipole"]
            scheduler_name = saved_state["scheduler_name"]
            loss_weights = saved_state["loss_weights"]
            # compatibility with older code
            if "use_irreps" in saved_state:
                use_irreps = saved_state["use_irreps"]
            else:
                use_irreps = False
            if "use_nonlinear_embedding" in saved_state:
                use_nonlinear_embedding = saved_state["use_nonlinear_embedding"]
            else:
                use_nonlinear_embedding = True

        # store argument values as attributes
        self.activation = activation
        self.num_features = num_features
        self.num_basis_functions = num_basis_functions
        self.num_modules = num_modules
        self.num_residual_electron = num_residual_electron
        self.num_residual_pre = num_residual_pre
        self.num_residual_local_x = num_residual_local_x
        self.num_residual_local_s = num_residual_local_s
        self.num_residual_local_p = num_residual_local_p
        self.num_residual_local_d = num_residual_local_d
        self.num_residual_local = num_residual_local
        self.num_residual_nonlocal_q = num_residual_nonlocal_q
        self.num_residual_nonlocal_k = num_residual_nonlocal_k
        self.num_residual_nonlocal_v = num_residual_nonlocal_v
        self.num_residual_post = num_residual_post
        self.num_residual_output = num_residual_output
        self.basis_functions = basis_functions
        self.exp_weighting = exp_weighting
        self.cutoff = cutoff
        self.lr_cutoff = lr_cutoff
        self.use_zbl_repulsion = use_zbl_repulsion
        self.use_electrostatics = use_electrostatics
        self.use_d4_dispersion = use_d4_dispersion
        self.use_irreps = use_irreps
        self.use_nonlinear_embedding = use_nonlinear_embedding
        self.compute_d4_atomic = compute_d4_atomic
        self.module_keep_prob = module_keep_prob
        self.Zmax = Zmax
        self.lr_decay = lr_factor
        self.weight_decay = weight_decay
        self.lr_patience = lr_patience
        self.dipole = dipole
        self.scheduler_name = scheduler_name
        self.loss_weights = loss_weights

        params = {
            "activation": activation,
            "num_features": num_features,
            "num_basis_functions": num_basis_functions,
            "num_modules": num_modules,
            "num_residual_electron": num_residual_electron,
            "num_residual_pre": num_residual_pre,
            "num_residual_local_x": num_residual_local_x,
            "num_residual_local_s": num_residual_local_s,
            "num_residual_local_p": num_residual_local_p,
            "num_residual_local_d": num_residual_local_d,
            "num_residual_local": num_residual_local,
            "num_residual_nonlocal_q": num_residual_nonlocal_q,
            "num_residual_nonlocal_k": num_residual_nonlocal_k,
            "num_residual_nonlocal_v": num_residual_nonlocal_v,
            "num_residual_post": num_residual_post,
            "num_residual_output": num_residual_output,
            "basis_functions": basis_functions,
            "exp_weighting": exp_weighting,
            "cutoff": cutoff,
            "lr_cutoff": lr_cutoff,
            "use_zbl_repulsion": use_zbl_repulsion,
            "use_electrostatics": use_electrostatics,
            "use_d4_dispersion": use_d4_dispersion,
            "use_irreps": use_irreps,
            "use_nonlinear_embedding": use_nonlinear_embedding,
            "compute_d4_atomic": compute_d4_atomic,
            "module_keep_prob": module_keep_prob,
            "weight_decay": weight_decay,
            "lr_patience": lr_patience,
            "lr_factor": lr_factor,
            "Zmax": Zmax,
            "dipole": dipole,
            "scheduler_name": scheduler_name,
            "loss_weights": loss_weights,
        }
        self.hparams.update(params)
        self.save_hyperparameters()

        # set device for model
        
        # for performing module dropout
        if self.module_keep_prob < 0.0 or self.module_keep_prob > 1.0:
            raise ValueError(
                "Argument 'module_keep_prob' must take values "
                "between 0.0 and 1.0 but received " + str(self.module_keep_prob)
            )
        if self.num_modules > 1:
            self.register_buffer(
                "keep_prob",
                torch.tensor(self.module_keep_prob ** (1 / (self.num_modules - 1))),
            )
        else:
            self.register_buffer("keep_prob", torch.tensor(self.module_keep_prob))

        # declare modules and parameters
        # element specific energy and charge bias
        self.register_parameter(
            "element_bias", nn.Parameter(torch.Tensor(self.Zmax, 2))
        )

        # embeddings
        self.nuclear_embedding = NuclearEmbedding(
            self.num_features, self.Zmax, zero_init=zero_init
        )
        if self.use_nonlinear_embedding:
            self.charge_embedding = NonlinearElectronicEmbedding(
                self.num_features, self.num_residual_electron, activation
            )
            self.magmom_embedding = NonlinearElectronicEmbedding(
                self.num_features, self.num_residual_electron, activation
            )
        else:
            self.charge_embedding = ElectronicEmbedding(
                self.num_features,
                self.num_residual_electron,
                activation,
                is_charge=True,
            )
            self.magmom_embedding = ElectronicEmbedding(
                self.num_features,
                self.num_residual_electron,
                activation,
                is_charge=False,
            )

        # radial basis functions
        if self.basis_functions == "exp-gaussian":
            self.radial_basis_functions = ExponentialGaussianFunctions(
                self.num_basis_functions, exp_weighting=self.exp_weighting
            )
        elif self.basis_functions == "exp-bernstein":
            self.radial_basis_functions = ExponentialBernsteinPolynomials(
                self.num_basis_functions, exp_weighting=self.exp_weighting
            )
        elif self.basis_functions == "gaussian":
            self.radial_basis_functions = GaussianFunctions(
                self.num_basis_functions, self.cutoff
            )
        elif self.basis_functions == "bernstein":
            self.radial_basis_functions = BernsteinPolynomials(
                self.num_basis_functions, self.cutoff
            )
        elif self.basis_functions == "sinc":
            self.radial_basis_functions = SincFunctions(
                self.num_basis_functions, self.cutoff
            )
        else:
            raise ValueError(
                "Argument 'basis_functions' may only take the "
                "values 'exp-gaussian','exp-bernstein','gaussian','bernstein',"
                " or 'sinc' but received '" + str(self.basis_functions) + "'."
            )

        # interaction modules (iterations)
        self.module = nn.ModuleList(
            [
                InteractionModule(
                    num_features=self.num_features,
                    num_basis_functions=self.num_basis_functions,
                    num_residual_pre=self.num_residual_pre,
                    num_residual_local_x=self.num_residual_local_x,
                    num_residual_local_s=self.num_residual_local_s,
                    num_residual_local_p=self.num_residual_local_p,
                    num_residual_local_d=self.num_residual_local_d,
                    num_residual_local=self.num_residual_local,
                    num_residual_nonlocal_q=self.num_residual_nonlocal_q,
                    num_residual_nonlocal_k=self.num_residual_nonlocal_k,
                    num_residual_nonlocal_v=self.num_residual_nonlocal_v,
                    num_residual_post=self.num_residual_post,
                    num_residual_output=self.num_residual_output,
                    activation=self.activation,
                )
                for i in range(self.num_modules)
            ]
        )

        # output layer (2 outputs for atomic energy and partial charge)
        self.output = nn.Linear(self.num_features, 2, bias=False)

        # ZBL inspired short-range repulsion
        if use_zbl_repulsion:
            self.zbl_repulsion_energy = ZBLRepulsionEnergy()

        # point-charge electrostatics
        if use_electrostatics:
            self.electrostatic_energy = ElectrostaticEnergy(
                cuton=0.25 * self.cutoff,
                cutoff=0.75 * self.cutoff,
                lr_cutoff=self.lr_cutoff,
            )

        # Grimme's D4 dispersion
        if use_d4_dispersion:
            self.d4_dispersion_energy = D4DispersionEnergy(cutoff=self.lr_cutoff)

        # constants used for calculating angular functions
        self._sqrt2 = math.sqrt(2.0)
        self._sqrt3 = math.sqrt(3.0)
        self._sqrt3half = 0.5 * self._sqrt3

        # initialize parameters
        if load_from is None:  # random initialization
            self.reset_parameters()
        else:  # load parameters from file
            try:
                self.load_state_dict(saved_state["state_dict"])
            # runtime exception may happen if state_dict was saved with an older
            # version of the code, but it should be possible to convert it
            except RuntimeError:
                self.load_state_dict(
                    self._convert_state_dict(saved_state["state_dict"])
                )
            if use_d4_dispersion:
                self.d4_dispersion_energy._compute_refc6()

        # build dictionary that determines which parameters require gradients
        self.build_requires_grad_dict()


        # define statistics
        self.train_E_l1 = torchmetrics.MeanAbsoluteError()
        self.train_E_rl2 = torchmetrics.MeanSquaredError(squared=False)
        self.train_F_l1 = torchmetrics.MeanAbsoluteError()
        self.train_F_rl2 = torchmetrics.MeanSquaredError(squared=False)
        self.train_D_l1 = torchmetrics.MeanAbsoluteError()

        self.val_E_l1 = torchmetrics.MeanAbsoluteError()
        self.val_E_rl2 = torchmetrics.MeanSquaredError(squared=False)
        self.val_F_l1 = torchmetrics.MeanAbsoluteError()
        self.val_F_rl2 = torchmetrics.MeanSquaredError(squared=False)
        self.val_D_l1 = torchmetrics.MeanAbsoluteError()

        self.test_E_l1 = torchmetrics.MeanAbsoluteError()
        self.test_E_rl2 = torchmetrics.MeanSquaredError(squared=False)
        self.test_F_l1 = torchmetrics.MeanAbsoluteError()
        self.test_F_rl2 = torchmetrics.MeanSquaredError(squared=False)
        self.test_D_l1 = torchmetrics.MeanAbsoluteError()

        
        
        # move metrics to correct device
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_E_l1.cuda()
        self.train_E_rl2.cuda()
        self.train_F_l1.cuda()
        self.train_F_rl2.cuda()

        self.val_E_l1.cuda()
        self.val_E_rl2.cuda()
        self.val_F_l1.cuda()
        self.val_F_rl2.cuda()
        

        self.test_E_l1.cuda()
        self.test_E_rl2.cuda()
        self.test_F_l1.cuda()
        self.test_F_rl2.cuda()


        self.loss = self.loss_function()

    def reset_parameters(self) -> None:
        """ Initialize parameters randomly. """
        nn.init.orthogonal_(self.output.weight)
        nn.init.zeros_(self.element_bias)

    def set_lr_cutoff(self, lr_cutoff: Optional[float] = None) -> None:
        """
        Can be used to set lr_cutoff after a model is trained. Only use this if
        you know what you're doing!
        """
        self.lr_cutoff = lr_cutoff
        if self.use_electrostatics:
            self.electrostatic_energy.set_lr_cutoff(lr_cutoff)
        if self.use_d4_dispersion:
            self.d4_dispersion_energy.set_cutoff(lr_cutoff)

    @property
    def dtype(self) -> torch.dtype:
        """ Return torch.dtype of parameters (input tensors must match). """
        return self.nuclear_embedding.element_embedding.dtype

    #@property
    #def device(self) -> torch.device:
    #    """ Return torch.device of parameters (input tensors must match). """
    #    return self.nuclear_embedding.element_embedding.device

    def train(self, mode: bool = True) -> None:
        """ Turn on training mode. """
        super(SpookyNetLightning, self).train(mode=mode)
        for name, param in self.named_parameters():
            param.requires_grad = self.requires_grad_dict[name]

    def eval(self) -> None:
        """ Turn on evaluation mode (smaller memory footprint)."""
        super(SpookyNetLightning, self).eval()
        for name, param in self.named_parameters():
            param.requires_grad = False

    def build_requires_grad_dict(self) -> None:
        """
        Build a dictionary of which parameters require gradient information (are
        trained). Can be manually edited to freeze certain parameters)
        """
        self.requires_grad_dict = {}
        for name, param in self.named_parameters():
            self.requires_grad_dict[name] = param.requires_grad

    def save(self, path: str) -> None:
        """
        Saves all model parameters and architecture hyperparameters to a file.
        This file can be passed to the 'load_from' argument during
        initialization to reconstruct the model from the saved state.

        Arguments:
            path (str):
                Filepath to which parameters are saved (a .pth extension is
                recommended).
        """
        torch.save(
            {
                "state_dict": self.state_dict(),
                "activation": self.activation,
                "num_features": self.num_features,
                "num_basis_functions": self.num_basis_functions,
                "num_modules": self.num_modules,
                "num_residual_electron": self.num_residual_electron,
                "num_residual_pre": self.num_residual_pre,
                "num_residual_local_x": self.num_residual_local_x,
                "num_residual_local_s": self.num_residual_local_s,
                "num_residual_local_p": self.num_residual_local_p,
                "num_residual_local_d": self.num_residual_local_d,
                "num_residual_local": self.num_residual_local,
                "num_residual_nonlocal_q": self.num_residual_nonlocal_q,
                "num_residual_nonlocal_k": self.num_residual_nonlocal_k,
                "num_residual_nonlocal_v": self.num_residual_nonlocal_v,
                "num_residual_post": self.num_residual_post,
                "num_residual_output": self.num_residual_output,
                "basis_functions": self.basis_functions,
                "exp_weighting": self.exp_weighting,
                "cutoff": self.cutoff,
                "lr_cutoff": self.lr_cutoff,
                "use_zbl_repulsion": self.use_zbl_repulsion,
                "use_electrostatics": self.use_electrostatics,
                "use_d4_dispersion": self.use_d4_dispersion,
                "use_irreps": self.use_irreps,
                "use_nonlinear_embedding": self.use_nonlinear_embedding,
                "compute_d4_atomic": self.compute_d4_atomic,
                "module_keep_prob": self.module_keep_prob,
                "Zmax": self.Zmax,
                "weight_decay": self.weight_decay,
                "lr_patience": self.lr_patience,
                "lr_factor": self.lr_decay,
                "dipole": self.dipole,
                "scheduler_name": self.scheduler_name,
                "loss_weights": self.loss_weights,
            },
            path,
        )

    def _convert_state_dict(self, old_state_dict: dict) -> dict:
        """
        Helper function to convert a state_dict saved with an old version of the
        code to the current version.
        """
        def prefix_postfix(string, pattern, prefix="resblock", sep=".", presep="_"):
            """ Helper function for converting keys """
            parts = string.split(sep)
            for i, part in enumerate(parts):
                if pattern + presep in part:
                    parts[i] = prefix + presep + part.split(presep)[-1] + sep + pattern
            return sep.join(parts)

        new_state_dict = {}
        for old_key in old_state_dict:
            if old_key == "idx" or old_key == "mul":
                continue

            if (
                "local_interaction.residual_" in old_key
                or "embedding.residual_" in old_key
            ):
                new_key = prefix_postfix(old_key, "residual")
            elif (
                "local_interaction.activation_" in old_key
                or "embedding.activation_" in old_key
            ):
                new_key = prefix_postfix(old_key, "activation")
            elif (
                "local_interaction.linear_" in old_key or "embedding.linear_" in old_key
            ):
                if "embedding.linear_q" in old_key:
                    new_key = old_key
                else:
                    new_key = prefix_postfix(old_key, "linear")
            elif ".local_interaction.residual." in old_key:
                new_key = old_key.replace(".residual.", ".resblock.residual.")
            elif ".local_interaction.activation." in old_key:
                new_key = old_key.replace(".activation.", ".resblock.activation.")
            elif ".local_interaction.linear." in old_key:
                new_key = old_key.replace(".linear.", ".resblock.linear.")
            elif ".local_interaction.projection_" in old_key:
                if "1.weight" in old_key:
                    tmp_key = old_key.replace("1.weight", "2.weight")
                    new_key = old_key.replace("1.weight", ".weight")
                    weight1 = old_state_dict[old_key]
                    weight2 = old_state_dict[tmp_key]
                    new_state_dict[new_key] = torch.cat([weight1, weight2])
                continue
            elif ".output." in old_key:
                new_key = old_key.replace(".output.", ".resblock.")
            else:
                new_key = old_key

            if "activation_pre" in new_key:
                new_key = new_key.replace("activation_pre", "activation1")
            if "activation_post" in new_key:
                new_key = new_key.replace("activation_post", "activation2")
            new_state_dict[new_key] = old_state_dict[old_key]
        return new_state_dict

    def get_number_of_parameters(self) -> int:
        """ Returns the total number of parameters. """
        num = 0
        for param in self.parameters():
            num += param.numel()
        return num

    def calculate_distances(
        self,
        R: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
        cell_offsets: Optional[torch.Tensor] = None,
        batch_seg: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate pairwise interatomic distances.
        N: Number of atoms.
        P: Number of atom pairs.
        B: Batch size (number of different molecules).

        Arguments:
            R (FloatTensor [N, 3]):
                Cartesian coordinates (x,y,z) of atoms.
            idx_i (LongTensor [P]):
                Index of atom i for all atomic pairs ij. Each pair must be
                specified as both ij and ji.
            idx_j (LongTensor [P]):
                Same as idx_i, but for atom j.
            cell (FloatTensor [B, 3, 3] or None):
                Lattice vectors of periodic cell for each molecule in the batch.
                When this is None, no periodic boundary conditions are used.
            cell_offsets (FloatTensor [P, 3] or None) :
                Offset vectors for atom j of each atomic pair given in multiples
                of the given cell. For example, the offset [1., 0., -2.] means
                that atom j is shifted one cell in positive x-direction and
                two cells in negative z-direction.
            batch_seg (LongTensor [N]):
                Index for each atom that specifies to which molecule in the
                batch it belongs. For example, when predicting a H2O and a CH4
                molecule, batch_seg would be [0, 0, 0, 1, 1, 1, 1, 1] to
                indicate that the first three atoms belong to the first molecule
                and the last five atoms to the second molecule.

        Returns:
            rij (FloatTensor [P]):
                Pairwise interatomic distances.
            vij (FloatTensor [P, 3]):
                Pairwise interatomic distance vectors.
        """
        #if R.device.type == "cpu":  # indexing is faster on CPUs
        #    Ri = R[idx_i]
        #    Rj = R[idx_j]
        #else:  # gathering is faster on GPUs
        Ri = torch.gather(R, 0, idx_i.view(-1, 1).expand(-1, 3))
        Rj = torch.gather(R, 0, idx_j.view(-1, 1).expand(-1, 3))
        if (
            cell is not None and cell_offsets is not None and batch_seg is not None
        ):  # apply PBCs
            #if cell.device.type == "cpu":  # indexing is faster on CPUs
            #    cells = cell[batch_seg][idx_i]

            #else:  # gathering is faster on GPUs
            cells = torch.gather(
                torch.gather(cell, 0, batch_seg.view(-1, 1, 1).expand(-1, 3, 3)),
                0,
                idx_i.view(-1, 1, 1).expand(-1, 3, 3),
            )
            offsets = torch.squeeze(torch.bmm(cell_offsets.view(-1, 1, 3), cells), -2)
            Rj += offsets
        vij = Rj - Ri
        return (norm(vij, dim=-1), vij)

    def _atomic_properties_static(
        self,
        Z: torch.Tensor,
        R: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
        cell_offsets: Optional[torch.Tensor] = None,
        num_batch: int = 1,
        batch_seg: Optional[torch.Tensor] = None,
    ) -> Tuple[
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        """
        Static part of the atomic properties computation which does not depend
        on parameters. For ensembles, this part of the computation can be
        shared.
        """
        # compute interatomic distances and cutoff values
        #print("R device: ", R.device)
        #print("idx_i device: ", idx_i.device)
        #print("idx_j device: ", idx_j.device)
        #print("batch_seg device: ", batch_seg.device)

        (rij, vij) = self.calculate_distances(
            R=R,
            idx_i=idx_i,
            idx_j=idx_j,
            cell=cell,
            cell_offsets=cell_offsets,
            batch_seg=batch_seg,
        )
        #print("rij device: ", rij.device)
        #print("vij device: ", vij.device)

        # short-range distances
        cutmask = rij < self.cutoff  # select all entries below cutoff
        sr_rij = rij[cutmask]
        pij = vij[cutmask] / sr_rij.unsqueeze(-1)
        if self.use_irreps:  # irreducible representation
            dij = torch.stack(
                [
                    self._sqrt3 * pij[:, 0] * pij[:, 1],  # xy
                    self._sqrt3 * pij[:, 0] * pij[:, 2],  # xz
                    self._sqrt3 * pij[:, 1] * pij[:, 2],  # yz
                    0.5 * (3 * pij[:, 2] * pij[:, 2] - 1.0),  # z2
                    self._sqrt3half
                    * (pij[:, 0] * pij[:, 0] - pij[:, 1] * pij[:, 1]),  # x2-y2
                ],
                dim=-1,
            )
        else:  # reducible Cartesian functions
            dij = torch.stack(
                [
                    pij[:, 0] * pij[:, 0],  # x2
                    pij[:, 1] * pij[:, 1],  # y2
                    pij[:, 2] * pij[:, 2],  # z2
                    self._sqrt2 * pij[:, 0] * pij[:, 1],  # x*y
                    self._sqrt2 * pij[:, 0] * pij[:, 2],  # x*z
                    self._sqrt2 * pij[:, 1] * pij[:, 2],  # y*z
                ],
                dim=-1,
            )
        sr_idx_i = idx_i[cutmask]
        sr_idx_j = idx_j[cutmask]
        cutoff_values = cutoff_function(sr_rij, self.cutoff)

        # mask for efficient attention
        if num_batch > 1 and batch_seg is not None:
            #one_hot = nn.functional.one_hot(batch_seg).to(
            #    dtype=R.dtype, device=R.device
            #)
            one_hot = nn.functional.one_hot(batch_seg).type_as(
                R
            )
            mask = one_hot @ one_hot.transpose(-1, -2)
        else:
            mask = None
        return (
            Z.size(0),
            cutoff_values,
            rij,
            sr_rij,
            pij,
            dij,
            sr_idx_i,
            sr_idx_j,
            mask,
        )

    def _atomic_properties_dynamic(
        self,
        N: int,
        Q: torch.Tensor,
        S: torch.Tensor,
        Z: torch.Tensor,
        R: torch.Tensor,
        cutoff_values: torch.Tensor,
        rij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        sr_rij: torch.Tensor,
        pij: torch.Tensor,
        dij: torch.Tensor,
        sr_idx_i: torch.Tensor,
        sr_idx_j: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
        num_batch: int = 1,
        batch_seg: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Dynamic part of the atomic properties computation which depends on
        parameters.
        """
        if batch_seg is None:  # assume a single batch
            batch_seg = Z.new_zeros(Z.size(0))

        # compute radial functions
        rbf = self.radial_basis_functions(sr_rij, cutoff_values)

        # initialize feature vectors
        z = self.nuclear_embedding(Z)
        if num_batch > 1:
            #electronic_mask = (
            #    nn.functional.one_hot(batch_seg)
            #    .to(dtype=rij.dtype, device=rij.device)
            #    .transpose(-1, -2)
            #)
            electronic_mask = (
                nn.functional.one_hot(batch_seg)
                .type_as(rij)
                .transpose(-1, -2)
            )
        else:
            electronic_mask = None
        q = self.charge_embedding(z, Q, num_batch, batch_seg, electronic_mask)
        s = self.magmom_embedding(z, S, num_batch, batch_seg, electronic_mask)
        x = z + q + s

        # initialize dropout mask
        #dropout_mask = torch.ones((num_batch, 1), dtype=x.dtype, device=x.device)
        dropout_mask = torch.ones((num_batch, 1), dtype=x.dtype)

        # perform iterations over modules
        f = x.new_zeros(x.size())  # initialize output features to zero
        for module in self.module:
            x, y = module(
                x, rbf, pij, dij, sr_idx_i, sr_idx_j, num_batch, batch_seg, mask
            )
            # apply dropout mask
            if self.training and self.module_keep_prob < 1.0:
                y = y * dropout_mask[batch_seg]
                dropout_mask = dropout_mask * torch.bernoulli(
                    self.keep_prob
                    * torch.ones(
                        dropout_mask.shape,
                        dtype=dropout_mask.dtype,
                        #device=dropout_mask.device,
                    )
                )
            f += y  # accumulate module output to features

        # predict atomic energy and partial charge and add bias terms
        #if self.element_bias.device.type == "cpu":  # indexing is faster on CPUs
        #    out = self.output(f) + self.element_bias[Z]
        #else:  # gathering is faster on GPUs
        out = self.output(f) + torch.gather(
            self.element_bias,
            0,
            Z.view(-1, 1).expand(-1, self.element_bias.shape[-1]),
        )
        ea = out.narrow(-1, 0, 1).squeeze(-1)  # atomic energy
        qa = out.narrow(-1, 1, 1).squeeze(-1)  # partial charge

        # correct partial charges for charge conservation
        # (spread leftover charge evenly over all atoms)
        w = torch.ones(N, dtype=qa.dtype, device=qa.device)
        #w = torch.ones(N, dtype=qa.dtype)
        Qleftover = Q.index_add(0, batch_seg, -qa)
        #print(w.device, batch_seg.device, Qleftover.device)
        wnorm = w.new_zeros(num_batch).index_add_(0, batch_seg, w)
        #if w.device.type == "cpu":  # indexing is faster on CPUs
        #    w = w / wnorm[batch_seg]
        #    qa = qa + w * Qleftover[batch_seg]
        #else:  # gathering is faster on GPUs
        w = w / torch.gather(wnorm, 0, batch_seg)
        qa = qa + w * torch.gather(Qleftover, 0, batch_seg)

        # compute ZBL inspired short-range repulsive contributions
        if self.use_zbl_repulsion:
            #ea_rep = self.zbl_repulsion_energy(
            #    N, Z.to(self.dtype), sr_rij, cutoff_values, sr_idx_i, sr_idx_j
            #)
            ea_rep = self.zbl_repulsion_energy(
                N, Z.type_as(sr_rij), sr_rij, cutoff_values, sr_idx_i, sr_idx_j
            )
        else:
            ea_rep = ea.new_zeros(N)

        # optimization when lr_cutoff is used
        if self.lr_cutoff is not None and (
            self.use_electrostatics or self.use_d4_dispersion
        ):
            mask = rij < self.lr_cutoff  # select all entries below lr_cutoff
            rij = rij[mask]
            idx_i = idx_i[mask]
            idx_j = idx_j[mask]

        # compute electrostatic contributions
        if self.use_electrostatics:
            ea_ele = self.electrostatic_energy(
                N, qa, rij, idx_i, idx_j, R, cell, num_batch, batch_seg
            )
        else:
            ea_ele = ea.new_zeros(N)
        # compute dispersion contributions
        if self.use_d4_dispersion:
            ea_vdw, pa, c6 = self.d4_dispersion_energy(
                N, Z, qa, rij, idx_i, idx_j, self.compute_d4_atomic
            )
        else:
            ea_vdw, pa, c6 = ea.new_zeros(N), ea.new_zeros(N), ea.new_zeros(N)
        return (f, ea, qa, ea_rep, ea_ele, ea_vdw, pa, c6)

    @torch.jit.export
    def atomic_properties(
        self,
        Z: torch.Tensor,
        Q: torch.Tensor,
        S: torch.Tensor,
        R: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
        cell_offsets: Optional[torch.Tensor] = None,
        num_batch: int = 1,
        batch_seg: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Computes atomic properties. The computation is split into a "static"
        part, which does not depend on any parameters and a "dynamic" part,
        which does depend on parameters. This allows to reuse the static part
        for different models for ensemble predictions.
        N: Number of atoms.
        P: Number of atom pairs.
        B: Batch size (number of different molecules).

        Arguments:
            Z (LongTensor [N]):
                Nuclear charges (atomic numbers) of atoms.
            Q (FloatTensor [B]):
                Total charge of each molecule in the batch.
            S (FloatTensor [B]):
                Total magnetic moment of each molecule in the batch. For
                example, a singlet has S=0, a doublet S=1, a triplet S=2, etc.
            R (FloatTensor [N, 3]):
                Cartesian coordinates (x,y,z) of atoms.
            idx_i (LongTensor [P]):
                Index of atom i for all atomic pairs ij. Each pair must be
                specified as both ij and ji.
            idx_j (LongTensor [P]):
                Same as idx_i, but for atom j.
            cell (FloatTensor [B, 3, 3] or None):
                Lattice vectors of periodic cell for each molecule in the batch.
                When this is None, no periodic boundary conditions are used.
            cell_offsets (FloatTensor [P, 3] or None) :
                Offset vectors for atom j of each atomic pair given in multiples
                of the given cell. For example, the offset [1., 0., -2.] means
                that atom j is shifted one cell in positive x-direction and
                two cells in negative z-direction.
            num_batch (int):
                Batch size (number of different molecules).
            batch_seg (LongTensor [N]):
                Index for each atom that specifies to which molecule in the
                batch it belongs. For example, when predicting a H2O and a CH4
                molecule, batch_seg would be [0, 0, 0, 1, 1, 1, 1, 1] to
                indicate that the first three atoms belong to the first molecule
                and the last five atoms to the second molecule.

        Returns:
            f (FloatTensor [N, self.num_features]):
                Atomic feature vectors (environment descriptors).
            ea (FloatTensor [N]):
                Atomic energy contributions.
            qa (FloatTensor [N]):
                Atomic partial charges.
            ea_rep (FloatTensor [N]):
                Atomic contributions to the ZBL inspired short-range repulsive
                potential. If self.use_zbl_repulsion is False, this is always
                zero.
            ea_ele (FloatTensor [N]):
                Atomic contributions to the point-charge electrostatics
                correction. If self.use_electrostatics is False, this is always
                zero.
            ea_vdw (FloatTensor [N]):
                Atomic contributions to Grimme's D4 dispersion correction. If
                self.use_d4_dispersion is False, this is always zero.
            pa (FloatTensor [N]):
                Atomic polarizabilities computed from Grimme's D4 method. If
                self.use_d4_dispersion or self.compute_d4_atomic is False, this
                is always zero.
            c6 (FloatTensor [N]):
                Atomic C6 coefficients computed from Grimme's D4 method. If
                self.use_d4_dispersion or self.compute_d4_atomic is False, this
                is always zero.
        """
        #print("Z device: ", Z.device)
        #print("Q device: ", Q.device)
        #print("S device: ", S.device)
        #print("R device: ", R.device)
        #print("idx_i device: ", idx_i.device)
        #print("idx_j device: ", idx_j.device)
        #print("batch_seg device: ", batch_seg.device)
        #print("num_batch: ", num_batch)

        (
            N,
            cutoff_values,
            rij,
            sr_rij,
            pij,
            dij,
            sr_idx_i,
            sr_idx_j,
            mask,
        ) = self._atomic_properties_static(
            Z=Z,
            R=R,
            idx_i=idx_i,
            idx_j=idx_j,
            cell=cell,
            cell_offsets=cell_offsets,
            num_batch=num_batch,
            batch_seg=batch_seg,
        )
        return self._atomic_properties_dynamic(
            N=N,
            Q=Q,
            S=S,
            Z=Z,
            R=R,
            cutoff_values=cutoff_values,
            rij=rij,
            idx_i=idx_i,
            idx_j=idx_j,
            sr_rij=sr_rij,
            pij=pij,
            dij=dij,
            sr_idx_i=sr_idx_i,
            sr_idx_j=sr_idx_j,
            cell=cell,
            num_batch=num_batch,
            batch_seg=batch_seg,
            mask=mask,
        )

    @torch.jit.export
    def energy(
        self,
        Z: torch.Tensor,
        Q: torch.Tensor,
        S: torch.Tensor,
        R: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
        cell_offsets: Optional[torch.Tensor] = None,
        num_batch: int = 1,
        batch_seg: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Computes the potential energy.
        B: Batch size (number of different molecules).

        Arguments:
            (see documentation of atomic_properties)
        Returns:
            energy (FloatTensor [B]):
                Potential energy of each molecule in the batch.
            
            (+ all return values of atomic_properties)
        """
        if batch_seg is None:  # assume a single batch
            batch_seg = Z.new_zeros(Z.size(0))
        (f, ea, qa, ea_rep, ea_ele, ea_vdw, pa, c6) = self.atomic_properties(
            Z=Z,
            Q=Q,
            S=S,
            R=R,
            idx_i=idx_i,
            idx_j=idx_j,
            cell=cell,
            cell_offsets=cell_offsets,
            num_batch=num_batch,
            batch_seg=batch_seg,
        )
        energy = ea.new_zeros(num_batch).index_add_(
            0, batch_seg, ea + ea_rep + ea_ele + ea_vdw
        )
        return (energy, f, ea, qa, ea_rep, ea_ele, ea_vdw, pa, c6)

    @torch.jit.export
    def energy_and_forces(
        self,
        Z: torch.Tensor,
        Q: torch.Tensor,
        S: torch.Tensor,
        R: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
        cell_offsets: Optional[torch.Tensor] = None,
        num_batch: int = 1,
        batch_seg: Optional[torch.Tensor] = None,
        create_graph: bool = False,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Computes the potential energy and forces.
        N: Number of atoms.
        B: Batch size (number of different molecules).

        Arguments:
            (see documentation of atomic_properties)

            create_graph (bool):
                If True, the computation graph is created for computing the
                forces (this is necessary if autograd needs to be run through
                the force computation, e.g. for computing the force loss).
        Returns:
            energy (FloatTensor [B]):
                Potential energy of each molecule in the batch.
            forces (FloatTensor [N, 3]):
                Forces acting on each atom.

            (+ all return values of atomic_properties)
        """
        (energy, f, ea, qa, ea_rep, ea_ele, ea_vdw, pa, c6) = self.energy(
            Z=Z,
            Q=Q,
            S=S,
            R=R,
            idx_i=idx_i,
            idx_j=idx_j,
            cell=cell,
            cell_offsets=cell_offsets,
            num_batch=num_batch,
            batch_seg=batch_seg,
        )

        if idx_i.numel() > 0:  # autograd will fail if there are no distances
            #print("sum energy: ", torch.sum(energy).requires_grad)
            #print(create_graph)
            #print("R: ", R.requires_grad)
            #print("R: ", R)

            #if R.requires_grad == False: 
                #print("R: ", R)
                # set to True to compute forces
                #R.requires_grad = True
            
            grad = torch.autograd.grad(
                [torch.sum(energy)], [R], 
                create_graph=create_graph,
            )[0]
            if grad is not None:  # necessary for torch.jit compatibility
                forces = -grad
            else:
                forces = torch.zeros_like(R)
        
        else:  # if there are no distances, the forces are zero
            forces = torch.zeros_like(R)
        return (energy, forces, f, ea, qa, ea_rep, ea_ele, ea_vdw, pa, c6)

    @torch.jit.export
    def energy_and_forces_and_hessian(
        self,
        Z: torch.Tensor,
        Q: torch.Tensor,
        S: torch.Tensor,
        R: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
        cell_offsets: Optional[torch.Tensor] = None,
        num_batch: int = 1,
        batch_seg: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Computes the potential energy, forces and the hessian.
        N: Number of atoms.
        B: Batch size (number of different molecules).

        Arguments:
            (see documentation of atomic_properties)

        Returns:
            energy (FloatTensor [B]):
                Potential energy of each molecule in the batch.
            forces (FloatTensor [N, 3]):
                Forces acting on each atom.
            hessian (FloatTensor [3*N, 3*N]):
                Hessian matrix. If more than one molecule is in the batch,
                the appropriate entries need to be collected from the matrix
                manually for each molecule.
            
            (+ all return values of atomic_properties)
        """
        (
            energy,
            forces,
            f,
            ea,
            qa,
            ea_rep,
            ea_ele,
            ea_vdw,
            pa,
            c6,
        ) = self.energy_and_forces(
            Z=Z,
            Q=Q,
            S=S,
            R=R,
            idx_i=idx_i,
            idx_j=idx_j,
            cell=cell,
            cell_offsets=cell_offsets,
            num_batch=num_batch,
            batch_seg=batch_seg,
            create_graph=True,
        )
        # The hessian is calculated by running autograd for each entry of the
        # gradient, so it scales 3N with the number of atoms N. This is quite
        # expensive, but unfortunately, there is no better way to do this until
        # torch implements forward mode automatic differentiation.
        grad = -forces.view(-1)
        s = grad.size(0)
        hessian = energy.new_zeros((s, s))
        if idx_i.numel() > 0:
            for idx in range(s):  # loop through entries of the hessian
                tmp = torch.autograd.grad([grad[idx]], [R], retain_graph=(idx < s))[0]
                if tmp is not None:  # necessary for torch.jit compatibility
                    hessian[idx] = tmp.view(-1)
        return energy, forces, hessian, f, ea, qa, ea_rep, ea_ele, ea_vdw, pa, c6

    def forward(
        self,
        Z: torch.Tensor,
        Q: torch.Tensor,
        S: torch.Tensor,
        R: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
        cell_offsets: Optional[torch.Tensor] = None,
        num_batch: int = 1,
        batch_seg: Optional[torch.Tensor] = None,
        create_graph: bool = True,
        use_forces: bool = True,
        use_dipole: bool = True,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Computes the total energy, forces, dipole moment vectors, and all atomic
        properties. Should be used for training (the graph is retained after
        the force calculation, so that autograd works when forces are used in
        the loss function).
        N: Number of atoms.
        B: Batch size (number of different molecules).

        Arguments:
            (see documentation of atomic_properties)

            create_graph (bool):
                If False, no graph is created when forces are calculated
                (must be True for using forces in loss functions).
            use_forces (bool):
                If False, skips force calculation and returns zeros.
            use_dipole (bool):
                If False, skips dipole calculation and returns zeros.

        Returns:
            energy (FloatTensor [B]):
                Potential energy of each molecule in the batch.
            forces (FloatTensor [N, 3]):
                Forces acting on each atom.
            dipole (FloatTensor [B, 3]):
                Dipole moment vector of each molecule in the batch.

            (+ all return values of atomic_properties)
        """
        if batch_seg is None:  # assume a single batch
            batch_seg = Z.new_zeros(Z.size(0))
        if use_forces:
            (
                energy,
                forces,
                f,
                ea,
                qa,
                ea_rep,
                ea_ele,
                ea_vdw,
                pa,
                c6,
            ) = self.energy_and_forces(
                Z=Z,
                Q=Q,
                S=S,
                R=R,
                idx_i=idx_i,
                idx_j=idx_j,
                cell=cell,
                cell_offsets=cell_offsets,
                num_batch=num_batch,
                batch_seg=batch_seg,
                create_graph=create_graph,
            )
        else:
            (energy, f, ea, qa, ea_rep, ea_ele, ea_vdw, pa, c6) = self.energy(
                Z=Z,
                Q=Q,
                S=S,
                R=R,
                idx_i=idx_i,
                idx_j=idx_j,
                cell=cell,
                cell_offsets=cell_offsets,
                num_batch=num_batch,
                batch_seg=batch_seg,
            )
            forces = torch.zeros_like(R)
        
        if use_dipole:
            dipole = qa.new_zeros((num_batch, 3)).index_add_(
                0, batch_seg, qa.view(-1, 1) * R
            )
        else:
            dipole = qa.new_zeros((num_batch, 3))

        return energy, forces, dipole, f, ea, qa, ea_rep, ea_ele, ea_vdw, pa, c6


    def loss_function(self):
        
        loss_function = {
            "E": torchmetrics.MeanSquaredError(),
            "F": torchmetrics.MeanSquaredError(),
        }  
        if self.dipole:
            loss_function["D"] = torchmetrics.MeanSquaredError()

        for key in loss_function.keys():
            loss_function[key].cuda()

        return loss_function
    

    def compute_loss(self, target, pred):

        loss = {}
        for key in target.keys():
            #print(key)
            loss[key] = self.loss[key](pred[key], target[key]) 

        return loss
    

    def shared_step(self, batch, mode):


        # batch dict
        N = batch["N"]
        N_atoms = len(batch["Z"])
        Z = batch["Z"]
        Q = batch["Q"]
        S = batch["S"]
        R = batch["R"]
        idx_i = batch["idx_i"]
        idx_j = batch["idx_j"]
        batch_seg = batch["batch_seg"]
        E = batch["E"]
        F = batch["F"]
        


        res_forces = self.energy_and_forces(
            Z=Z,
            Q=Q,
            S=S,
            R=R,
            idx_i=idx_i,
            idx_j=idx_j,
            batch_seg=batch_seg,
            num_batch=N,
            create_graph=True
            # use_dipole=True
        )

        
        dipole = res_forces[3].new_zeros((N, 3)).index_add_(
            0, batch_seg, res_forces[3].view(-1, 1) * R
        )
    
        E_pred = res_forces[0]
        F_pred = res_forces[1]
        

        target_dict = {
            "E": E,
            "F": F,
        }
        
        pred_dict = {
            "E": E_pred,
            "F": F_pred,
        }

        if self.dipole:
            target_dict["D"] = batch["D"]
            pred_dict["D"] = dipole
        
        all_loss = self.compute_loss(pred_dict, target_dict)
        self.update_metrics(target_dict, pred_dict, mode)
    
        # this might need to be changed for custom loss fxns
        
        for key in all_loss.keys():
            sum_loss = all_loss[key] * self.loss_weights[key]
        all_loss["loss"] = sum_loss


        show_prog = False
        if mode == "train":
            show_prog = True
            

        self.log(
            f"{mode}_loss", 
            all_loss["loss"],             
            on_step=False,
            on_epoch=True,
            prog_bar=show_prog,
            batch_size=N,
            sync_dist=True,
            rank_zero_only=True
        )

        loss_E_l1 = torch.nn.L1Loss(reduction="sum")
        E_per_atom = 1000 * loss_E_l1(target_dict["E"], pred_dict["E"]) / N_atoms

        self.log(
            f"{mode}_E_l1_per_atom",
            E_per_atom ,
            on_step=False,
            on_epoch=True,
            prog_bar=not show_prog,
            batch_size=N,
            sync_dist=True,
            rank_zero_only=True
        )
        return all_loss
    

    def training_step(self, batch): 
        torch.set_grad_enabled(True)
        return self.shared_step(batch, mode="train")


    def validation_step(self, batch):
        torch.set_grad_enabled(True)
        return self.shared_step(batch, mode="val")


    def test_step(self, batch):
        torch.set_grad_enabled(True)
        return self.shared_step(batch, mode="test")


    def on_train_epoch_start(self):
        self.start_time = time.time()


    def on_train_epoch_end(self):
        
        if self.dipole: 
            e_l1, e_rmse, f_l1, f_rmse, d_l1 = self.compute_metrics("train")
            self.log("train_D_l1", d_l1, sync_dist=True)
            #self.log("train_D_rmse", d_rmse, prog_bar=True, sync_dist=True)
            
        else:
            e_l1, e_rmse, f_l1, f_rmse = self.compute_metrics("train")

        self.log("train_E_l1", e_l1, sync_dist=True)
        self.log("train_E_rmse", e_rmse, sync_dist=True)
        self.log("train_F_l1", f_l1, sync_dist=True)
        self.log("train_F_rmse", f_rmse, prog_bar=True, sync_dist=True)
               
        duration = time.time() - self.start_time
        
        self.log("epoch_time", 
                    duration, 
                on_epoch=True,
                prog_bar=True, 
                logger=True,
                sync_dist=True, 
                rank_zero_only=True
        ) 


    def on_validation_epoch_end(self):

        if self.dipole: 
            e_l1, e_rmse, f_l1, f_rmse, d_l1 = self.compute_metrics("val")
            self.log("val_D_l1", d_l1, sync_dist=True)
            #self.log("val_D_rmse", d_rmse, prog_bar=True, sync_dist=True)

        else: 
            e_l1, e_rmse, f_l1, f_rmse = self.compute_metrics("val")
        
        self.log("val_E_l1", e_l1, sync_dist=True)
        self.log("val_E_rmse", e_rmse, sync_dist=True)
        self.log("val_F_l1", f_l1, sync_dist=True)
        self.log("val_F_rmse", f_rmse, prog_bar=True, sync_dist=True)


    def on_test_epoch_end(self):
        if self.dipole: 
            e_l1, e_rmse, f_l1, f_rmse, d_l1 = self.compute_metrics("test")

            self.log("test_D_l1", d_l1, sync_dist=True)
            #self.log("test_D_rmse", d_rmse, prog_bar=True, sync_dist=True)

        else:
            e_l1, e_rmse, f_l1, f_rmse = self.compute_metrics("test")
        self.log("test_E_l1", e_l1, sync_dist=True)
        self.log("test_E_rmse", e_rmse, sync_dist=True)
        self.log("test_F_l1", f_l1, sync_dist=True)
        self.log("test_F_rmse", f_rmse, prog_bar=True, sync_dist=True)


    def update_metrics(self, pred, target, mode):
        """
        Update metrics using torchmetrics interfaces
        """

        if mode == "train":
            self.train_E_l1.update(pred["E"], target["E"])
            self.train_E_rl2.update(pred["E"], target["E"])
            self.train_F_l1.update(pred["F"], target["F"])
            self.train_F_rl2.update(pred["F"], target["F"])
            
            #if N_atoms is not None:
            #    self.train_atom_count.update(N_atoms)

            if self.dipole:
                self.train_D_l1.update(pred["D"], target["D"])
                
        
        elif mode == "val":
            self.val_E_l1.update(pred["E"], target["E"])
            self.val_E_rl2.update(pred["E"], target["E"])
            self.val_F_l1.update(pred["F"], target["F"])
            self.val_F_rl2.update(pred["F"], target["F"])
            #if N_atoms is not None:
            #    self.val_atom_count.update(N_atoms)

            if self.dipole:
                self.val_D_l1.update(pred["D"], target["D"])
                

        elif mode == "test":
            self.test_E_l1.update(pred["E"], target["E"])
            self.test_E_rl2.update(pred["E"], target["E"])
            self.test_F_l1.update(pred["F"], target["F"])
            self.test_F_rl2.update(pred["F"], target["F"])
            
            #if N_atoms is not None:
            #    self.test_atom_count.update(N_atoms)

            if self.dipole:
                self.test_D_l1.update(pred["D"], target["D"])
                


    def compute_metrics(self, mode):
        """
        Compute metrics using torchmetrics interfaces
        """

        if mode == "train":
            E_torch_l1 = self.train_E_l1.compute()
            E_torch_rmse = self.train_E_rl2.compute()
            F_torch_l1 = self.train_F_l1.compute()
            F_torch_rmse = self.train_F_rl2.compute()
            if self.dipole:
                D_torch_l1 = self.train_D_l1.compute()
                #D_torch_rmse = self.train_D_rl2.compute()

            #count_atoms = self.train_atom_count.compute()

        elif mode == "val":
            E_torch_l1 = self.val_E_l1.compute()
            E_torch_rmse = self.val_E_rl2.compute()
            F_torch_l1 = self.val_F_l1.compute()
            F_torch_rmse = self.val_F_rl2.compute()
            if self.dipole:
                D_torch_l1 = self.val_D_l1.compute()
                #D_torch_rmse = self.val_D_rl2.compute()

            #count_atoms = self.val_atom_count.compute()

        elif mode == "test":
            E_torch_l1 = self.test_E_l1.compute()
            E_torch_rmse = self.test_E_rl2.compute()
            F_torch_l1 = self.test_F_l1.compute()
            F_torch_rmse = self.test_F_rl2.compute()
            if self.dipole:
                D_torch_l1 = self.test_D_l1.compute()
                #D_torch_rmse = self.test_D_rl2.compute()

            #count_atoms = self.test_atom_count.compute()
        
        if self.dipole:
            return E_torch_l1, E_torch_rmse, F_torch_l1, F_torch_rmse, D_torch_l1
        else:     
            return E_torch_l1, E_torch_rmse, F_torch_l1, F_torch_rmse
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            amsgrad=True,
            lr=self.hparams.learning_rate,
            #weight_decay=self.hparams.weight_decay,
        )

        scheduler = self._config_lr_scheduler(optimizer)

        lr_scheduler = {"scheduler": scheduler, "monitor": "val_loss"}

        return [optimizer], [lr_scheduler]
    

    def _config_lr_scheduler(self, optimizer):
        scheduler_name = self.hparams["scheduler_name"].lower()

        if scheduler_name == "reduce_on_plateau":
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=0.5,
                patience=self.hparams.lr_patience,
                threshold=0
            )

        elif scheduler_name == "none":
            scheduler = None
        else:
            raise ValueError(f"Not supported lr scheduler: {scheduler_name}")

        return scheduler