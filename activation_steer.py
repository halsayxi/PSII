import torch
from typing import Sequence, Union, Iterable, Optional


class ActivationSteerer:
    """
    Handles standard steering vectors and specific value vector injection.
    """

    _POSSIBLE_LAYER_ATTRS: Iterable[str] = (
        "transformer.h",
        "encoder.layer",
        "model.layers",
        "gpt_neox.layers",
        "block",
    )

    def __init__(
        self,
        model: torch.nn.Module,
        # Demographic vector
        steering_vector: Optional[Union[torch.Tensor, Sequence[float]]] = None,
        *,
        coeff: Optional[float] = 1.0,
        layer_idx: Optional[int] = -1,
        positions: Optional[str] = "all",
        noise_std: Optional[float] = 0.0,
        # Value vector
        value_vector: Optional[torch.Tensor] = None,
        alpha: float = 0.1,
        sigma_noise: float = 0.01,
        debug: bool = False,
    ):
        self.model = model
        self.alpha, self.sigma_noise = float(alpha), float(sigma_noise)
        self.debug = debug
        self._handles = []
        self.coeff = float(coeff) if coeff is not None else 1.0
        self.layer_idx = int(layer_idx) if layer_idx is not None else -1
        self.positions = positions.lower() if positions is not None else "all"
        self.noise_std = float(noise_std) if noise_std is not None else 0.0

        p = next(model.parameters())
        hidden = getattr(model.config, "hidden_size", None)

        # Setup Steering Vector
        if steering_vector is not None:
            self.vector = torch.as_tensor(
                steering_vector, dtype=p.dtype, device=p.device
            )
            if self.vector.ndim != 1:
                raise ValueError("steering_vector must be 1‑D")
            if hidden and self.vector.numel() != hidden:
                raise ValueError(
                    f"Vector length {self.vector.numel()} ≠ model hidden_size {hidden}"
                )
            valid_positions = {"all", "prompt", "response"}
            if self.positions not in valid_positions:
                raise ValueError("positions must be 'all', 'prompt', 'response'")
        else:
            self.vector = None

        # Setup Value Vector
        if value_vector is not None:
            self.ml_vector = value_vector.to(dtype=p.dtype, device=p.device)
        else:
            self.ml_vector = None

    def _locate_layer(self, layer_idx=None):
        if layer_idx is None:
            layer_idx = self.layer_idx
        for path in self._POSSIBLE_LAYER_ATTRS:
            cur = self.model
            for part in path.split("."):
                if hasattr(cur, part):
                    cur = getattr(cur, part)
                else:
                    break
            else:
                # if hasattr(cur, "__len__"):
                #     print(f"[ActivationSteerer] Found layers at `{path}`")
                #     print(f"[ActivationSteerer] Number of layers: {len(cur)}")
                if not hasattr(cur, "__getitem__"):
                    continue
                if not (-len(cur) <= layer_idx < len(cur)):
                    raise IndexError(f"layer_idx{layer_idx} out of range")
                if self.debug:
                    print(f"[ActivationSteerer] hooking {path}[{layer_idx}]")
                return cur[layer_idx]

        raise ValueError("Could not find layer list on the model.")

    def _hook_fn(self, module, ins, out):
        if self.vector is None:
            return out

        # Prepare standard steering addition
        steer = self.coeff * self.vector
        if self.noise_std > 0:
            steer += torch.randn_like(steer) * self.noise_std

        def _process(t):
            t2 = t.clone()
            # Standard steering injection (Applied based on positions)
            s = steer.to(t.device)
            if not hasattr(self, "_prompt_done"):
                self._prompt_done = False
            if self.positions == "prompt":
                if not self._prompt_done:
                    t2[:, :, :] += s
                    self._prompt_done = True
            elif self.positions == "response":
                t2[:, -1, :] += s
            elif self.positions == "all":
                if not self._prompt_done:
                    t2[:, :, :] += s
                    self._prompt_done = True
                else:
                    t2[:, -1, :] += s
            return t2

        if torch.is_tensor(out):
            return _process(out)
        elif isinstance(out, (tuple, list)) and torch.is_tensor(out[0]):
            return (_process(out[0]), *out[1:])
        return out

    def _hook_fn_ml(self, module, ins, out):
        if self.ml_vector is None:
            return out

        # Prepare value addition: alpha * (e_lang + sigma * noise)
        noise = (
            torch.randn_like(self.ml_vector) * self.sigma_noise
            if self.sigma_noise > 0
            else torch.zeros_like(self.ml_vector)
        )
        ml_add = self.alpha * (self.ml_vector + noise)

        def _process(t):
            t2 = t.clone()
            # Value injection
            s = ml_add.to(t.device)
            t2[:, -1, :] += s
            return t2

        if torch.is_tensor(out):
            return _process(out)
        elif isinstance(out, (tuple, list)) and torch.is_tensor(out[0]):
            return (_process(out[0]), *out[1:])
        return out

    def __enter__(self):
        if self.vector is not None:
            layer = self._locate_layer(self.layer_idx)
            self._handles.append(layer.register_forward_hook(self._hook_fn))
        if self.ml_vector is not None:
            last_layer = self._locate_layer(-1)
            self._handles.append(last_layer.register_forward_hook(self._hook_fn_ml))

        return self

    def __exit__(self, *exc):
        self.remove()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        for attr in ["_prompt_done"]:
            if hasattr(self, attr):
                delattr(self, attr)


class ActivationSteererMultiple:
    """
    Manages multiple steerers and an optional global value vector.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        instructions: Sequence[dict] = [],
        *,
        value_vector: Optional[torch.Tensor] = None,
        alpha: float = 0.1,
        sigma_noise: float = 0.01,
        debug: bool = False,
    ):
        self.model = model
        self.instructions = instructions
        self.debug = debug
        self._steerers = []

        # Process regular steering instructions (Demographic steering)
        # These steerers use the layer_idx provided in their respective dicts.
        for inst in self.instructions:
            self._steerers.append(
                ActivationSteerer(
                    model,
                    steering_vector=inst.get("steering_vector"),
                    coeff=inst.get("coeff", 0.0),
                    layer_idx=inst.get("layer_idx", -1),
                    positions=inst.get("positions", "all"),
                    noise_std=inst.get("noise_std", 0.0),
                    debug=debug,
                )
            )

        # Setup the Value Steering module
        # This module ignores global layer_idx and is hardcoded to the last layer (-1).
        if value_vector is not None:
            if self.debug:
                print(
                    "[ActivationSteererMultiple] Initializing global value module for layer -1"
                )

            # Create an ActivationSteerer where steering_vector is None,
            # but value_vector is provided.
            self._steerers.append(
                ActivationSteerer(
                    model,
                    steering_vector=None,
                    value_vector=value_vector,
                    alpha=alpha,
                    sigma_noise=sigma_noise,
                    debug=debug,
                )
            )

    def __enter__(self):
        for steerer in self._steerers:
            steerer.__enter__()
        return self

    def __exit__(self, *exc):
        self.remove()

    def remove(self):
        for steerer in self._steerers:
            steerer.remove()
