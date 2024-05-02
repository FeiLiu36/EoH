from typing import Union, Tuple, Optional, Any
from typing_extensions import Literal
import numpy as np
import eagerpy as ep
import logging
import torch
import time
import math

from foolbox.devutils import flatten
from foolbox.devutils import atleast_kd

from foolbox.types import Bounds

from foolbox.models import Model

from foolbox.criteria import Criterion

from foolbox.distances import l2

from foolbox.tensorboard import TensorBoard

from foolbox.attacks.blended_noise import LinearSearchBlendedUniformNoiseAttack

from foolbox.attacks.base import MinimizationAttack
from foolbox.attacks.base import T
from foolbox.attacks.base import get_criterion
from foolbox.attacks.base import get_is_adversarial
from foolbox.attacks.base import raise_if_kwargs
from foolbox.attacks.base import verify_input_bounds



class EvoAttack(MinimizationAttack):
    distance = l2
    # can only perform l2 attacks
    def __init__(
        self,
        library_use,
        init_attack: Optional[MinimizationAttack] = None,
        steps: int = 25000,
        min_epsilon: float = 0.0,
    ):
        if init_attack is not None and not isinstance(init_attack, MinimizationAttack):
            raise NotImplementedError
        self.init_attack = init_attack
        self.steps = steps
        self.min_epsilon = min_epsilon
        self.library_use = library_use

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Criterion, T],
        *,
        early_stop: Optional[float] = None,
        starting_points: Optional[T] = None,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        originals, restore_type = ep.astensor_(inputs)
        del inputs, kwargs

        verify_input_bounds(originals, model)

        criterion = get_criterion(criterion)
        is_adversarial = get_is_adversarial(criterion, model)

        if starting_points is None:
            init_attack: MinimizationAttack
            if self.init_attack is None:
                init_attack = LinearSearchBlendedUniformNoiseAttack(steps=50)
                logging.info(
                    f"Neither starting_points nor init_attack given. Falling"
                    f" back to {init_attack!r} for initialization."
                )
            else:
                init_attack = self.init_attack
            best_advs = init_attack.run(
                model, originals, criterion, early_stop=early_stop
            )
        else:
            best_advs = ep.astensor(starting_points)

        is_adv = is_adversarial(best_advs)
        if not is_adv.all():
            failed = is_adv.logical_not().float32().sum()
            if starting_points is None:
                raise ValueError(
                    f"init_attack failed for {failed} of {len(is_adv)} inputs"
                )
            else:
                raise ValueError(
                    f"{failed} of {len(is_adv)} starting_points are not adversarial"
                )
        del starting_points

        N = len(originals)
        ndim = originals.ndim
        min_, max_ = model.bounds
        np_bounds = np.array([min_, max_])

        # converting back to torch
        # candidate_history = np.zeros((100, *originals.shape))
        # adv_history = np.zeros((100, originals.shape[0]))
        # history_idx = 0
        t1 = time.time()

        self.hyperparams = 0.001 * np.ones(originals.shape[0])
        alpha_p = 0.95
        p = np.zeros(originals.shape[0])
        for step in range(1, self.steps + 1):
            originals = originals.raw
            best_advs = best_advs.raw

            orginals_np = originals.cpu().numpy()
            best_advs_np = best_advs.cpu().numpy()

            candidates_np = np.zeros(orginals_np.shape)
            standard_noise_np = np.random.normal(size=orginals_np.shape)
            for candidate_i in range(orginals_np.shape[0]):
                candidates_np_i = self.library_use.draw_proposals(
                    orginals_np[candidate_i],
                    best_advs_np[candidate_i],
                    standard_noise_np[candidate_i],
                    self.hyperparams[candidate_i:candidate_i + 1],
                )
                candidates_np[candidate_i] = candidates_np_i

            candidates_np = np.clip(candidates_np, np_bounds[0], np_bounds[1])

            t2 = time.time()
            if t2 - t1 > 120:
                return best_advs
            candidates = torch.from_numpy(candidates_np).float().to(originals.device)

            is_adv = is_adversarial(candidates)
            # is_adv_np = is_adv.cpu().numpy()
            # if history_idx == 100:
            #     candidate_history = np.concatenate([candidate_history[:-1], candidates_np[None, :]])
            #     adv_history = np.concatenate([adv_history[:-1], is_adv_np[None, :]])
            #     history_idx = 99
            # else:
            #     candidate_history[history_idx] = candidates_np
            #     adv_history[history_idx] = is_adv_np
            # history_idx += 1
            originals = ep.astensor(originals)
            candidates = ep.astensor(candidates)

            is_adv = ep.astensor(is_adv)
            best_advs = ep.astensor(best_advs)

            distances = ep.norms.l2(flatten(originals - candidates), axis=-1)
            source_norms = ep.norms.l2(flatten(originals - best_advs), axis=-1)
            closer = distances < source_norms
            is_best_adv = ep.logical_and(is_adv, closer)
            is_best_adv = atleast_kd(is_best_adv, ndim)

            # update params
            is_best_adv_np = is_best_adv.raw.cpu().numpy()
            is_best_adv_np = is_best_adv_np.astype(float).reshape(-1)
            p = alpha_p * p + (1 - alpha_p) * is_best_adv_np
            self.hyperparams *= np.power(self._f_p(p), 1 / 10)

            best_advs = ep.where(is_best_adv, candidates, best_advs)

            self.current_epsilons = ep.norms.l2(flatten(best_advs - originals), axis=-1)
            if (self.current_epsilons < self.min_epsilon).all():
                return restore_type(best_advs)
            # print("Step {}: {:.5f}".format(step, self.current_epsilons.mean()))
        print("Step {}: {:.5f}".format(self.steps, self.current_epsilons.mean()))

        return restore_type(best_advs)

    def _f_p(self, p):
        # piecewise linear function
        # f(0)=l, f(1)=h, f(0.25)=1
        l = 0.5
        h = 1.5
        p_threshold = 0.25
        p_less_idx = p < p_threshold
        p_greater_idx = p >= p_threshold
        f_p = np.zeros_like(p)
        f_p[p_less_idx] = l + (1 - l) * p[p_less_idx] / p_threshold
        f_p[p_greater_idx] = 1 + (h - 1) * (p[p_greater_idx] - p_threshold) / (
            1 - p_threshold
        )
        return f_p
