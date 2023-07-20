import itertools
from functools import cached_property
from typing import Callable, Sequence, Tuple

import attr
import numpy as np
from cirq_ft.algos import (QROM, LessThanEqualGate, MultiTargetCSwap,
                           PrepareUniformSuperposition,
                           StatePreparationAliasSampling, select_and_prepare)
from cirq_ft.infra import Registers, SelectionRegisters
from cirq_ft.linalg.lcu_util import \
    preprocess_lcu_coefficients_for_reversible_sampling
from numpy.typing import ArrayLike, NDArray

import cirq


@cirq.value_equality()
@attr.frozen
class InnerPrepareQROM(select_and_prepare.PrepareOracle):
    r"""
    Args:

    Parameters:

    References:
    """

    num_spatial: int
    altp: Sequence[NDArray[np.int_]]
    altq: Sequence[NDArray[np.int_]]
    keep: Sequence[NDArray[np.int_]]
    mu: Sequence[int]

    @classmethod
    def build_from_coefficients(
        cls,
        Wlpq: NDArray[np.float64],
        *,
        probability_epsilon: float = 1.0e-5,
    ) -> 'InnerPrepareQROM':
        r"""Factory method to build In_l-prep_l from SF coefficients
        Args:
        Returns:
        """
        num_spatial = Wlpq.shape[-1]
        altp_l, altq_l, keep_l, mu_l = [], [], [], []
        for Wpq in Wlpq:
            alt, keep, mu = preprocess_lcu_coefficients_for_reversible_sampling(
                lcu_coefficients=Wpq.ravel(), epsilon=probability_epsilon
            )
            altp, altq = np.unravel_index(alt, (num_spatial, num_spatial))
            altp_l.append(altp.reshape((num_spatial, num_spatial)))
            altq_l.append(altq.reshape((num_spatial, num_spatial)))
            keep_l.append(np.array(keep).reshape((num_spatial, num_spatial)))
            mu_l.append(mu)
        return InnerPrepareQROM(
            num_spatial,
            altp_l,
            altq_l,
            keep_l,
            mu_l,
        )

    @cached_property
    def selection_registers(self) -> SelectionRegisters:
        num_spatial = self.num_spatial
        num_aux = len(self.altp)
        regs = SelectionRegisters.build(
            l=((num_aux-1).bit_length(), num_aux),
            p=((num_spatial-1).bit_length(), num_spatial),
            q=((num_spatial-1).bit_length(), num_spatial),
        )
        return regs

    @cached_property
    def sigma_mu_bitsize(self) -> int:
        return self.mu[0]

    @cached_property
    def alternates_bitsize(self) -> int:
        return sum(reg.bitsize for reg in self.selection_registers) + 1

    @cached_property
    def keep_bitsize(self) -> int:
        return self.mu[0]

    @cached_property
    def junk_registers(self) -> Registers:
        num_spatial = self.num_spatial
        return Registers.build(
            altp=(num_spatial - 1).bit_length(),
            altq=(num_spatial - 1).bit_length(),
            keep=self.keep_bitsize,
            sigma_mu=self.sigma_mu_bitsize,
            less_than_equal=1,
        )

    # @cached_property
    # def theta_register(self) -> Registers:
    #     return Registers.build(theta=1)

    @cached_property
    def registers(self) -> Registers:
        return Registers([*self.selection_registers, *self.junk_registers])

    def decompose_from_registers(
        self, context: cirq.DecompositionContext, **quregs: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        l, p, q = quregs['l'], quregs['p'], quregs['q']
        altp, altq = quregs["altp"], quregs["altq"]
        keep = quregs["keep"]
        less_than_equal = quregs["less_than_equal"]
        sigma_mu = quregs["sigma_mu"]
        yield PrepareUniformSuperposition(n=self.num_spatial*self.num_spatial).on_registers(controls=[], target=[*p, *q])
        yield cirq.H.on_each(*sigma_mu)
        qrom = QROM.build(self.altp, self.altq, self.keep)
        yield qrom.on_registers(
            selection0=l,
            selection1=p,
            selection2=q,
            target0=altp,
            target1=altq,
            target2=keep,
        )
        yield LessThanEqualGate(self.mu[0], self.mu[0]).on(*keep, *sigma_mu, *less_than_equal)
        yield MultiTargetCSwap.make_on(
            control=less_than_equal, target_x=[*altp, *altq], target_y=[*p, *q]
        )
        yield LessThanEqualGate(self.mu[0], self.mu[0]).on(*keep, *sigma_mu, *less_than_equal)

    def _value_equality_values_(self):
        return self.registers

@cirq.value_equality()
@attr.frozen
class SingleFactorizationBlockEncoding(select_and_prepare.PrepareOracle):
    r"""
    Args:

    Parameters:

    References:
    """

    num_aux: int
    num_spatial: int
    Wlpq: NDArray
    Tpq: NDArray
    mu: float = 8

    @cached_property
    def registers(self) -> Registers:
        return Registers([*self.selection_registers, *self.junk_registers])

    @cached_property
    def selection_registers(self) -> SelectionRegisters:
        return SelectionRegisters.build(
            succ_l=(1, 2),
            l=((self.num_aux + 1 - 1).bit_length(), self.num_aux),
            l_ne_0=(1, 2),
            succ_pq=(1, 2),
            p=((self.num_spatial - 1).bit_length(), self.num_spatial),
            q=((self.num_spatial - 1).bit_length(), self.num_spatial),
            ctrl_pq=(1, 2),
            alpha=(1, 2),
        )

    @cached_property
    def target_registers(self) -> Registers:
        return Registers.build(psi_up=self.num_spatial, psi_dn=self.num_spatial)


    def decompose_from_registers(
        self, context: cirq.DecompositionContext, **quregs: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        p, q, l = quregs['p'], quregs['q'], quregs['l']
        succ_l = quregs['succ_l']
        succ_pq = quregs['succ_pq']
        ctrl_pq = quregs['ctrl_pq']
        alpha = quregs['alpha']
        c_0 = [np.sum(np.abs(self.Tpq))]
        # ignore p <= q condition for the moment for simplicity
        c_l = np.einsum("lpq->l", np.abs(self.Wlpq))
        coeffs = np.concatenate((c_0, c_l))
        # outer prepare
        outer_prep = StatePreparationAliasSampling.from_lcu_probs(coeffs/np.sum(coeffs), probability_epsilon=2**(-self.mu) / len(coeffs))
        outer_prep_anc = {
            reg.name: context.qubit_manager.qalloc(reg.bitsize) for reg in outer_prep.junk_registers
        }
        outer_prep_sel = {
            reg.name: l for ir, reg in enumerate(outer_prep.selection_registers)
        }
        yield outer_prep.on_registers(**outer_prep_sel, **outer_prep_anc)
        inner_prep = InnerPrepareQROM.build_from_coefficients(self.Wlpq, probability_epsilon=2**(-self.mu) / len(self.Wlpq[0]))
        inner_prep_anc = {
            reg.name: context.qubit_manager.qalloc(reg.bitsize) for reg in inner_prep.junk_registers
        }
        yield outer_prep.on_registers(**quregs, **inner_prep_anc)


    def _value_equality_values_(self):
        return self.registers