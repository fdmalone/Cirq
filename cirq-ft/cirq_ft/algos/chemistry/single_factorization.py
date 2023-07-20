from functools import cached_property
from typing import Callable, Sequence, Tuple

import attr
import numpy as np
from cirq_ft.algos import (
    QROM,
    LessThanEqualGate,
    MultiTargetCSwap,
    PrepareUniformSuperposition,
    select_and_prepare,
    unary_iteration_gate
)
from cirq_ft.infra import Registers, SelectionRegisters
from cirq_ft.linalg.lcu_util import preprocess_lcu_coefficients_for_reversible_sampling
from numpy.typing import NDArray, ArrayLike

import itertools

import cirq

@cirq.value_equality()
@attr.frozen
class LthInnerPrepare(select_and_prepare.PrepareOracle):
    r"""
    Args:

    Parameters:

    References:
    """

    num_spatial: int
    altp: NDArray[np.int_]
    altq: NDArray[np.int_]
    keep: NDArray[np.int_]
    mu: int

    @classmethod
    def build_from_coefficients(
        cls,
        Wpq: NDArray[np.float64],
        *,
        probability_epsilon: float = 1.0e-5,
    ) -> 'LthInnerPrepare':
        r"""
        Args:
        Returns:
        """
        num_spatial = Wpq.shape[-1]
        alt, keep, mu = preprocess_lcu_coefficients_for_reversible_sampling(
            lcu_coefficients=Wpq.ravel(), epsilon=probability_epsilon
        )
        altp, altq = np.unravel_index(alt, (num_spatial, num_spatial))
        return LthInnerPrepare(
            altp,
            altq,
            keep,
            mu,
        )

    @cached_property
    def selection_registers(self) -> SelectionRegisters:
        num_spatial = self.altUV.shape[-1]
        regs = SelectionRegisters.build(
            p=((num_spatial-1).bit_length(), num_spatial),
            q=((num_spatial-1).bit_length(), num_spatial),
            succ_pq=(1,),
        )
        return regs

    @cached_property
    def sigma_mu_bitsize(self) -> int:
        return self.mu

    @cached_property
    def alternates_bitsize(self) -> int:
        return sum(reg.bitsize for reg in self.selection_registers) + 1

    @cached_property
    def keep_bitsize(self) -> int:
        return self.mu

    @cached_property
    def junk_registers(self) -> Registers:
        num_spatial = self.altp[0].shape[-1]
        return Registers.build(
            altp=(num_spatial - 1).bit_length(),
            altq=(num_spatial - 1).bit_length(),
            keep=self.keep_bitsize,
            sigma_mu=self.sigma_mu_bitsize,
            less_than_equal=1,
        )

    @cached_property
    def theta_register(self) -> Registers:
        return Registers.build(theta=1)

    @cached_property
    def registers(self) -> Registers:
        return Registers([*self.selection_registers, *self.junk_registers])

    def decompose_from_registers(
        self, context: cirq.DecompositionContext, **quregs: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        p, q = quregs['p'], quregs['q']
        altp, altq = quregs["altp"], quregs["altq"]
        keep = quregs["keep"]
        less_than_equal = quregs["less_than_equal"]
        sigma_mu = quregs["sigma_mu"]
        yield PrepareUniformSuperposition(n=self.num_spatial*self.num_spatial).on_registers(controls=[], target=[p, q])
        yield cirq.H.on_each(*sigma_mu)
        qrom = QROM.build(self.altp, *self.altq, self.keep)
        yield qrom.on_registers(
            selection0=p,
            selection1=q,
            target0=altp,
            target1=altq,
            target2=keep,
        )
        yield LessThanEqualGate(self.mu, self.mu).on(*keep, *sigma_mu, *less_than_equal)
        yield MultiTargetCSwap.make_on(
            control=less_than_equal, target_x=[*altp, *altq], target_y=[*p, *q]
        )
        yield LessThanEqualGate(self.mu, self.mu).on(*keep, *sigma_mu, *less_than_equal)

    def _value_equality_values_(self):
        return self.registers

@cirq.value_equality()
@attr.frozen
class InnerPrepare(unary_iteration_gate.UnaryIterationGate):
    """Gate to load data[l] in the target register when the selection stores an index l.
    """

    Wlpq: Sequence[NDArray]
    selection_bitsizes: Tuple[int, ...]
    target_bitsizes: Tuple[int, ...]
    num_controls: int = 0

    @classmethod
    def build(cls, *data: ArrayLike, num_controls: int = 0) -> 'QROM':
        _data = [np.array(d, dtype=int) for d in data]
        selection_bitsizes = tuple((s - 1).bit_length() for s in _data[0].shape)
        target_bitsizes = tuple(max(int(np.max(d)).bit_length(), 1) for d in data)
        return QROM(
            data=_data,
            selection_bitsizes=selection_bitsizes,
            target_bitsizes=target_bitsizes,
            num_controls=num_controls,
        )

    def __attrs_post_init__(self):
        shapes = [d.shape for d in self.data]
        assert all([isinstance(s, int) for s in self.selection_bitsizes])
        assert all([isinstance(t, int) for t in self.target_bitsizes])
        assert len(set(shapes)) == 1, f"Data must all have the same size: {shapes}"
        assert len(self.target_bitsizes) == len(self.data), (
            f"len(self.target_bitsizes)={len(self.target_bitsizes)} should be same as "
            f"len(self.data)={len(self.data)}"
        )
        assert all(
            t >= int(np.max(d)).bit_length() for t, d in zip(self.target_bitsizes, self.data)
        )
        assert isinstance(self.selection_bitsizes, tuple)
        assert isinstance(self.target_bitsizes, tuple)

    @cached_property
    def control_registers(self) -> Registers:
        return (
            Registers.build(control=self.num_controls)
            if self.num_controls
            else Registers([])
        )

    @cached_property
    def selection_registers(self) -> SelectionRegisters:
        if len(self.data[0].shape) == 1:
            return SelectionRegisters.build(
                selection=(self.selection_bitsizes[0], self.data[0].shape[0])
            )
        else:
            return SelectionRegisters.build(
                **{
                    f'selection{i}': (sb, len)
                    for i, (len, sb) in enumerate(zip(self.data[0].shape, self.selection_bitsizes))
                }
            )

    @cached_property
    def target_registers(self) -> Registers:
        return Registers.build(
            **{f'target{i}': len for i, len in enumerate(self.target_bitsizes)}
        )

    def __repr__(self) -> str:
        data_repr = f"({','.join(cirq._compat.proper_repr(d) for d in self.data)})"
        selection_repr = repr(self.selection_bitsizes)
        target_repr = repr(self.target_bitsizes)
        return (
            f"cirq_ft.QROM({data_repr}, selection_bitsizes={selection_repr}, "
            f"target_bitsizes={target_repr}, num_controls={self.num_controls})"
        )

    def _load_nth_data(
        self,
        selection_idx: Tuple[int, ...],
        gate: Callable[[cirq.Qid], cirq.Operation],
        **target_regs: Sequence[cirq.Qid],
    ) -> cirq.OP_TREE:
        for i, d in enumerate(self.data):
            target = target_regs[f'target{i}']
            for q, bit in zip(target, f'{int(d[selection_idx]):0{len(target)}b}'):
                if int(bit):
                    yield gate(q)

    def decompose_zero_selection(
        self, context: cirq.DecompositionContext, **quregs: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        controls = self.control_registers.merge_qubits(**quregs)
        target_regs = {k: v for k, v in quregs.items() if k in self.target_registers}
        zero_indx = (0,) * len(self.data[0].shape)
        yield self._lth_prepare(zero_indx, cirq.X, **target_regs)

    def nth_operation(
        self, context: cirq.DecompositionContext, control: cirq.Qid, **kwargs
    ) -> cirq.OP_TREE:
        selection_idx = tuple(kwargs[reg.name] for reg in self.selection_registers)
        target_regs = {k: v for k, v in kwargs.items() if k in self.target_registers}
        yield self._load_nth_data(selection_idx, lambda q: cirq.CNOT(control, q), **target_regs)

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["@"] * self.num_controls
        wire_symbols += ["In"] * self.selection_registers.bitsize
        for i, target in enumerate(self.target_registers):
            wire_symbols += [f"QROM_{i}"] * target.bitsize
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def __pow__(self, power: int):
        if power in [1, -1]:
            return self
        return NotImplemented  # coverage: ignore

    def _value_equality_values_(self):
        return (self.selection_registers, self.target_registers, self.control_registers)

@cirq.value_equality()
@attr.frozen
class InnerPrepareQROM(select_and_prepare.PrepareOracle):
    r"""
    Args:

    Parameters:

    References:
    """

    num_spatial: int
    altp: NDArray[np.int_]
    altq: NDArray[np.int_]
    keep: NDArray[np.int_]
    mu: int

    @classmethod
    def build_from_coefficients(
        cls,
        Wlpq: NDArray[np.float64],
        *,
        probability_epsilon: float = 1.0e-5,
    ) -> 'LthInnerPrepare':
        r"""
        Args:
        Returns:
        """
        num_spatial = Wlpq.shape[-1]
        num_aux = Wlpq.shape[0]
        for Wpq in Wlpq:
            alt, keep, mu = preprocess_lcu_coefficients_for_reversible_sampling(
                lcu_coefficients=Wpq.ravel(), epsilon=probability_epsilon
            )
        altp, altq = np.unravel_index(alt, (num_aux, num_spatial, num_spatial))
        return LthInnerPrepare(
            altp,
            altq,
            keep,
            mu,
        )

    @cached_property
    def selection_registers(self) -> SelectionRegisters:
        num_spatial = self.altUV.shape[-1]
        regs = SelectionRegisters.build(
            l=((num_spatial-1).bit_length(), num_spatial),
            p=((num_spatial-1).bit_length(), num_spatial),
            q=((num_spatial-1).bit_length(), num_spatial),
            succ_pq=(1,),
        )
        return regs

    @cached_property
    def sigma_mu_bitsize(self) -> int:
        return self.mu

    @cached_property
    def alternates_bitsize(self) -> int:
        return sum(reg.bitsize for reg in self.selection_registers) + 1

    @cached_property
    def keep_bitsize(self) -> int:
        return self.mu

    @cached_property
    def junk_registers(self) -> Registers:
        num_spatial = self.altp[0].shape[-1]
        return Registers.build(
            altp=(num_spatial - 1).bit_length(),
            altq=(num_spatial - 1).bit_length(),
            keep=self.keep_bitsize,
            sigma_mu=self.sigma_mu_bitsize,
            less_than_equal=1,
        )

    @cached_property
    def theta_register(self) -> Registers:
        return Registers.build(theta=1)

    @cached_property
    def registers(self) -> Registers:
        return Registers([*self.selection_registers, *self.junk_registers])

    def decompose_from_registers(
        self, context: cirq.DecompositionContext, **quregs: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        p, q = quregs['p'], quregs['q']
        altp, altq = quregs["altp"], quregs["altq"]
        keep = quregs["keep"]
        less_than_equal = quregs["less_than_equal"]
        sigma_mu = quregs["sigma_mu"]
        yield PrepareUniformSuperposition(n=self.num_spatial*self.num_spatial).on_registers(controls=[], target=[p, q])
        yield cirq.H.on_each(*sigma_mu)
        qrom = QROM.build(self.altp, *self.altq, self.keep)
        yield qrom.on_registers(
            selection0=p,
            selection1=q,
            target0=altp,
            target1=altq,
            target2=keep,
        )
        yield LessThanEqualGate(self.mu, self.mu).on(*keep, *sigma_mu, *less_than_equal)
        yield MultiTargetCSwap.make_on(
            control=less_than_equal, target_x=[*altp, *altq], target_y=[*p, *q]
        )
        yield LessThanEqualGate(self.mu, self.mu).on(*keep, *sigma_mu, *less_than_equal)

    def _value_equality_values_(self):
        return self.registers