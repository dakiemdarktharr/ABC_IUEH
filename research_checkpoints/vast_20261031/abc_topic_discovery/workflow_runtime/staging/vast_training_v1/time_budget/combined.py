"""Compose the separately funded time and USD gates, without granting a run."""
from .guard import TimeBudgetGate, TimeBudgetStop
from .ledger import ReservationAuthority
from ..ops_v2.budget_gate import BUDGET_POLICY
from ..runner.fit_api import BudgetGate


class CombinedBudgetGate:
    """Root constructs this before CUDA startup and injects it into the trainer.

    The watchdog callback must signal an independent supervisor. Creating this
    object does not authenticate that supervisor or independently anchor history.
    """
    def __init__(self, time_gate, usd_gate, *, stop_requested=None):
        if type(time_gate) is not TimeBudgetGate or type(time_gate._authority) is not ReservationAuthority:
            raise PermissionError('a reservation-backed time gate is required')
        if not isinstance(usd_gate, BudgetGate):
            raise PermissionError('the live USD gate is required')
        self._time = time_gate
        self._usd = usd_gate
        self._stop_requested = stop_requested or (lambda: False)
        self._check_policy()

    def _check_policy(self):
        policy = self._usd.budget_policy
        if (type(policy) is not dict or set(policy) != set(BUDGET_POLICY)
                or any(type(policy[k]) not in (int, float) or policy[k] != v
                       for k, v in BUDGET_POLICY.items())):
            raise PermissionError('USD gate policy differs from the $4 soft / $16 hard limits')

    @property
    def budget_policy(self):
        return self._usd.budget_policy

    def before_cuda(self):
        if self._stop_requested():
            raise TimeBudgetStop('supervisor requested checkpoint and stop')
        if self._time._authority._lock is None:
            raise TimeBudgetStop('reservation lock is no longer held')
        # Same margin as the update gate: startup consumes the reservation too.
        self._time.before_update(remaining_updates=1)
        return True

    def before_update(self, *, remaining_updates):
        self.before_cuda()
        self._check_policy()
        if self._usd.before_update(remaining_updates=remaining_updates) is not True:
            raise TimeBudgetStop('USD gate denied update')
        # The USD lookup itself may consume the remaining wall budget.
        return self._time.before_update(remaining_updates=remaining_updates)
