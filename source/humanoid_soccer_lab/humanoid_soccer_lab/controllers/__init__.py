"""Model-based controllers used as baselines for the learned policies.

These are deliberately not learning code. They exist so that any claim about a
policy is measured against a competent controller rather than against nothing.
"""

from .dcm_balance import DcmBalanceCfg, DcmBalanceController

__all__ = ["DcmBalanceCfg", "DcmBalanceController"]
