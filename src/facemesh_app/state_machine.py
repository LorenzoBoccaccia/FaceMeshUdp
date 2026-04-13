# state_machine.py
"""
StateMachine module for managing dispatcher states.
Provides DispatcherState enum and StateMachine class for state management.
"""

import logging
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class DispatcherState(Enum):
    """
    Dispatcher states for the frame processing pipeline.
    """
    IDLE = "IDLE"
    CALIBRATION = "CALIBRATION"
    OPERATIONAL = "OPERATIONAL"


StateTransitionCallback = Callable[[DispatcherState, DispatcherState], None]


class StateMachine:
    """
    Manages state transitions for the dispatcher with callback support.
    """
    
    def __init__(self, initial_state: DispatcherState = DispatcherState.IDLE):
        """
        Initialize StateMachine with an initial state.
        
        Args:
            initial_state: The starting state (default: IDLE)
        """
        self._state: DispatcherState = initial_state
        self._callback: Optional[StateTransitionCallback] = None
        logger.info(f"StateMachine initialized with state: {initial_state.value}")
    
    def get_state(self) -> DispatcherState:
        """
        Get the current state.
        
        Returns:
            The current DispatcherState
        """
        return self._state
    
    def transition_to(self, new_state: DispatcherState) -> None:
        """
        Transition to a new state.
        
        Args:
            new_state: The target state to transition to
            
        Raises:
            ValueError: If attempting to transition to the same state
        """
        if self._state == new_state:
            error_msg = f"Cannot transition to same state: {new_state.value}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        old_state = self._state
        self._state = new_state
        logger.info(f"State transition: {old_state.value} -> {new_state.value}")
        
        # Call callback if registered
        if self._callback is not None:
            self._callback(old_state, new_state)
    
    def set_transition_callback(self, callback: StateTransitionCallback) -> None:
        """
        Register a callback to be invoked on state transitions.
        
        Args:
            callback: The callback function to register
        """
        self._callback = callback
        logger.debug("Transition callback registered")
    
    def clear_transition_callback(self) -> None:
        """
        Remove the registered callback.
        """
        self._callback = None
        logger.debug("Transition callback cleared")
