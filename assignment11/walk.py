from dataclasses import dataclass
from typing import Mapping, Dict, Optional, Tuple, Iterator
from rl.distribution import Categorical, FiniteDistribution
from rl.markov_process import FiniteMarkovRewardProcess


S = Tuple[int, int]
StateReward = FiniteDistribution[Tuple[S, float]]
RewardTransition = Mapping[S, Optional[StateReward]]


@dataclass
class RandomWalk2D(FiniteMarkovRewardProcess[S]):
    """Toy two-dimensional random walk MRP.
    
    The MRP's states are locations within a two-dimensional
    "grid world", with the edges being terminal states. At 
    each time step, we movce either UP, DOWN, LEFT, or, RIGHT
    with defined probability. The top and right edges give a
    reward of 1, while every other state has a reward of 0.
    """

    B1: int
    B2: int
    lprob: float
    rprob: float
    dprob: float
    uprob: float

    def __post_init__(self) -> None:
        """Pass the transition map to the parent class."""
        super().__init__(self.get_transition_map())
        
    @property
    def states(self) -> Iterator[S]:
        """Generate all possible states."""
        for i in range(self.B1 + 1):
            for j in range(self.B2 + 1):
                yield i, j
                
    def reward(self, state: S) -> float:
        """Get the reward for a given state."""
        i, j = state
        if i == self.B1 or j == self.B2:
            return 1.0
        return 0.0

    def get_transition_map(self) -> RewardTransition:
        """Map states to distributions of next_state, reward pairs."""
        tmap: Dict[S, Optional[Categoricalp[Tuple[S, float]]]] = {}
        
        terminal = lambda i, j: i == self.B1 or j == self.B2 or i == 0 or j == 0
        
        for s in self.states:
            i, j = s
            if terminal(i, j):
                tmap[s] = None
                continue
              
            # these are in i, j format NOT row, col
            steps = [(1,0), (-1,0), (0,1), (0, -1)]
            probs = [self.uprob, self.dprob, self.rprob, self.lprob]
            dist = {}
            for (a, b), prob in zip(steps, probs):
                next_state = (i + a, j + b)
                dist[(next_state, self.reward(next_state))] = prob
            tmap[s] = Categorical(dist)
                
        return tmap