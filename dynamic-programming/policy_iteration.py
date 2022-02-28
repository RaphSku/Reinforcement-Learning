import numpy as np
from typing import Tuple, Sequence
import policy_evaluation
import policy_improvement


class PolicyIteration:
    """Policy iteration consists of policy evaluation and policy
    improvement in order to converge to a stable policy.

    Parameters
    ----------
        policy_evaluator : policy_evaluation.PolicyEvaluation

        policy_improver  : policy_improvement.PolicyImprovement
    """
    def __init__(self, policy_evaluator: policy_evaluation.PolicyEvaluation, policy_improver: policy_improvement.PolicyImprovement):
        self.policy_evaluator = policy_evaluator
        self.policy_improver  = policy_improver


    def iterate(self, max_iterations: int) -> Tuple[Sequence[Sequence[float]], Sequence[Sequence[float]]]:
        """Iterates over policy evaluation and policy improvement until a stable policy is reached, the policy
        iteration fails after a maximum number of iterations

        Parameters
        ----------
            max_iterations : int
                Number of maximum number of iterations

        Returns
        -------
            2d array, 2d array
                The value matrix and the policy is returned after policy iteration
        """
        iteration = 0
        while iteration < max_iterations:
            self.policy_evaluator.evaluate_policy(max_iterations = 1000)
            is_policy_stable = self.policy_improver.improve_policy()
            if is_policy_stable:
                break

        return self.policy_evaluator.grid_world.state_values, self.policy_evaluator.policy


def main():
    grid_world       = policy_evaluation.GridWorld2D(rows = 4, columns = 4, initialiser = np.random.rand)
    policy_fetcher   = policy_evaluation.Policy(json_path_to_file = "./custom_policy.json")
    policy_evaluator = policy_evaluation.PolicyEvaluation(grid_world = grid_world, policy = policy_fetcher, discount_rate = 0.9)
    policy_improver  = policy_improvement.PolicyImprovement(grid_world = grid_world, policy_evaluator = policy_evaluator)

    policy_iterator  = PolicyIteration(policy_evaluator = policy_evaluator, policy_improver = policy_improver)
    states, policy   = policy_iterator.iterate(max_iterations = 1000)
    print(f"The state matrix is the following:\n{states}")
    print(f"Policy:\n{policy}")


if __name__ == "__main__":
    main()