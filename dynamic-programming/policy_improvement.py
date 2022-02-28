import numpy as np
import policy_evaluation
from typing import Tuple


class PolicyImprovement:
    """Improves the policy based on the greedy action

    Parameters
    ----------
        grid_world       : policy_evaluation.GridWorld2D
            A matrix with two terminal states and usually randomly initialised values
        policy_evaluator : policy_evaluation.PolicyEvaluation
            Policy improvement works in conjunction with policy evaluation and we need
            the discount-rate as also the policy which was used for policy evaluation

    Methods
    -------
        improve_policy() bool
            Algorithm which improves the policy and returns if the policy is stable or not
    """


    def __init__(self, grid_world: policy_evaluation.GridWorld2D, policy_evaluator: policy_evaluation.PolicyEvaluation):
        self.grid_world    = grid_world
        self.policy        = policy_evaluator.policy
        self.discount_rate = policy_evaluator.discount_rate


    def improve_policy(self) -> bool:
        """Algorithm for improving the policy
        
        Returns
        -------
            bool
                Whether the policy is stable or not
        """
        is_policy_stable = True
        for row in range(self.grid_world.rows):
            for column in range(self.grid_world.columns):
                if self.policy[row][column]["state"] == "T":
                        continue
                old_greedy_action, new_greedy_action = self.__improve(row = row, column = column)
                if old_greedy_action != new_greedy_action:
                    is_policy_stable = False
        return is_policy_stable


    def __improve(self, row: int, column: int) -> Tuple[int, int]:
        """Performs an improvement for one state

        Parameters
        ----------
            row    : int
                Identifies the row of the cell which determines the value
            column : int
                Identifies the column of the cell which determines the value

        Returns
        -------
            int, int
                The old greedy action and the new greedy action are returned
        """
        old_greedy_action = np.argmax(self.policy[row][column]["probs"])
        new_actions       = []
        for index in range(len(self.policy[row][column]["probs"])):
            # up action
            if index == 0:
                new_actions.append(-1 + self.discount_rate * self.grid_world[int(row - 1), column])
            # right action
            if index == 1:
                if int(column + 1) >= self.grid_world.columns:
                    new_actions.append(-1 + self.discount_rate * self.grid_world[row, 0])
                    continue
                new_actions.append(-1 + self.discount_rate * self.grid_world[row, int(column + 1)])
            # down action
            if index == 2:
                if int(row + 1) >= self.grid_world.rows:
                    new_actions.append(-1 + self.discount_rate * self.grid_world[0, column])
                    continue
                new_actions.append(-1 + self.discount_rate * self.grid_world[int(row + 1), column])
            # left action
            if index == 3:
                new_actions.append(-1 + self.discount_rate * self.grid_world[row, int(column - 1)])
        new_greedy_action = np.argmax(new_actions)
        self.policy[row][column]["probs"] = np.zeros(len(self.policy[row][column]["probs"]))
        self.policy[row][column]["probs"][new_greedy_action] = 1.0

        return old_greedy_action, new_greedy_action
                

def main():
    grid_world       = policy_evaluation.GridWorld2D(rows = 4, columns = 4, initialiser = np.random.rand)
    policy_fetcher   = policy_evaluation.Policy(json_path_to_file = "./custom_policy.json")
    policy_evaluator = policy_evaluation.PolicyEvaluation(grid_world = grid_world, policy = policy_fetcher, discount_rate = 0.9)
    policy_evaluator.evaluate_policy(max_iterations = 1000)

    policy_improvement = PolicyImprovement(grid_world = grid_world, policy_evaluator = policy_evaluator)
    is_policy_stable   = policy_improvement.improve_policy()
    print(f"The policy is stable: {is_policy_stable}!")


if __name__ == "__main__":
    main()