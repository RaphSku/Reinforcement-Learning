import numpy as np
import json
from typing import Callable, Sequence


class GridWorld2D:
    """Defines the values of a 2D grid world

    Parameters
    ----------
        rows        : int
            The number of `rows` the grid should have
        columns     : int
            The number of `columns` the grid should have
        initialiser : Callable
            A function which is used to initialise the values of
            the 2D grid world
    """


    def __init__(self, rows: int, columns: int, initialiser: Callable):
        self.rows         = rows
        self.columns      = columns
        self.state_values = self.__set_value(initialiser)    


    def __set_value(self, initialiser: Callable) -> Sequence[float]:
        """The values will be initialised on a 2D grid,
        the terminal states which are always at uper left corner
        and lower right corner are initialised with the value 0

        Parameters
        ----------
            initialiser : Callable
                A typical initialiser is np.random.rand which samples
                from an uniform distribution in the range [0, 1)
        
        Returns
        -------
            array-like[float]
                Basically a matrix is returned with the initialised values
        """
        value         = initialiser(self.rows, self.columns)
        value[0][0]   = 0
        value[-1][-1] = 0

        return value


    def __getitem__(self, indices):
        row, column = indices

        return self.state_values[row][column]

    
    def __setitem__(self, indices, value):
        row, column = indices
        self.state_values[row][column] = value


class Policy:
    """A JSON document is used in order
    to define a policy and this class extracts
    the relevant properties
    
    Parameters
    ----------
        json_path_to_file : str
            The path to the JSON file which defines the policy. 
            The JSON document contains a 2D array with two properties,
            the name of the state and the probabilities for each state
    
    Methods
    -------
        get_policy()
            Opens the Policy JSON file and returns the policy which
            should be evaluated
    """


    def __init__(self, json_path_to_file: str):
        self.json_path_to_file = json_path_to_file


    def get_policy(self):
        """The JSON file has to be a 2D matrix, filled
        with dictionaries, the dictionary contains as a key
        the name of the state and as a value the array of probabilities
        for the actions = {up, right, down, left}

        Returns
        -------
            list
                A 2D matrix which contains the dictionary which describes
                the policy
        """
        with open(self.json_path_to_file) as json_file:
            content = self.__check_policy(json.load(json_file)["policy"])

            return content


    def __check_policy(self, policy: Sequence[dict[str, float]]) -> Sequence[dict[str, float]]:
        """The policy has to be checked if its valid,
        that means for every state the sum of probabilities
        to select a certain action has to be 1.

        Parameters
        ----------
            policy : array-like[dict[str, float]]
                For each state a dictionary contains the state name and the action
                probabilities

        Raises
        ------
            ValueError
                If some set of actions for a state does not sum up 
                to 1, then this error is triggered
        """
        for row in range(len(policy)):
            for column in range(len(policy[0])):
                if np.sum(policy[row][column]["probs"]) != 1.0 and policy[row][column]["state"] != "T":
                    raise ValueError(f"Policy needs to be normalised for every state, the \
                        probabilities have to sum up to 1! The error occured in row: {row} and column: {column}!")

        return policy

        
class PolicyEvaluation:
    """Evaluates the given policy for the given 2D grid world
    
    Parameters
    ----------
        grid_world    : GridWorld2D
            Value field on which the policy should be evaluated
        policy        : Policy
            The given policy which should be evaluated
        discount_rate : float
            Determines how future rewards influence the agent

    Methods
    -------
        evaluate_policy(max_iterations : int) array-like
            The policy is evaluated on the 2D grid world and the result is the
            updated state matrix
    """
    def __init__(self, grid_world: GridWorld2D, policy: Policy, discount_rate: float):
        self.grid_world    = grid_world
        self.policy        = policy.get_policy()
        self.discount_rate = discount_rate


    def evaluate_policy(self, max_iterations: int) -> Sequence[Sequence[float]]:
        """The policy is updated iteratively, if the policy
        evaluation is successful, then the value grid world
        is returned, otherwise it fails after the maximum
        number of iterations. Also, the state matrix is printed
        
        Parameters
        ----------
            max_iterations : int
                When reaching the maximum number of iterations,
                the evaluation stops

        Returns
        -------
            array-like  
                The state matrix is returned
        """
        iteration = 0
        while True:
            delta = 0
            for row in range(self.grid_world.rows):
                for column in range(self.grid_world.columns):
                    current_value  = self.grid_world[row, column]
                    next_value     = self.__evaluate_subpolicy(row = row, column = column)
                    self.grid_world[row, column] = next_value
                    delta          = max(delta, np.abs(current_value - self.grid_world[row, column]))
                    iteration += 1
            if iteration == max_iterations:
                if delta >= 0.01:
                    print(f"Evaluation failed because delta which is {delta} is greater than 0.01!")
                break
            if delta < 0.01:
                print(f"Evaluation was successful!")
                break
        self.__print_result()
        
        return self.grid_world.state_values

    def __evaluate_subpolicy(self, row: int, column: int) -> float:
        """For every state, one has to iterate over all actions
        and sum up all contributions from the next state which result
        from taking the selected action

        Parameters
        ----------
            row    : int
                The current row of the 2D grid world
            column : int
                The current column of the 2D grid world

        Returns
        -------
            float
                The value update
        """
        actions       = self.policy[row][column]
        current_state = self.policy[row][column]["state"]
        next_value    = 0
        for index, action_probability in enumerate(actions["probs"]):
            reward = -1
            if current_state == "T":
                reward = 0
            # up action
            if index == 0:
                next_value += action_probability * (reward + self.discount_rate * self.grid_world[int(row - 1), column])
                continue
            # right action
            if index == 1:
                if int(column + 1) >= self.grid_world.columns:
                    next_value += action_probability * (reward + self.discount_rate * self.grid_world[row, 0])
                    continue
                next_value += action_probability * (reward + self.discount_rate * self.grid_world[row, int(column + 1)])
                continue
            # down action
            if index == 2:
                if int(row + 1) >= self.grid_world.rows:
                    next_value += action_probability * (reward + self.discount_rate * self.grid_world[0, column])
                    continue
                next_value += action_probability * (reward + self.discount_rate * self.grid_world[int(row + 1), column])
                continue
            # left action
            if index == 3:
                next_value += action_probability * (reward + self.discount_rate * self.grid_world[row, int(column - 1)])

        return next_value


    def __print_result(self) -> None:
        print(f"Policy Evaluation result:")
        for row in range(self.grid_world.rows):
            for column in range(self.grid_world.columns):
                if (row == 0 and column == 0) or (row == self.grid_world.rows - 1 and column == self.grid_world.columns - 1):
                    print(f"+{self.grid_world[row, column]:.2f}", end = " | ")
                    continue
                print(f"{self.grid_world[row, column]:.2f}", end = " | ")
            print("\n")


def main():
    grid_world       = GridWorld2D(rows = 4, columns = 4, initialiser = np.random.rand)
    policy_fetcher   = Policy(json_path_to_file = "./custom_policy.json")
    policy_evaluater = PolicyEvaluation(grid_world = grid_world, policy = policy_fetcher, discount_rate = 0.9)
    policy_evaluater.evaluate_policy(max_iterations = 1000)


if __name__ == "__main__":
    main()