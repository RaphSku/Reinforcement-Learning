import os
import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from typing import Callable, Union

pio.kaleido.scope.default_format = "png"
pio.kaleido.scope.default_width  = 1920
pio.kaleido.scope.default_height = 1080


class ClassicalBandit:
    """A classical Bandit implementation in the context of reinforcement learning

    Parameters
    -----------
        number_of_levers    : int      -> sets the number of the bandit's levers
        reward_size         : int      -> the size with which the reward distribution is sampled
                                          per layer
        reward_distribution : function -> when given an action, it returns the corresponding reward
                                          of the reward distribution
        
        __stationary_distribution : np.ndarray of shape (number_of_levers, reward_size) -> represents the reward distribution per lever
        __optimal_action          : int                                                 -> is the true optimal action based on the reward distribution 

    Methods
    -----------
        get_reward(action : int)       -> returns the appropriate reward for a given action
        get_reward_distribution()      -> returns the stationary reward distribution
        get_optimal_action()           -> returns the optimal action
        
        __set_reward_distribution()    -> creates the stationary reward distribution for the bandit
    """
    __stationary_distribution    = None
    __optimal_action             = None


    def __init__(self, number_of_levers: int, reward_size: int):
        self.number_of_levers = number_of_levers
        self.reward_size      = reward_size
        
        self.reward_distribution = self.__set_reward_distribution()


    def get_reward(self, action: int) -> np.ndarray:
        """Returns the appropriate reward for a given action
        
        Parameters
        ----------
            action : int -> The action value should be in the range of [0, number_of_levers)

        Returns
        ----------
            np.ndarray -> A reward from the reward distribution of shape (number_of_levers, reward_size) is selected
        """

        return self.reward_distribution(action)


    def get_reward_distribution(self) -> np.ndarray:
        """Returns the stationary reward distribution
        
        Returns
        ----------
            np.ndarray -> The reward distribution of shape (number_of_levers, reward_size)
        """

        return self.__stationary_distribution


    def get_optimal_action(self) -> int:
        """Returns the optimal action
        
        Returns
        ----------
            int -> The true optimal action
        """

        return self.__optimal_action


    def __set_reward_distribution(self) -> Callable:
        """Depending on the distribution mode, we get the reward from the stationary or non-stationary distribution

        Returns 
        ----------
            function(action : int) -> A reward is selected based on the created reward distributions
        """
        sigma = np.random.randn(self.number_of_levers)
        mu    = np.random.randn(self.number_of_levers)
        self.__stationary_distribution = np.array([sigma[i] * np.random.rand(self.reward_size) + mu[i] for i in range(self.number_of_levers)]).T


        self.__optimal_action          = np.argmax([(np.quantile(element, 0.75) + np.median(element)) / 2 for element in self.__stationary_distribution.T])

        return lambda action : sigma[action] + np.random.rand(1) + mu[action]           
    

class Agent:
    """The Agent is playing the k-Bandit 

    Parameters
    -----------
        bandit              : ClassicalBandit -> Configured Bandit for the agent to play
        epsilon             : float           -> Determines how often the agent will tend to use the greedy action

        __Q                 : np.ndarray      -> Stores the estimated Q-Values for every Bandit's lever
        __N                 : np.ndarray      -> Stores the count on how often the agent pulled at a certain lever
        __reward_history    : np.ndarray      -> Stores the rewards which the agent encountered throughout playing the Bandit
        __action_history    : np.ndarray      -> Stores the actions which the agent encountered throughout playing the Bandit

    Methods
    -----------
        play(times: int)               -> The agent will play the Bandit (times) times
        get_history()                  -> Returns the reward- and action-history
    """
    __Q              = None
    __N              = None
    __reward_history = None
    __action_history = None


    def __init__(self, bandit: ClassicalBandit, epsilon: float):
        self.bandit  = bandit
        self.epsilon = epsilon


    def play(self, times: int) -> np.ndarray:
        """The Agent will play the k-Bandit 
        
        Parameters
        ----------
            times : int -> The number of times the agent should play the bandit

        Returns
        ----------
            np.ndarray  -> Returns the estimated Q-Values which were estimated by the agent after playing the Bandit
        """
        self.__Q              = np.zeros(self.bandit.number_of_levers)
        self.__N              = np.zeros(self.bandit.number_of_levers)
        self.__reward_history = np.zeros(times)
        self.__action_history = np.zeros(times)

        for t in range(times):
            prob = np.random.uniform(low = 0.0, high = 1.0, size = 1)
            if prob < self.epsilon:
                # exploration
                action = np.random.randint(low = 0, high = self.bandit.number_of_levers, size = 1)
            else:
                # greedy action
                action = np.argmax(self.__Q)
            reward = self.bandit.get_reward(action)
            self.__N[action] += 1
            self.__Q[action] += 1 / self.__N[action] * (reward - self.__Q[action])

            # here we only store information needed for performance measuring
            self.__reward_history[t] = reward
            self.__action_history[t] = action

        return self.__Q
    

    def get_history(self) -> Union[np.ndarray, np.ndarray]:
        """The reward- and action-history is returned which is used in the monitoring
        
        Returns
        ---------
            np.ndarray, np.ndarray -> The first return argument is the reward_history and the second one is the action_history
        """

        return self.__reward_history, self.__action_history


class BanditMonitoring:
    """Monitor which displays the performance of the agent when playing the k-Bandit

    Parameters
    -----------
        number_of_levers : int         -> The number of levers which the Bandit had
        epsilon          : float       -> Determines how often the agent picked the greedy action
        init_storage     : bool        -> If a storage directory was already created, it will be True, otherwise False and 
                                          a directory will be created prior to the plots being saved to that directory

    Methods
    -----------
        show_reward_distribution(reward_df : pd.DataFrame, order : str)   -> The reward distribution will be displayed as a violin plot and stored 
                                                                             in "the current working directory/bandit_metrics" path
        display_performance(history : pd.DataFrame, optimal_action : int) -> Different performance metrics are plotted and stored
                                                                             in the same directory as show_reward_distribution will store the plots
    """


    def __init__(self, number_of_levers : int, epsilon : float):
        self.number_of_levers = number_of_levers
        self.epsilon          = epsilon
        self.init_storage     = False

    
    def show_reward_distribution(self, reward_df: pd.DataFrame, order: str = "column") -> None:
        """The reward distribution which the Bandit used is plotted

        Parameters
        ----------
            reward_df : pd.DataFrame -> This is the Bandit's reward distribution
            order     : str          -> Tells the monitor whether reward_df is stored in column- or row-major, the
                                        default is column-major storage in the dataframe
        """
        fig = go.Figure()
        for index in range(reward_df.shape[1]):
            if order == "column":
                fig.add_trace(
                    go.Violin(y = reward_df.loc[:, index], name = f"Lever {index}")
                )
            elif order == "row":
                fig.add_trace(
                    go.Violin(y = reward_df.loc[index, :], name = f"Lever {index}")
                )
        fig.update_traces(box_visible      = True,
                          meanline_visible = True)
        fig.update_layout(font_family = "Tahoma", 
                          title_text  = "Stationary Reward Distribution",
                          xaxis_title = "",
                          yaxis_title = "Reward",
                          legend_title = "Levers:",
                          font = dict(size = 20))
        fig.show()
        
        if self.init_storage == False:
            if not os.path.isdir("bandit_metrics"):
                os.mkdir("bandit_metrics")
                self.init_storage = True
        
        fig.write_image(f"bandit_metrics/stationary_reward_distribution.png")

    
    def display_performance(self, history : pd.DataFrame, optimal_action : int) -> None:
        """Performance is shown in 4 plots: \n
        1st plot: Cumulated Rewards over Time Steps \n
        2nd plot: Average   Rewards over Time Steps \n
        3rd plot: Optimal Action    over Time Steps \n
        4th plot: Action Histogram \n

        Parameters
        ----------
            history        : pd.DataFrame   -> The history contains the reward- and action-history from the agent
            optimal_action : int            -> The optimal action which was determined by the reward distributions of the Bandit
        """
        timesteps = np.linspace(0, history.iloc[:, 0].to_numpy().shape[0], history.iloc[:, 0].to_numpy().shape[0])
        cumulated_rewards, average_rewards = self.__calculate_reward_metrics(reward_history = history.iloc[:, 0].to_numpy())

        action_history          = history.iloc[:, 1].to_numpy()
        optimal_action_decision = self.__calculate_action_metrics(action_history = action_history, optimal_action = optimal_action)

        fig = make_subplots(rows = 2, cols = 2, 
                            subplot_titles = ("Cumulated Reward Curve", "Average Reward Plot", "Normalised Optimal Action Taken", "Normalised Action Histogram"))
        fig.add_trace(go.Scatter(x = timesteps, y = cumulated_rewards, mode = "lines"), 
                        row = 1, col = 1)
        fig.add_trace(go.Scatter(x = timesteps, y = average_rewards, mode = "markers"),
                        row = 1, col = 2)
        fig.add_trace(go.Scatter(x = timesteps, y = optimal_action_decision, mode = "markers"),
                        row = 2, col = 1)
        fig.add_trace(go.Histogram(x = action_history, histnorm = "probability"),
                        row = 2, col = 2)

        fig.update_xaxes(title_text="Timesteps", row=1, col=1)
        fig.update_xaxes(title_text="Timesteps", row=1, col=2)
        fig.update_xaxes(title_text="Timesteps", row=2, col=1)
        fig.update_xaxes(title_text="Actions",   row=2, col=2)

        fig.update_yaxes(title_text="Cumulated Rewards",    row=1, col=1)
        fig.update_yaxes(title_text="Average Reward",       row=1, col=2)
        fig.update_yaxes(title_text="Optimal Action",       row=2, col=1)
        fig.update_yaxes(title_text="Normalised Frequency", row=2, col=2)

        fig.update_layout(font_family = "Tahoma", 
                          title_text  = f"{self.number_of_levers}-Bandit Performance Measurments (Epsilon = {self.epsilon})",
                          showlegend  = False,
                          font = dict(size = 16))
        fig.update_annotations(font_size = 20)
        fig.show()

        if self.init_storage == False:
            if not os.path.isdir("bandit_metrics"):
                os.mkdir("bandit_metrics")
                self.init_storage = True

        fig.write_image(f"bandit_metrics/bandit_performance_metrics_epsilon_{self.epsilon}_levers_{self.number_of_levers}.png")


    def __calculate_reward_metrics(self, reward_history : np.ndarray) -> Union[np.ndarray, np.ndarray]:
        """Computes the metrics which are associated with the reward history
        
        Parameters
        ----------
            reward_history : np.ndarray     -> The reward history of the agent

        Returns
        ----------
            np.ndarray, np.ndarray          -> The first return argument is the cumulated rewards and the second one is the average rewards per timestep
        """
        timesteps         = reward_history.shape[0]
        cumulated_rewards = np.zeros(timesteps)
        cumulated_reward  = 0
        for index in range(timesteps):
            cumulated_reward        += reward_history[index]
            cumulated_rewards[index] = reward_history[index]
        
        average_rewards = np.zeros(timesteps)
        for index in range(timesteps):
            average_rewards[index] = np.sum(reward_history[:index]) / (index + 1)

        return cumulated_rewards, average_rewards


    def __calculate_action_metrics(self, action_history : np.ndarray, optimal_action : int) -> np.ndarray:
        """Computes the metrics which are associated with the action history

        Parameters
        ----------
            action_history : np.ndarray -> The action history of the agent
            optimal_action : int        -> The true optimal action
        
        Returns
        ----------
            np.ndarray                  -> The normalised best action per timestep
        """
        timesteps            = action_history.shape[0]
        optimal_action_taken = np.zeros(timesteps)
        optimal_action_count = 0
        for index in range(timesteps):
            if action_history[index] == optimal_action:
                optimal_action_count += 1
            optimal_action_taken[index] = optimal_action_count / (index + 1)
        
        return optimal_action_taken


parser = argparse.ArgumentParser(description = "Parameters for the k-Bandit")
parser.add_argument('--k',           dest = "k",           type = int,   default = 10,   help = "Number of Levers the Bandit should have")
parser.add_argument('--reward_size', dest = "reward_size", type = int,   default = 100,  help = "Sample Size for the Reward Distribution")
parser.add_argument('--epsilon',     dest = "epsilon",     type = float, default = 0.1,  help = "Epsilon for the Greedy Action")
parser.add_argument('--timesteps',   dest = "timesteps",   type = int,   default = 1000, help = "Number of Timesteps")
parser.add_argument('--seed',        dest = "seed",        type = int,   default = 100,  help = "Seed for the PRNG")
args   = parser.parse_args()

if __name__ == "__main__":
    NUMBER_OF_LEVERS    = args.k
    REWARD_SIZE         = args.reward_size
    EPSILON             = args.epsilon
    TIMESTEPS           = args.timesteps
    SEED                = np.random.seed(args.seed)

    stationary_bandit   = ClassicalBandit(number_of_levers = NUMBER_OF_LEVERS, reward_size = REWARD_SIZE)
    agent               = Agent(stationary_bandit, epsilon = EPSILON)
    Q_estimated         = agent.play(times = TIMESTEPS)

    reward_distribution = stationary_bandit.get_reward_distribution()
    
    monitor             = BanditMonitoring(number_of_levers = NUMBER_OF_LEVERS, epsilon = EPSILON)
    monitor.show_reward_distribution(reward_df = pd.DataFrame(reward_distribution))

    reward_history, action_history = agent.get_history()
    history                        = pd.DataFrame(data = {'Reward History': reward_history, 'Action History': action_history})
    optimal_action                 = stationary_bandit.get_optimal_action()
    monitor.display_performance(history = history, optimal_action = optimal_action)

    print(f"The estimated Q values are: \n {Q_estimated}")
    print(f"The estimated optimal action is: \n {np.argmax(Q_estimated)}")
    print(f"The optimal action is: \n {optimal_action}")