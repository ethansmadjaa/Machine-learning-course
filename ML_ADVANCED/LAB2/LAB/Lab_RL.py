import random

import matplotlib.pyplot as plt
import pandas as pd

alpha = 0.1  # learning rate
gamma = 0.95  # Discount factor
epsilon = 0.1  # exploration rate
episodes = 1000  # number of training episodes

# action definitions
# 0: Hold, 1: Buy 10 shares, 2: Sell 10 shares
actions = [0, 1, 2]

# trading parameters
initial_cash = 5000
share_amount = 10


def portfolio_value(cash, shares, price):
    return cash + shares * price


def valid_actions(cash, shares, price):
    valid = [0]
    if cash >= share_amount * price:
        valid.append(1)  # buy is valid if enough cash
    if shares >= share_amount:
        valid.append(2)  # sell is valid if enough shares
    return valid


train_data = pd.read_csv('./data/bank_of_america.csv', parse_dates=['Date'])
train_data.sort_values('Date', inplace=True)
train_prices = train_data['Close'].values
n_train = len(train_prices)

# Initialize Q-table as a dictionary
# State defined as (day, shares)
Q = {}


def get_Q(state, action):
    return Q.get((state, action), 0.0)


def update_Q(state, action, value):
    Q[(state, action)] = value


for ep in range(episodes):
    # reset initial conditions at the start of each episode
    cash = initial_cash
    shares = 0
    total_portfolio = portfolio_value(cash, shares, train_prices[0])

    for day in range(n_train - 1):
        state = (day, shares)

        # choose valid actions using epsilon-greedy policy
        possible_actions = valid_actions(cash, shares, train_prices[day])
        if random.random() < epsilon:
            action = random.choice(possible_actions)
        else:
            # Choose action with highest Q value among valid ones
            q_values = {a: get_Q(state, a) for a in possible_actions}
            action = max(q_values, key=q_values.get)

        # action
        price_today = train_prices[day]
        price_next = train_prices[day + 1]
        # simulate action
        if action == 1:  # Buy
            cash -= share_amount * price_today
            shares += share_amount
        elif action == 2:  # Sell
            cash += share_amount * price_today
            shares -= share_amount
        # if hold, no change

        # Reward: change in portfolio value from today to next day
        portfolio_today = portfolio_value(cash, shares, price_today)
        portfolio_next = portfolio_value(cash, shares, price_next)
        reward = portfolio_next - portfolio_today

        # Next state: move to next day (shares already updated)
        next_state = (day + 1, shares)

        # Q-learning update
        # Compute max Q for next state over all possible actions (using next day's price)
        valid_next_actions = valid_actions(cash, shares, price_next)
        max_Q_next = max([get_Q(next_state, a) for a in valid_next_actions], default=0.0)

        current_Q = get_Q(state, action)
        new_Q = (1 - alpha) * current_Q + alpha * (reward + gamma * max_Q_next)
        update_Q(state, action, new_Q)

    # Print episode summary for monitoring
    if (ep + 1) % 10 == 0:
        print(f"Episode {ep + 1}/{episodes} completed.")
        print(f"Total Portfolio: {total_portfolio}")

# testing on GE data
test_data = pd.read_csv('./data/ge.csv', parse_dates=['Date'])
test_data.sort_values('Date', inplace=True)
test_prices = test_data['Close'].values
n_test = len(test_prices)

# initialize testing portfolio
cash = initial_cash
shares = 0
portfolio_values = []
actions_taken = []
dates = test_data['Date'].tolist()

for day in range(n_test - 1):
    state = (day, shares)
    price_today = test_prices[day]
    price_next = test_prices[day + 1]

    # determine valid actions in test environment
    possible_actions = valid_actions(cash, shares, price_today)
    # Choose action using learned Q-table (greedy policy)
    q_values = {a: get_Q(state, a) for a in possible_actions}
    action = max(q_values, key=q_values.get)
    actions_taken.append(action)

    # execute action
    if action == 1:  # Buy
        cash -= share_amount * price_today
        shares += share_amount
    elif action == 2:  # Sell
        cash += share_amount * price_today
        shares -= share_amount
    # else hold

    # record portfolio value after the action (using today's price)
    portfolio_values.append(portfolio_value(cash, shares, price_today))

# Append final portfolio value for the last day
portfolio_values.append(portfolio_value(cash, shares, test_prices[-1]))
actions_taken.append(0)  # No action on final day


plt.figure(figsize=(14, 6))
plt.subplot(2, 1, 1)
plt.plot(dates, portfolio_values, label='Portfolio Value')
plt.xlabel('Date')
plt.ylabel('Portfolio Value (USD)')
plt.title('Portfolio Value Over Time (Test Data)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
# Convert actions to labels for plotting
action_labels = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
action_text = [action_labels[a] for a in actions_taken]
plt.plot(dates, actions_taken, 'o-', label='Actions (0=Hold, 1=Buy, 2=Sell)')
plt.xlabel('Date')
plt.ylabel('Action')
plt.title('Actions Taken Over Time GE (Test Data)')
plt.yticks([0, 1, 2], ['Hold', 'Buy', 'Sell'])
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./data/portfolio_value.png")


