import random

def roll_dice():
    return random.randint(1, 6), random.randint(1, 6)

def display_remaining_numbers(numbers):
    print(f"Remaining numbers: {', '.join(map(str, numbers))}")

import itertools

def get_valid_moves(numbers, dice_sum):
    valid_moves = []
    for r in range(1, len(numbers) + 1):
        for combination in itertools.combinations(numbers, r):
            if sum(combination) == dice_sum:
                valid_moves.append(combination)
    return valid_moves

def main(strategy=None):
    numbers = list(range(1, 10))
    game_over = False
    turn_count = 0

    if strategy is None:
        print("Welcome to Shut the Box!")
    else:
        print("Welcome to Shut the Box! The bot will now play the game using the provided strategy.\n")

    while not game_over:
        turn_count += 1
        print(f"Turn {turn_count}:")
        
        dice = roll_dice()
        dice_sum = sum(dice)
        print(f"You rolled {dice[0]} and {dice[1]}. Total: {dice_sum}")
        
        valid_moves = get_valid_moves(numbers, dice_sum)
        
        if valid_moves:
            display_remaining_numbers(numbers)

            if strategy is None:
                print("Valid moves:")
                for i, move in enumerate(valid_moves, start=1):
                    print(f"{i}: {', '.join(map(str, move))}")

                while True:
                    choice = input("Enter the index of the chosen move: ")

                    if choice.isdigit() and 1 <= int(choice) <= len(valid_moves):
                        chosen_move = valid_moves[int(choice) - 1]
                        break
                    print("Invalid move. Please try again.")
            else:
                chosen_move = strategy(valid_moves, numbers)
                print(f"Bot chose to shut: {', '.join(map(str, chosen_move))}")

            for num in chosen_move:
                numbers.remove(num)
        else:
            print("No valid moves. Game over.")
            print(f"Final score: {turn_count - 1} turns")
            return(False)
        
        if not numbers:
            print("Congratulations! You shut all the numbers!")
            game_over = True
            return(True)

def simple_strategy(valid_moves, numbers):
    if not valid_moves:
        return None
    max_move = max(valid_moves, key=lambda move: sum(move))
    return max_move


import itertools

def probability_of_rolling_remaining_numbers(remaining_numbers):
    total_ways = 36  # There are 6 * 6 ways to roll two dice
    successful_ways = 0
    
    # Get all subsets of remaining_numbers (excluding the empty set)
    all_subsets = [subset for r in range(1, len(remaining_numbers) + 1)
                   for subset in itertools.combinations(remaining_numbers, r)]

    # Calculate the number of ways to roll each sum
    for subset in all_subsets:
        subset_sum = sum(subset)
        if 2 <= subset_sum <= 12:  # The sum must be between 2 and 12 (inclusive) for two dice
            successful_ways += ways_to_roll_sum(subset_sum)

    return successful_ways / total_ways

def ways_to_roll_sum(target_sum):
    ways = 0
    for die1 in range(1, 7):
        for die2 in range(1, 7):
            if die1 + die2 == target_sum:
                ways += 1
    return ways

def max_probability_strategy(valid_moves, remaining_numbers):
    if not valid_moves:
        return None

    def remaining_numbers_after_move(move, remaining_numbers):
        return [number for number in remaining_numbers if number not in move]

    best_move = None
    best_probability = -1

    for move in valid_moves:
        remaining_numbers_after_current_move = remaining_numbers_after_move(move, remaining_numbers)
        probability = probability_of_rolling_remaining_numbers(remaining_numbers_after_current_move)

        if probability > best_probability:
            best_probability = probability
            best_move = move

    return best_move

import random
import itertools

def simulate_game(remaining_numbers, num_simulations=1000):
    scores = []

    for _ in range(num_simulations):
        numbers_copy = remaining_numbers.copy()
        while numbers_copy:
            dice_sum = sum(roll_dice())
            valid_moves = get_valid_moves(numbers_copy, dice_sum)

            if not valid_moves:
                break

            # Choose a random move
            chosen_move = random.choice(valid_moves)
            for num in chosen_move:
                numbers_copy.remove(num)

        scores.append(sum(numbers_copy))

    return sum(scores) / num_simulations

def expected_value_strategy(valid_moves, remaining_numbers, num_simulations=1000):
    if not valid_moves:
        return None

    best_move = None
    best_expected_value = float("inf")

    for move in valid_moves:
        remaining_numbers_after_current_move = [number for number in remaining_numbers if number not in move]
        expected_value = simulate_game(remaining_numbers_after_current_move, num_simulations)

        if expected_value < best_expected_value:
            best_expected_value = expected_value
            best_move = move

    return best_move


class ShutTheBoxEnvironment:
    def __init__(self):
        self.reset()

    def reset(self):
        self.numbers = list(range(1, 10))
        return self.state()

    def state(self):
        return tuple(sorted(self.numbers))

    def roll_dice(self):
        return (random.randint(1, 6), random.randint(1, 6))

    def step(self, action):
        dice_sum = sum(self.roll_dice())
        valid_moves = get_valid_moves(self.numbers, dice_sum)

        if not valid_moves:
            reward = -sum(self.numbers)  # negative reward, based on the sum of the remaining numbers
            done = True
        else:
            if action not in valid_moves:
                return self.state(), -100, True  # heavy penalty for invalid moves

            for num in action:
                self.numbers.remove(num)

            if not self.numbers:
                reward = 100  # large positive reward for shutting all numbers
                done = True
            else:
                reward = 0
                done = False

        return self.state(), reward, done

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def get_q_value(self, state, action):
        state_action = (state, action)
        if state_action not in self.q_table:
            self.q_table[state_action] = 0
        return self.q_table[state_action]

    def choose_action(self, state, valid_moves):
        if random.random() < self.epsilon:
            return random.choice(valid_moves)  # exploration
        return max(valid_moves, key=lambda move: self.get_q_value(state, move))  # exploitation

    def update(self, state, action, reward, next_state, next_valid_moves):
        current_q = self.get_q_value(state, action)
        next_q = max([self.get_q_value(next_state, next_move) for next_move in next_valid_moves]) if next_valid_moves else 0
        self.q_table[(state, action)] = current_q + self.alpha * (reward + self.gamma * next_q - current_q)

def train_agent(agent, env, episodes=10000):
    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            valid_moves = [move for move in itertools.chain(*[itertools.combinations(env.numbers, r) for r in range(1, len(env.numbers) + 1)]) if sum(move) in range(2, 13)]
            action = agent.choose_action(state, valid_moves)
            next_state, reward, done = env.step(action)
            next_valid_moves = [move for move in itertools.chain(*[itertools.combinations(env.numbers, r) for r in range(1, len(env.numbers) + 1)]) if sum(move) in range(2, 13)]
            agent.update(state, action, reward, next_state, next_valid_moves)
            state = next_state

def get_all_possible_sums():
    all_sums = set()
    for die1_roll1 in range(1, 7):
        for die1_roll2 in range(1, 7):
            for die2_roll1 in range(1, 7):
                for die2_roll2 in range(1, 7):
                    all_sums.add((die1_roll1 + die1_roll2, die2_roll1 + die2_roll2))
    return all_sums

def probability_of_covering_next_two_moves(move, remaining_numbers):
    all_possible_sums = get_all_possible_sums()
    
    #print(f'All possible sums: {all_possible_sums}')
    remaining_numbers_after_current_move = [number for number in remaining_numbers if number not in move]
    count = 0
    total = 0

    for sum1, sum2 in all_possible_sums:
        if sum1 in remaining_numbers_after_current_move and sum2 in remaining_numbers_after_current_move:
            count += ways_to_roll_sum(sum1) * ways_to_roll_sum(sum2)
        total += ways_to_roll_sum(sum1) * ways_to_roll_sum(sum2)

    return count / total

def max_probability_next_two_moves_strategy(valid_moves, remaining_numbers):
    if not valid_moves:
        return None

    best_move = None
    best_probability = -1

    for move in valid_moves:
        probability = probability_of_covering_next_two_moves(move, remaining_numbers)
        print(f"Move: {move}, Probability: {probability:.4f}")
        if probability > best_probability:
            best_probability = probability
            best_move = move

    return best_move

if __name__ == "__main__":

    win = 0
    num = 10000
    for i in range(num):
        if main(strategy=max_probability_next_two_moves_strategy):  # Pass the strategy function or set to None for human player
            win+=1
    print(f'{win/num*100} % win rate')

import itertools 
def get_valid_moves(numbers, dice_sum):
    valid_moves = []
    for r in range(1, len(numbers) + 1):
        for combination in itertools.combinations(numbers, r):
            if sum(combination) == dice_sum:
                valid_moves.append(combination)
    return valid_moves
