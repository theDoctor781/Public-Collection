import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class TicTacToe:
    def __init__(self):
        self.board = np.array([[' ' for _ in range(3)] for _ in range(3)])
        self.current_player = 'X'

    def reset(self):
        self.board = np.array([[' ' for _ in range(3)] for _ in range(3)])
        self.current_player = 'X'

    def get_state(self):
        return ''.join(self.board.flatten())

    def is_valid_move(self, row, col):
        return self.board[row][col] == ' '

    def make_move(self, row, col):
        if self.is_valid_move(row, col):
            self.board[row][col] = self.current_player
            self.current_player = 'X' if self.current_player == 'O' else 'O'

    def check_winner(self):
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != ' ':
                return self.board[i][0]
            if self.board[0][i] == self.board[1][i] == self.board[2][i] != ' ':
                return self.board[0][i]

        if self.board[0][0] == self.board[1][1] == self.board[2][2] != ' ':
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != ' ':
            return self.board[0][2]

        if ' ' not in self.board.flatten():
            return 'TIE'

        return None

    def is_game_over(self):
        return self.check_winner() is not None or ' ' not in self.board.flatten()

    def get_empty_positions(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i][j] == ' ']

    def __str__(self):
        return '\n'.join([' | '.join(row) for row in self.board])

class QLearningModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QLearningModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_q_learning_model(episodes, learning_rate, gamma):
    input_size = 9
    hidden_size = 128
    output_size = 9

    model = QLearningModel(input_size, hidden_size, output_size)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for episode in range(episodes):
        game = TicTacToe()
        state = torch.zeros(input_size, dtype=torch.float)
        game_over = False

        while not game_over:
            if np.random.rand() < 0.2:
                # Zufällige Aktion auswählen
                action = np.random.choice(game.get_empty_positions())
            else:
                # Aktion basierend auf dem Q-Wert auswählen
                q_values = model(state)
                valid_moves = [pos[0] * 3 + pos[1] for pos in game.get_empty_positions()]
                q_values_valid_moves = q_values[valid_moves]
                action_idx = valid_moves[torch.argmax(q_values_valid_moves)]
                action = (action_idx // 3, action_idx % 3)

            prev_state = state.clone()
            game.make_move(action[0], action[1])
            state = torch.tensor([1 if cell == 'X' else -1 if cell == 'O' else 0 for row in game.board for cell in row], dtype=torch.float)

            if game.is_game_over():
                game_over = True
                reward = 0
                result = game.check_winner()
                if result == 'X':
                    reward = 1
                elif result == 'O':
                    reward = -1
            else:
                reward = 0

            with torch.no_grad():
                max_next_q = torch.max(model(state))

            target_q = reward + gamma * max_next_q
            predicted_q = model(prev_state)[action[0] * 3 + action[1]]
            loss = loss_function(predicted_q, target_q)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if episode % 100 == 0:
            print(f"Episode {episode} - Loss: {loss.item()}")

    return model

def test_q_learning_model(model):
    game = TicTacToe()
    state = torch.zeros(9, dtype=torch.float)
    game_over = False

    while not game_over:
        print(game)
        valid_moves = [pos[0] * 3 + pos[1] for pos in game.get_empty_positions()]
        q_values = model(state)
        q_values_valid_moves = q_values[valid_moves]
        action_idx = valid_moves[torch.argmax(q_values_valid_moves)]
        action = (action_idx // 3, action_idx % 3)

        game.make_move(action[0], action[1])
        state = torch.tensor([1 if cell == 'X' else -1 if cell == 'O' else 0 for row in game.board for cell in row], dtype=torch.float)

        if game.is_game_over():
            game_over = True
            result = game.check_winner()
            if result == 'X':
                print("Du hast gewonnen!")
            elif result == 'O':
                print("Die KI hat gewonnen!")
            else:
                print("Unentschieden!")

if __name__ == "__main__":
    episodes = 5000
    learning_rate = 0.001
    gamma = 0.9

    model = train_q_learning_model(episodes, learning_rate, gamma)
    test_q_learning_model(model)
