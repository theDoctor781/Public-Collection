def print_board(board):
    for row in board:
        print(" ".join(row))
    print()

def is_winner(board, player):
    for i in range(3):
        if all(cell == player for cell in board[i]) or all(board[j][i] == player for j in range(3)):
            return True
    # Überprüfe Diagonalen
    if all(board[i][i] == player for i in range(3)) or all(board[i][2 - i] == player for i in range(3)):
        return True
    return False

def is_board_full(board):
    return all(cell != ' ' for row in board for cell in row)

def minimax(board, depth, maximizing_player):
    if is_winner(board, 'X'):
        return -1
    elif is_winner(board, 'O'):
        return 1
    elif is_board_full(board):
        return 0

    if maximizing_player:
        max_eval = float('-inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'O'
                    eval = minimax(board, depth + 1, False)
                    board[i][j] = ' '
                    max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'X'
                    eval = minimax(board, depth + 1, True)
                    board[i][j] = ' '
                    min_eval = min(min_eval, eval)
        return min_eval

def find_best_move(board):
    best_val = float('-inf')
    best_move = (-1, -1)

    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                board[i][j] = 'O'
                move_val = minimax(board, 0, False)
                board[i][j] = ' '

                if move_val > best_val:
                    best_move = (i, j)
                    best_val = move_val

    return best_move

def play_game():
    board = [[' ' for _ in range(3)] for _ in range(3)]

    while True:
        print_board(board)

        player_move = tuple(map(int, input("Dein Zug (Zeile Spalte): ").split()))
        if board[player_move[0]][player_move[1]] != ' ':
            print("Ungültiger Zug, Feld bereits belegt. Versuche es erneut.")
            continue

        board[player_move[0]][player_move[1]] = 'X'

        if is_winner(board, 'X'):
            print_board(board)
            print("Du hast gewonnen!")
            break

        if is_board_full(board):
            print_board(board)
            print("Unentschieden!")
            break

        print("Computer zieht...")
        computer_move = find_best_move(board)
        board[computer_move[0]][computer_move[1]] = 'O'

        if is_winner(board, 'O'):
            print_board(board)
            print("Computer gewinnt!")
            break

        if is_board_full(board):
            print_board(board)
            print("Unentschieden!")
            break

if __name__ == "__main__":
    play_game()
