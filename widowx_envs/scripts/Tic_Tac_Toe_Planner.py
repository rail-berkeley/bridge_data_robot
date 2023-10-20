import random

class Tic_Tac_Toe: 
    def __init__(self) -> None:
        self.BOARD = [[" " for _ in range(3)] for _ in range(3)]
        self.CURRENT_PLAYER = "X"
        self.GAME_OVER = False

    def print_board(self):
        for row in self.BOARD:
            print(" | ".join(row))
            print("-" * 9)

    def check_winner(self, board, player):
        # Check rows, columns, and diagonals
        for i in range(3):
            if all(board[i][j] == player for j in range(3)) or all(board[j][i] == player for j in range(3)):
                return True
        if all(board[i][i] == player for i in range(3)) or all(board[i][2 - i] == player for i in range(3)):
            return True
        return False

    def is_board_full(self, board):
        return all(cell != " " for row in board for cell in row)

    def get_empty_cells(self, board):
        empty_cells = []
        for i in range(3):
            for j in range(3):
                if board[i][j] == " ":
                    empty_cells.append((i, j))
        return empty_cells

    def minimax(self, board, depth, is_maximizing):
        if self.check_winner(board, "O"):
            return 1
        if self.check_winner(board, "X"):
            return -1
        if self.is_board_full(board):
            return 0

        if is_maximizing:
            best_score = -float("inf")
            for row, col in self.get_empty_cells(board):
                board[row][col] = "O"
                score = self.minimax(board, depth + 1, False)
                board[row][col] = " "
                best_score = max(score, best_score)
            return best_score
        else:
            best_score = float("inf")
            for row, col in self.get_empty_cells(board):
                board[row][col] = "X"
                score = self.minimax(board, depth + 1, True)
                board[row][col] = " "
                best_score = min(score, best_score)
            return best_score

    def find_best_move(self, board):
        best_move = None
        best_score = -float("inf")
        for row, col in self.get_empty_cells(board):
            board[row][col] = "O"
            score = self.minimax(board, 0, False)
            board[row][col] = " "
            if score > best_score:
                best_score = score
                best_move = (row, col)
        return best_move

    def one_turn(self, move=""):
        self.print_board()

        if self.CURRENT_PLAYER == "X":
            #move = input(f"Player {self.CURRENT_PLAYER}, enter your move (e.g., A2): ").strip().upper()
            #while len(move) != 2 or not move[0] in "ABC" or not move[1] in "123" or self.BOARD[int(move[1]) - 1][ord(move[0]) - ord("A")] != " ":
            #    print("Invalid move. Try again.")
            #    move = input(f"Player {self.CURRENT_PLAYER}, enter your move (e.g., A2): ").strip().upper()
            row, col = int(move[1]) - 1, ord(move[0]) - ord("A")
        else:
            print(f"Player {self.CURRENT_PLAYER} is thinking...")
            row, col = self.find_best_move(self.BOARD)

        self.BOARD[row][col] = self.CURRENT_PLAYER

        self.print_board()

        if self.check_winner(self.BOARD, self.CURRENT_PLAYER):
            self.print_board()
            print(f"Player {self.CURRENT_PLAYER} wins!")
            self.GAME_OVER = True
            #return
        elif self.is_board_full(self.BOARD):
            self.print_board()
            print("It's a draw!")
            self.GAME_OVER = True 
            #return 

        self.CURRENT_PLAYER = "X" if self.CURRENT_PLAYER == "O" else "O"


    #board = [[" " for _ in range(3)] for _ in range(3)]
    #current_player = "X"
