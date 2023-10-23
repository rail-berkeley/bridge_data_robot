class Tic_Tac_Toe:
    def print_board(self, board):
      for row in board:
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

    def get_LLM_move(self, board):
      row, col = self.find_best_move(board)
      board[row][col] = "O"
      return (row, col)

    def game_over(self, board):
      if self.check_winner(board, 'X'):
        return "player X wins!"
      elif self.check_winner(board, 'O'):
        return "player O wins!"
      elif self.is_board_full(board):
        return "It's a draw."
      else:
        return ""