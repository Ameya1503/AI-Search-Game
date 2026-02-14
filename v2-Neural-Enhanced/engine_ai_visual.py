import tensorflow as tf
import numpy as np
import math
import os

from engine import initial_board, get_all_moves, apply_move, game_over, flatten_board

try:
    model = tf.keras.models.load_model('checkers_model.h5', compile=False)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

def get_meter(score):
    """Creates a visual bar based on the AI's confidence."""
    # Scale score from -1 to 1 into a 20-character bar
    percent = (score + 1) / 2
    percent = max(0, min(1, percent)) # Keep between 0 and 1
    bar_len = int(percent * 20)
    bar = "█" * bar_len + "-" * (20 - bar_len)
    return f"[{bar}] {score:+.4f}"

def print_visual_board(board, score):
    """Clears the terminal and draws a clean dashboard."""
    os.system('cls' if os.name == 'nt' else 'clear') 
    print("\n  =========================================")
    print("     NEURAL OPTIMIZED CHECKERS ENGINE      ")
    print("  =========================================")
    print(f"   AI CONFIDENCE: {get_meter(score)}")
    print("  -----------------------------------------")
    print("      1  2  3  4  5  6  7  8")
    for i, row in enumerate(board):
        row_str = f"   {i+1} "
        for cell in row:
            if cell == 'w': row_str += " ○ " 
            elif cell == 'b': row_str += " ● " 
            elif cell == 'W': row_str += " ♔ " 
            elif cell == 'B': row_str += " ♚ " 
            else: row_str += " . "
        print(row_str)
    print("  =========================================\n")

def neural_heuristic(board):
    """The MLP replaces your manual math here."""
    flat = np.array([flatten_board(board)])
    prediction = model.predict(flat, verbose=0)
    return float(prediction[0][0])

def minimax_neural(board, depth, alpha, beta, maximizing):
    """Recursive search using the Neural Network for evaluation."""
    over, winner = game_over(board)
    if over:
        if winner == 'b': return 1.0, None # Max win for AI
        if winner == 'w': return -1.0, None # Max loss for AI
        return 0, None

    if depth == 0:
        return neural_heuristic(board), None

    cur_side = 'b' if maximizing else 'w'
    moves = get_all_moves(board, cur_side)
    if not moves:
        return ((-1.0 if maximizing else 1.0), None)

    best_move = None
    if maximizing:
        value = -math.inf
        for m in moves:
            nb = apply_move(board, m)
            v, _ = minimax_neural(nb, depth-1, alpha, beta, False)
            if v > value:
                value = v; best_move = m
            alpha = max(alpha, value)
            if alpha >= beta: break
        return value, best_move
    else:
        value = math.inf
        for m in moves:
            nb = apply_move(board, m)
            v, _ = minimax_neural(nb, depth-1, alpha, beta, True)
            if v < value:
                value = v; best_move = m
            beta = min(beta, value)
            if alpha >= beta: break
        return value, best_move

def play_visual():
    board = initial_board()
    turn = 'w' # You are white
    current_score = 0.0
    
    while True:
        print_visual_board(board, current_score)
        over, winner = game_over(board)
        if over:
            print(f"!!! GAME OVER !!! Winner: {'YOU' if winner=='w' else 'NEURAL AI'}")
            break
            
        if turn == 'w':
            moves = get_all_moves(board, 'w')
            for i, m in enumerate(moves):
                print(f" {i}: {'->'.join(str(c) for c in m)}")
            
            try:
                prompt = input("\n Choose move index: ")
                idx = int(prompt)
                board = apply_move(board, moves[idx])
                # Update score after your move
                current_score = neural_heuristic(board)
                turn = 'b'
            except:
                print("Invalid input, try again.")
        else:
            print(" AI is analyzing patterns with MLP...")
            current_score, mv = minimax_neural(board, 3, -math.inf, math.inf, True)
            if mv:
                board = apply_move(board, mv)
            turn = 'w'

if __name__ == "__main__":
    play_visual()