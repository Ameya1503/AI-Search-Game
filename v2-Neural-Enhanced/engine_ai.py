import tensorflow as tf
import numpy as np
import math
from engine import initial_board, get_all_moves, apply_move, game_over, print_board, flatten_board, parse_move_strict

# 1. LOAD THE NEURAL BRAIN
print("Loading Neural Network...")
model = tf.keras.models.load_model('checkers_model.h5', compile=False)

def neural_heuristic(board):
    """Uses the MLP to predict the win-probability/score of a board state."""
    flat = np.array([flatten_board(board)])
    # The model predicts the score based on the 64 squares
    prediction = model.predict(flat, verbose=0)
    return float(prediction[0][0])

def minimax_neural(board, depth, alpha, beta, maximizing):
    over, winner = game_over(board)
    if over:
        if winner == 'b': return math.inf, None
        else: return -math.inf, None
    if depth == 0:
        return neural_heuristic(board), None

    cur_side = 'b' if maximizing else 'w'
    moves = get_all_moves(board, cur_side)
    if not moves: return (-math.inf if maximizing else math.inf), None

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

def play_neural():
    board = initial_board()
    print('\n--- Neural Network Optimized Checkers ---')
    print('The AI is now using the Multi-Layer Perceptron to think!')
    
    turn = 'w' # You are white
    while True:
        print_board(board)
        over, winner = game_over(board)
        if over:
            print("You Win!" if winner == 'w' else "Neural AI Wins!")
            break
            
        if turn == 'w':
            moves = get_all_moves(board, 'w')
            for i, m in enumerate(moves):
                print(f"{i}: {'->'.join(str(c) for c in m)}")
            idx = int(input("Choose move index: "))
            board = apply_move(board, moves[idx])
            turn = 'b'
        else:
            print("Neural AI is thinking...")
            score, mv = minimax_neural(board, 3, -math.inf, math.inf, True)
            print(f"AI Confidence Score: {score:.4f}")
            board = apply_move(board, mv)
            turn = 'w'

if __name__ == "__main__":
    play_neural()