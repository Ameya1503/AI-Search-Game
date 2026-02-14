import csv
import math
import os
# Importing your game logic from engine.py
from engine import initial_board, get_all_moves, apply_move, minimax, flatten_board, heuristic_informed

def collect_data(games=100):
    # This path saves the file exactly where your script is located
    # No subfolders = no "FileNotFound" errors
    file_path = 'training_data.csv'

    print(f"Starting data collection. Target file: {file_path}")

    try:
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            for g in range(games):
                board = initial_board()
                turn = 'w'
                print(f"Simulating game {g+1} of {games}...")
                
                for move_count in range(100): # Max 100 moves to prevent infinite loops
                    moves = get_all_moves(board, turn)
                    if not moves: 
                        break
                    
                    # We use Depth 2 for data collection speed
                    # This uses your existing 'informed' heuristic to teach the MLP
                    val, mv = minimax(board, 2, -math.inf, math.inf, True, turn, 'informed')
                    
                    if mv is None: 
                        break

                    # Convert 8x8 board to 64 numerical inputs for the Neural Network
                    flat_state = flatten_board(board)
                    
                    # Write the 64 board squares + the heuristic score (Target)
                    writer.writerow(flat_state + [val])
                    
                    board = apply_move(board, mv)
                    # Switch turns: w -> b, b -> w
                    turn = 'b' if turn == 'w' else 'w'
        
        print(f"Success! Data collection complete. File saved as: {os.path.abspath(file_path)}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    collect_data(100)