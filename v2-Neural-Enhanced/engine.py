#cd "C:\Users\Ameya\OneDrive\Desktop\AIML GAME"
#python game.py



import copy
import math
import random

BOARD_SIZE = 8
MAX_DEPTH = 4

def flatten_board(board):
    """Converts 8x8 board to a list of 64 numbers for the MLP."""
    flat = []
    mapping = {None: 0, 'w': 1, 'W': 2, 'b': -1, 'B': -2}
    for row in board:
        for cell in row:
            flat.append(mapping[cell])
    return flat

def initial_board():
    board = [[None]*BOARD_SIZE for _ in range(BOARD_SIZE)]
    for r in range(3):
        for c in range(BOARD_SIZE):
            if (r + c) % 2 == 1:
                board[r][c] = 'w'  # white at top (player)
    for r in range(5,8):
        for c in range(BOARD_SIZE):
            if (r + c) % 2 == 1:
                board[r][c] = 'b'  # black at bottom (computer)
    return board



def in_bounds(r,c):
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE

def print_board(board):
    print('   ' + ' '.join(str(c+1) for c in range(BOARD_SIZE)))
    for r in range(BOARD_SIZE):
        row = []
        for c in range(BOARD_SIZE):
            cell = board[r][c]
            row.append(cell if cell else '.')
        print(f'{r+1:2} ' + ' '.join(row))
    print()


def get_all_moves(board, side):
    moves = []
    capture_moves = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            p = board[r][c]
            if not p: continue
            if side == 'b' and p.lower() == 'b':
                cm = piece_captures(board, r, c)
                if cm:
                    capture_moves.extend(cm)
                else:
                    moves.extend(piece_simple_moves(board, r, c))
            if side == 'w' and p.lower() == 'w':
                cm = piece_captures(board, r, c)
                if cm:
                    capture_moves.extend(cm)
                else:
                    moves.extend(piece_simple_moves(board, r, c))
    return capture_moves if capture_moves else moves


def piece_simple_moves(board, r, c):
    p = board[r][c]
    res = []
    dirs = []
    if p == 'b': dirs = [(-1,-1),(-1,1)]
    elif p == 'w': dirs = [(1,-1),(1,1)]
    else: dirs = [(-1,-1),(-1,1),(1,-1),(1,1)]  # kings
    for dr,dc in dirs:
        nr, nc = r+dr, c+dc
        if in_bounds(nr,nc) and board[nr][nc] is None:
            res.append([(r,c),(nr,nc)])
    return res


def piece_captures(board, r, c):
    p = board[r][c]
    if not p: return []
    color = 'b' if p.lower()=='b' else 'w'
    dirs = [(-1,-1),(-1,1),(1,-1),(1,1)]
    results = []

    def recurse(bd, r0, c0, path, visited):
        found = False
        for dr,dc in dirs:
            midr, midc = r0+dr, c0+dc
            endr, endc = r0+2*dr, c0+2*dc
            if not (in_bounds(midr,midc) and in_bounds(endr,endc)): continue
            mid = bd[midr][midc]
            end = bd[endr][endc]
            if mid and mid.lower() != p.lower() and end is None:
                
                bd2 = copy.deepcopy(bd)
                bd2[endr][endc] = bd2[r0][c0]
                bd2[r0][c0] = None
                bd2[midr][midc] = None
                
                if bd2[endr][endc] == 'b' and endr == 0: bd2[endr][endc] = 'B'
                if bd2[endr][endc] == 'w' and endr == BOARD_SIZE-1: bd2[endr][endc] = 'W'
                recurse(bd2, endr, endc, path+[(endr,endc)], visited|{(midr,midc)})
                found = True
        if not found and len(path) > 1:
            results.append(path)

    recurse(board, r, c, [(r,c)], set())
    return results


def apply_move(board, move):
    bd = copy.deepcopy(board)
    r0,c0 = move[0]
    piece = bd[r0][c0]
    bd[r0][c0] = None
    for (r,c) in move[1:]:
        
        pr = (r0 + r)//2
        pc = (c0 + c)//2
        if abs(r-r0) == 2 and bd[pr][pc] is not None:
            bd[pr][pc] = None
        r0,c0 = r,c
    bd[r0][c0] = piece
    
    if piece == 'b' and r0 == 0: bd[r0][c0] = 'B'
    if piece == 'w' and r0 == BOARD_SIZE-1: bd[r0][c0] = 'W'
    return bd


def game_over(board):
    b_moves = get_all_moves(board, 'b')
    w_moves = get_all_moves(board, 'w')
    b_pieces = sum(1 for r in board for c in r if c and c.lower()=='b')
    w_pieces = sum(1 for r in board for c in r if c and c.lower()=='w')
    if b_pieces==0 or not b_moves:
        return True, 'w' 
    if w_pieces==0 or not w_moves:
        return True, 'b'
    return False, None

# Heuristics

def heuristic_uninformed(board, side):
    
    score = 0
    for r in board:
        for cell in r:
            if not cell: continue
            v = 1.0
            if cell.isupper(): v = 1.5
            if cell.lower() == 'b': score += v
            else: score -= v
    return score if side=='b' else -score

def heuristic_informed(board, side):
    # weighted pieces + king bonus + mobility + center control
    piece_score = 0
    center_bonus = 0
    for i,r in enumerate(board):
        for j,cell in enumerate(r):
            if not cell: continue
            v = 1.0
            if cell.isupper(): v = 1.75
            sign = 1 if cell.lower()=='b' else -1
            piece_score += sign * v
            # center control
            if 2 <= i <= 5 and 2 <= j <=5:
                center_bonus += sign * 0.2
    mobility = (len(get_all_moves(board,'b')) - len(get_all_moves(board,'w'))) * 0.1
    score = piece_score + center_bonus + mobility
    return score if side=='b' else -score

# Minimax with alpha-beta, side is 'b' (max) or 'w' (min)

def minimax(board, depth, alpha, beta, maximizing, side, mode):
    over, winner = game_over(board)
    if over:
        if winner == 'b': return (math.inf if maximizing else -math.inf), None
        else: return (-math.inf if maximizing else math.inf), None
    if depth == 0:
        h = heuristic_informed(board, 'b') if mode=='informed' else heuristic_uninformed(board, 'b')
        return h, None

    cur_side = 'b' if maximizing else 'w'
    moves = get_all_moves(board, cur_side)
    if not moves:
        # current player has no moves => loses
        return (-math.inf if maximizing else math.inf), None

    best_move = None
    if maximizing:
        value = -math.inf
        for m in moves:
            nb = apply_move(board, m)
            v, _ = minimax(nb, depth-1, alpha, beta, False, side, mode)
            if v > value:
                value = v; best_move = m
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value, best_move
    else:
        value = math.inf
        for m in moves:
            nb = apply_move(board, m)
            v, _ = minimax(nb, depth-1, alpha, beta, True, side, mode)
            if v < value:
                value = v; best_move = m
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value, best_move


def parse_move(s):
    try:
        parts = s.strip().split('-')
        coords = []
        for p in parts:
            r = int(p[0]) - 1 if p[0].isdigit() else int(p[:-1])

    except Exception:
        return None

    return parse_move_strict(s)

import re

def parse_move_strict(s):
    
    s = s.strip().lower()
    tokens = s.split('-')
    coords = []
    for t in tokens:
        m = re.match(r"^(\d)\s*[,c]?\s*(\d)$", t)
        if not m:
            return None
        r = int(m.group(1))-1
        c = int(m.group(2))-1
        if not in_bounds(r,c): return None
        coords.append((r,c))
    return coords

# Game loop

def play():
    board = initial_board()
    print('Simplified Checkers: You are White (w). Computer is Black (b).')
    mode = ''
    while mode not in ('informed','uninformed'):
        mode = input('Choose AI mode (informed/uninformed): ').strip().lower()
    human_side = 'w'
    computer_side = 'b'
    turn = 'w'  # white starts
    while True:
        print_board(board)
        over, winner = game_over(board)
        if over:
            if winner == human_side:
                print('You win!')
            else:
                print('Computer wins!')
            break
        if turn == human_side:
            moves = get_all_moves(board, human_side)
            if not moves:
                print('No legal moves. You lose.')
                break
            print('Your moves: (format r c - r c; example: 6c3-5b4 not supported here; use "r c - r c" as "6 3-5 4")')

            for i,m in enumerate(moves):
                s = '->'.join(f'({r+1},{c+1})' for r,c in m)
                print(f'{i}: {s}')
            user = input('Enter move index or move (e.g., 6 3-5 4): ').strip()
            if user.isdigit():
                idx = int(user)
                if 0 <= idx < len(moves):
                    board = apply_move(board, moves[idx])
                    turn = computer_side
                    continue
                else:
                    print('Invalid index')
                    continue
            parsed = parse_move_strict(user)
            if not parsed:
                print('Could not parse move. Use index or format like: 6 3-5 4')
                continue
            # verify move is legal
            if parsed in moves:
                board = apply_move(board, parsed)
                turn = computer_side
            else:
                print('Illegal move.')
                continue
        else:
            print('Computer thinking...')
            val, mv = minimax(board, MAX_DEPTH, -math.inf, math.inf, True, computer_side, mode)
            if mv is None:
                print('Computer has no moves. You win!')
                break
            print('Computer plays:', '->'.join(f'({r+1},{c+1})' for r,c in mv))
            board = apply_move(board, mv)
            turn = human_side

if __name__ == '__main__':
    play()

