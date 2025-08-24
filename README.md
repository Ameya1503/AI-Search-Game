# AI-Search-Game
A checkers game AI using MInimax algorithm and alpha-beta pruning with heuristics(Informed search) and BFS (uninformed search) 
## 🎮 Features
- Human vs Computer gameplay
- Two AI modes:
  - **Uninformed**: basic evaluation using piece count
  - **Informed**: advanced heuristic (weighted pieces, king bonus, mobility, center control)
- Minimax with alpha-beta pruning (search depth = 4 by default)
- 8x8 board with checkers-like rules
- Clear text-based board output

---

## ▶️ How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/AI-Search-Game.git
   cd AI-Search-Game
   python game.py


🧠Algorithms used:
---
1)Minimax Algorithm: used for decision-making in a two-player game.
2)Alpha-Beta Pruning: optimization to reduce the number of nodes evaluated.


Evaluation Functions:
---
1)Uninformed: simple difference in number of pieces.
2)Informed: weighted piece count, king bonus, mobility difference, center control.


⚙️Requirements:
---
1)Python 3.x
2)No external libraries required (only built-in modules: math, copy, random, re).


📑Project Presentation
---

For a detailed walkthrough of the project, check out the [Project Presentation (PDF)](docs/AI-Search-Game-Presentation.pdf).



📊Results & Observations
---
- The uninformed AI makes moves based only on piece count, leading to short-term decisions.
- The informed AI evaluates board position more intelligently, often leading to stronger gameplay.
- With depth = 4, the AI provides a reasonable challenge while staying computationally efficient.


🚀Future Improvements
---
- Add a GUI (with Pygame) for visual gameplay.
- Allow adjustable search depth.
- Implement more sophisticated heuristics.
- Add multiplayer mode (human vs human).
