# Neural-Optimized Checkers Engine (v2.0)

This project documents the evolution of a Checkers AI from a classical game theory approach to a modern Deep Learning-based "Neural Engine."

## 📂 Project Structure
* **`/v1-Classical-Minimax`**: The foundational engine using human-defined logic.
* **`/v2-Neural-Enhanced`**: The optimized engine using an MLP to evaluate board states.
* **`/visualizations`**: Contains performance plots (Correlation and Heatmaps).

---

## 🕹️ Phase 1: The Classical Minimax Engine (v1)
The original version was built using the **Minimax Algorithm with Alpha-Beta Pruning**.

### Key Features:
* **State Space Search**: Explores moves recursively to a fixed depth.
* **Heuristic Evaluation**: Used a manual formula involving piece counts, center square control, and mobility.
* **Results**: While effective, the engine was limited by "human-bias"—it only understood the strategic patterns I manually programmed into it.

---

## 🧠 Phase 2: Neural Optimization (v2)
In the second phase, I replaced the manual heuristic formula with a **Deep Multi-Layer Perceptron (MLP)** built using TensorFlow.

### Model Architecture:
I chose a tapered "Bottleneck" architecture to ensure high-dimensional feature abstraction:
* **Input Layer**: 64 neurons (representing each square of the 8x8 board).
* **Hidden Layer 1 (128 neurons)**: Captures broad spatial relationships.
* **Hidden Layer 2 (64 neurons)**: Filters for more specific strategic motifs.
* **Hidden Layer 3 (32 neurons)**: Compresses patterns into a final strategic score.
* **Activation (ReLU)**: Used in all hidden layers to solve the **Vanishing Gradient problem** and ensure efficient training.



---

## 📊 Performance & Accuracy
The model was trained for **30 Epochs** using the **Adam Optimizer** and **Mean Squared Error (MSE)** loss.

* **Training Results**: Achieved a validation loss of **1.5e-15**.
* **Correlation**: A near-perfect diagonal match between the classical heuristic and neural predictions, proving successful **Function Approximation**.
* **Inference**: The engine now makes decisions based on "learned intuition" rather than static formulas.



---

## ⚔️ Comparison Analysis
| Feature | v1 (Classical) | v2 (Neural Optimized) |
| :--- | :--- | :--- |
| **Logic Source** | Human-coded Heuristics | Learned Patterns (MLP) |
| **Decision Speed** | Slower (Mathematical complexity) | Faster (Matrix multiplication) |
| **Adaptability** | Rigid/Rule-based | Flexible/Data-driven |
| **Accuracy (MSE)** | N/A | 1.5e-15 (High Fidelity) |

---

## 🚀 How to Run
1. Navigate to `/v2-Neural-Enhanced`.
2. Ensure `checkers_model.h5` and `engine.py` are in the same folder.
3. Run the dashboard: `python engine_ai_visual.py`.

---
**Ameya Nangle**
*B.Tech ECE (Minor in AI/ML) | MIT World Peace University*
