# 🧠 Knapsack Problem Solver using Genetic Algorithm

This project provides an interactive and visual solution to the **Knapsack Problem** using a **Genetic Algorithm (GA)**. It supports both variants of the problem:

- **0-1 Knapsack**: Each item can be selected at most once.
- **Unbounded Knapsack**: Each item can be selected multiple times.

The application includes a full-featured **Graphical User Interface (GUI)** built with **Tkinter**, and a **Command-Line Interface (CLI)** for flexible usage.

---

## 🚀 Features

- 🔁 Solves both **0-1** and **Unbounded** knapsack problems using Genetic Algorithms.
- 🖥️ GUI built with **Tkinter**, featuring:
  - Tabbed interface: Input, Visualization, and Results
  - Dynamic plotting of fitness evolution using `matplotlib`
  - Real-time logs, generation tracking, and progress bar
  - Visual representation of knapsack contents and fill level
- 💻 CLI mode for terminal-based execution
- 🔍 Supports randomly generating realistic example item sets
- 🎯 Uses crossover, mutation, tournament selection, and elitism
- 🧪 Designed for experimentation and educational demonstration
- ⚙️ Stops early if no improvement after a threshold

---

## 📁 File Structure

- knapsack.py # Main Python script with GUI, CLI, and algorithm implementation
- README.md # This documentation file

---
---

## 🛠️ Requirements

- Python 3.8 or higher

### 📦 Required Packages

Install dependencies using pip:

```bash
pip install matplotlib numpy
```
---
## ▶️ How to Run
**🖥️ Run the GUI (Default):**

python knapsack.py
This will launch the graphical interface where you can:
Enter or randomly generate items
Choose knapsack type (0-1 or unbounded)
Run and visualize the algorithm in real-time
View results and fitness graph
**💻 Run from Command Line:**

python knapsack.py --c
You’ll be prompted to input:
Knapsack capacity
Number of items
Item weights and values
Problem type (0-1 or unbounded)
It will then output the best solution found by the genetic algorithm.

---
**📊 Example Workflow (GUI)**
Enter knapsack capacity (e.g., 100)

Add items manually or click “Fill with Example Data”

Choose between:

0-1 Knapsack – items used at most once

Unbounded Knapsack – items can be reused

Click “Run Algorithm”

Observe the live progress on the fitness graph

View the optimal result in the Results tab

---
## 🙋‍♂️ Author

**Seif Ahmed**  
[LinkedIn Profile](https://www.linkedin.com/in/seif-almaz/)
Computer Science Student – Helwan National University

---
