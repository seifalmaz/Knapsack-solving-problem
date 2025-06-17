import random
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Define class for items
class Item:
    def __init__(self, weight, value):
        self.weight = weight
        self.value = value

# Global parameters
POPULATION_SIZE = 100
MUTATION_RATE = 0.05
GENERATIONS = 100
MAX_GENERATIONS_WITHOUT_IMPROVEMENT = 20
ELITISM_COUNT = 2

# Global variables for GUI updating
gui_best_fitness = 0
gui_current_generation = 0
gui_solution = None
gui_items = []
gui_capacity = 0
gui_mode = ""
gui_running = False
gui_generation_data = []

def initialize_population_01(num_items):
    population = []
    for _ in range(POPULATION_SIZE):
        chromosome = [random.randint(0, 1) for _ in range(num_items)]
        population.append(chromosome)
    return population

def initialize_population_unbounded(num_items):
    population = []
    for _ in range(POPULATION_SIZE):
        chromosome = [0] * num_items
        remaining_capacity = knapsack_capacity
        
        # 50% chance of using a greedy approach (highest value/weight ratio first)
        if random.random() < 0.5:
            value_weight_ratios = [(i, items_available[i].value / items_available[i].weight 
                                  if items_available[i].weight > 0 else 0) 
                                 for i in range(num_items)]
            sorted_indices = [idx for idx, _ in sorted(value_weight_ratios, key=lambda x: x[1], reverse=True)]
            
            for idx in sorted_indices:
                if items_available[idx].weight <= 0:
                    continue
                max_count = remaining_capacity // items_available[idx].weight
                if max_count > 0:
                    count = random.randint(0, max_count)
                    chromosome[idx] = count
                    remaining_capacity -= count * items_available[idx].weight
        else:
            # Random approach
            items_order = list(range(num_items))
            random.shuffle(items_order)
            
            for idx in items_order:
                if items_available[idx].weight <= 0:
                    continue
                max_count = remaining_capacity // items_available[idx].weight
                if max_count > 0:
                    count = random.randint(0, max_count)
                    chromosome[idx] = count
                    remaining_capacity -= count * items_available[idx].weight
        population.append(chromosome)
    return population

def calculate_fitness_01(chromosome, items, capacity):
    total_weight = sum(chromosome[i] * items[i].weight for i in range(len(chromosome)))
    total_value = sum(chromosome[i] * items[i].value for i in range(len(chromosome)))

    if total_weight > capacity:
        return 0  # Invalid solution
    return total_value

def calculate_fitness_unbounded(chromosome, items, capacity):
    total_weight = sum(chromosome[i] * items[i].weight for i in range(len(chromosome)))
    total_value = sum(chromosome[i] * items[i].value for i in range(len(chromosome)))

    if total_weight <= capacity:
        return total_value  # Valid solution
    else:
        # Apply a penalty for exceeding capacity
        penalty_factor = 0.5
        penalty = penalty_factor * (total_weight - capacity)
        return max(0, total_value - penalty)

def selection(population, fitness_scores):
    selected_parents = []
    tournament_size = 5

    for _ in range(len(population) // 2):
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_participants = [(fitness_scores[i], population[i]) for i in tournament_indices]
        winner_fitness, winner_chromosome = max(tournament_participants)
        selected_parents.append(winner_chromosome)

    return selected_parents

def crossover(parent1, parent2):
    if len(parent1) <= 1:
        child1 = parent1[:]
        child2 = parent2[:]
        return child1, child2
    
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate_01(chromosome):
    for i in range(len(chromosome)):
        if random.random() < MUTATION_RATE:
            chromosome[i] = 1 - chromosome[i]  # Flip bit
    return chromosome

def mutate_unbounded(chromosome, items):
    # Random mutation
    for i in range(len(chromosome)):
        if random.random() < MUTATION_RATE:
            current_count = chromosome[i]
            max_possible = knapsack_capacity // (items[i].weight if items[i].weight > 0 else 1)
            
            # Adjust count up or down
            if current_count == 0:
                new_count = random.randint(0, min(3, max_possible))
            elif current_count == max_possible:
                new_count = max(0, current_count - random.randint(0, 3))
            else:
                adjustment = random.choice([-2, -1, 1, 2])
                new_count = max(0, min(max_possible, current_count + adjustment))
                
            chromosome[i] = new_count
    
    # Repair if invalid
    total_weight = sum(chromosome[i] * items[i].weight for i in range(len(chromosome)))
    
    if total_weight > knapsack_capacity:
        # Remove items with lowest value/weight ratio until valid
        value_weight_ratios = [(i, items[i].value / items[i].weight 
                              if items[i].weight > 0 else 0) 
                             for i in range(len(chromosome))]
        sorted_indices = [idx for idx, _ in sorted(value_weight_ratios, key=lambda x: x[1])]
        
        for idx in sorted_indices:
            while chromosome[idx] > 0 and total_weight > knapsack_capacity:
                chromosome[idx] -= 1
                total_weight -= items[idx].weight
                
    return chromosome

def genetic_algorithm(items, capacity, mode):
    global knapsack_capacity, items_available
    knapsack_capacity = capacity
    items_available = items
    
    num_items = len(items)

    # Initialize based on mode
    if mode == '0-1':
        population = initialize_population_01(num_items)
        calculate_fitness = calculate_fitness_01
        mutate = mutate_01
    elif mode == 'unbounded':
        population = initialize_population_unbounded(num_items)
        calculate_fitness = calculate_fitness_unbounded
        mutate = lambda chrom: mutate_unbounded(chrom, items)

    best_solution = None
    best_fitness = 0
    generations_without_improvement = 0
    
    for generation in range(GENERATIONS):
        # Calculate fitness for all chromosomes
        fitness_scores = [calculate_fitness(chromosome, items, capacity) for chromosome in population]

        # Find best solution in current generation
        current_best_fitness = max(fitness_scores)
        current_best_index = fitness_scores.index(current_best_fitness)
        current_best_solution = population[current_best_index]

        # Update best solution if improved
        if current_best_fitness > best_fitness:
            # Verify solution is valid
            total_weight = sum(current_best_solution[i] * items[i].weight for i in range(len(current_best_solution)))
            if total_weight <= capacity:
                best_fitness = current_best_fitness
                best_solution = current_best_solution.copy()
                generations_without_improvement = 0
        else:
            generations_without_improvement += 1
            
        # Early stopping
        if generations_without_improvement >= MAX_GENERATIONS_WITHOUT_IMPROVEMENT:
            break

        # Elitism - preserve best solutions
        elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:ELITISM_COUNT]
        elites = [population[i].copy() for i in elite_indices]

        # Selection
        parents = selection(population, fitness_scores)

        # Create next generation through crossover and mutation
        next_population = []
        for i in range(0, len(parents) - (len(parents) % 2), 2):
            child1, child2 = crossover(parents[i], parents[i+1])
            next_population.append(mutate(child1))
            next_population.append(mutate(child2))

        # Add any leftover parent
        if len(parents) % 2 != 0:
            next_population.append(mutate(parents[-1]))

        # Add elites to next generation
        next_population.extend(elites)

        # Maintain population size
        while len(next_population) < POPULATION_SIZE:
            if mode == '0-1':
                next_population.append(initialize_population_01(num_items)[0])
            elif mode == 'unbounded':
                next_population.append(initialize_population_unbounded(num_items)[0])

        population = next_population[:POPULATION_SIZE]

    return best_solution, best_fitness

def genetic_algorithm_with_gui(items, capacity, mode, update_interval=1):
    global knapsack_capacity, items_available, gui_best_fitness, gui_current_generation, gui_solution, gui_running, gui_generation_data
    
    knapsack_capacity = capacity
    items_available = items
    gui_best_fitness = 0
    gui_current_generation = 0
    gui_solution = None
    gui_generation_data = []
    gui_running = True
    
    num_items = len(items)

    # Initialize based on mode
    if mode == '0-1':
        population = initialize_population_01(num_items)
        calculate_fitness = calculate_fitness_01
        mutate = mutate_01
    elif mode == 'unbounded':
        population = initialize_population_unbounded(num_items)
        calculate_fitness = calculate_fitness_unbounded
        mutate = lambda chrom: mutate_unbounded(chrom, items)

    best_solution = None
    best_fitness = 0
    generations_without_improvement = 0
    
    for generation in range(GENERATIONS):
        if not gui_running:
            break
            
        gui_current_generation = generation
        
        # Calculate fitness for all chromosomes
        fitness_scores = [calculate_fitness(chromosome, items, capacity) for chromosome in population]

        # Find best solution in current generation
        current_best_fitness = max(fitness_scores)
        current_best_index = fitness_scores.index(current_best_fitness)
        current_best_solution = population[current_best_index]

        # Update generation data for plotting
        avg_fitness = sum(fitness_scores) / len(fitness_scores)
        gui_generation_data.append((generation, current_best_fitness, avg_fitness))
        
        # Update best solution if improved
        if current_best_fitness > best_fitness:
            # Verify solution is valid
            total_weight = sum(current_best_solution[i] * items[i].weight for i in range(len(current_best_solution)))
            if total_weight <= capacity:
                best_fitness = current_best_fitness
                best_solution = current_best_solution.copy()
                gui_best_fitness = best_fitness
                gui_solution = best_solution
                generations_without_improvement = 0
        else:
            generations_without_improvement += 1
            
        # Early stopping
        if generations_without_improvement >= MAX_GENERATIONS_WITHOUT_IMPROVEMENT:
            break

        # Elitism - preserve best solutions
        elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:ELITISM_COUNT]
        elites = [population[i].copy() for i in elite_indices]

        # Selection
        parents = selection(population, fitness_scores)

        # Create next generation through crossover and mutation
        next_population = []
        for i in range(0, len(parents) - (len(parents) % 2), 2):
            child1, child2 = crossover(parents[i], parents[i+1])
            next_population.append(mutate(child1))
            next_population.append(mutate(child2))

        # Add any leftover parent
        if len(parents) % 2 != 0:
            next_population.append(mutate(parents[-1]))

        # Add elites to next generation
        next_population.extend(elites)

        # Maintain population size
        while len(next_population) < POPULATION_SIZE:
            if mode == '0-1':
                next_population.append(initialize_population_01(num_items)[0])
            elif mode == 'unbounded':
                next_population.append(initialize_population_unbounded(num_items)[0])

        population = next_population[:POPULATION_SIZE]
        
        # Sleep to slow down visualization
        time.sleep(update_interval)

    gui_running = False
    return best_solution, best_fitness

def print_solution(items, solution, capacity, fitness):
    print(f"Maximum value: {fitness}")
    print(f"Knapsack capacity: {capacity}")
    
    total_weight = sum(solution[i] * items[i].weight for i in range(len(solution)))
    print(f"Total weight: {total_weight}/{capacity}")
    
    print("\nItems in knapsack:")
    for i in range(len(solution)):
        if solution[i] > 0:
            print(f"Item {i+1}: {solution[i]} × (Weight: {items[i].weight}, Value: {items[i].value})")

class KnapsackGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Knapsack Problem Solver")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        
        self.items = []
        self.algorithm_thread = None
        self.update_timer = None
        
        self.create_widgets()
        self.setup_layout()
        
    def create_widgets(self):
        # Create tabs
        self.tab_control = ttk.Notebook(self.root)
        self.input_tab = ttk.Frame(self.tab_control)
        self.visualization_tab = ttk.Frame(self.tab_control)
        self.results_tab = ttk.Frame(self.tab_control)
        
        self.tab_control.add(self.input_tab, text="Input Data")
        self.tab_control.add(self.visualization_tab, text="Visualization")
        self.tab_control.add(self.results_tab, text="Results")
        
        # Input Tab Widgets
        self.create_input_widgets()
        
        # Visualization Tab Widgets
        self.create_visualization_widgets()
        
        # Results Tab Widgets
        self.create_results_widgets()
    
    def create_input_widgets(self):
        # Capacity Frame
        capacity_frame = ttk.LabelFrame(self.input_tab, text="Knapsack Capacity")
        self.capacity_var = tk.StringVar(value="100")
        ttk.Label(capacity_frame, text="Capacity:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(capacity_frame, textvariable=self.capacity_var, width=10).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Items Frame
        items_frame = ttk.LabelFrame(self.input_tab, text="Items")
        
        # Items Table
        self.items_tree = ttk.Treeview(items_frame, columns=("id", "weight", "value"), show="headings", height=10)
        self.items_tree.heading("id", text="Item ID")
        self.items_tree.heading("weight", text="Weight")
        self.items_tree.heading("value", text="Value")
        self.items_tree.column("id", width=80)
        self.items_tree.column("weight", width=80)
        self.items_tree.column("value", width=80)
        
        # Scrollbar for items table
        scrollbar = ttk.Scrollbar(items_frame, orient="vertical", command=self.items_tree.yview)
        self.items_tree.configure(yscrollcommand=scrollbar.set)
        
        # Add item frame
        add_item_frame = ttk.Frame(items_frame)
        self.weight_var = tk.StringVar()
        self.value_var = tk.StringVar()
        
        ttk.Label(add_item_frame, text="Weight:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(add_item_frame, textvariable=self.weight_var, width=10).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(add_item_frame, text="Value:").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        ttk.Entry(add_item_frame, textvariable=self.value_var, width=10).grid(row=0, column=3, padx=5, pady=5, sticky="w")
        
        add_btn = ttk.Button(add_item_frame, text="Add Item", command=self.add_item)
        remove_btn = ttk.Button(add_item_frame, text="Remove Selected", command=self.remove_item)
        clear_btn = ttk.Button(add_item_frame, text="Clear All", command=self.clear_items)
        
        add_btn.grid(row=0, column=4, padx=5, pady=5)
        remove_btn.grid(row=0, column=5, padx=5, pady=5)
        clear_btn.grid(row=0, column=6, padx=5, pady=5)
        
        # Quick fill with example items
        example_btn = ttk.Button(items_frame, text="Fill with Example Data", command=self.fill_example_data)
        
        # Problem Type Frame
        problem_frame = ttk.LabelFrame(self.input_tab, text="Problem Type")
        self.problem_type = tk.StringVar(value="0-1")
        ttk.Radiobutton(problem_frame, text="0-1 Knapsack (each item can be used at most once)", 
                         variable=self.problem_type, value="0-1").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Radiobutton(problem_frame, text="Unbounded Knapsack (each item can be used multiple times)", 
                         variable=self.problem_type, value="unbounded").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        
        # Run Button Frame
        run_frame = ttk.Frame(self.input_tab)
        self.run_btn = ttk.Button(run_frame, text="Run Algorithm", command=self.run_algorithm)
        self.stop_btn = ttk.Button(run_frame, text="Stop", command=self.stop_algorithm, state="disabled")
        
        self.run_btn.grid(row=0, column=0, padx=5, pady=5)
        self.stop_btn.grid(row=0, column=1, padx=5, pady=5)
    
    def create_visualization_widgets(self):
        # Progress Frame
        progress_frame = ttk.LabelFrame(self.visualization_tab, text="Algorithm Progress")
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", length=300, mode="determinate", variable=self.progress_var)
        
        self.generation_label = ttk.Label(progress_frame, text="Generation: 0")
        self.best_fitness_label = ttk.Label(progress_frame, text="Best Fitness: 0")
        
        # Graph Frame for fitness over generations
        graph_frame = ttk.LabelFrame(self.visualization_tab, text="Fitness over Generations")
        
        # Use Figure with tight_layout for better appearance
        self.figure = plt.Figure(figsize=(9, 5), tight_layout=True)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.ax.set_title("Fitness over Generations")
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Fitness")
        self.ax.grid(True)
        
        # Create empty plots that will be updated
        self.best_fitness_line, = self.ax.plot([], [], 'b-', label='Best Fitness')
        self.avg_fitness_line, = self.ax.plot([], [], 'g-', label='Average Fitness')
        self.ax.legend()
        
        # Log Frame
        log_frame = ttk.LabelFrame(self.visualization_tab, text="Algorithm Log")
        self.log_text = scrolledtext.ScrolledText(log_frame, width=80, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def create_results_widgets(self):
        # Results Frame
        results_frame = ttk.LabelFrame(self.results_tab, text="Solution Results")
        
        self.solution_text = scrolledtext.ScrolledText(results_frame, width=80, height=20)
        self.solution_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Visual representation of knapsack
        knapsack_frame = ttk.LabelFrame(self.results_tab, text="Visual Representation")
        
        self.solution_canvas = tk.Canvas(knapsack_frame, bg="white", width=600, height=300)
        self.solution_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def setup_layout(self):
        self.tab_control.pack(expand=1, fill="both", padx=10, pady=10)
        
        # Input Tab Layout
        capacity_frame = self.input_tab.winfo_children()[0]
        items_frame = self.input_tab.winfo_children()[1]
        problem_frame = self.input_tab.winfo_children()[2]
        run_frame = self.input_tab.winfo_children()[3]
        
        capacity_frame.pack(fill="x", padx=10, pady=5)
        items_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.items_tree.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        scrollbar = items_frame.winfo_children()[1]
        scrollbar.grid(row=0, column=1, padx=0, pady=5, sticky="ns")
        
        add_item_frame = items_frame.winfo_children()[2]
        add_item_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        
        example_btn = items_frame.winfo_children()[3]
        example_btn.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        
        problem_frame.pack(fill="x", padx=10, pady=5)
        run_frame.pack(padx=10, pady=10)
        
        # Visualization Tab Layout
        progress_frame = self.visualization_tab.winfo_children()[0]
        graph_frame = self.visualization_tab.winfo_children()[1]
        log_frame = self.visualization_tab.winfo_children()[2]
        
        progress_frame.pack(fill="x", padx=10, pady=5)
        self.progress_bar.pack(padx=10, pady=5)
        self.generation_label.pack(padx=10, pady=2)
        self.best_fitness_label.pack(padx=10, pady=2)
        
        graph_frame.pack(fill="both", expand=True, padx=10, pady=5)
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Results Tab Layout
        results_frame = self.results_tab.winfo_children()[0]
        knapsack_frame = self.results_tab.winfo_children()[1]
        
        results_frame.pack(fill="both", expand=True, padx=10, pady=5)
        knapsack_frame.pack(fill="both", expand=True, padx=10, pady=5)
    
    def add_item(self):
        try:
            weight = int(self.weight_var.get())
            value = int(self.value_var.get())
            
            if weight < 0 or value < 0:
                messagebox.showerror("Error", "Weight and value must be non-negative.")
                return
                
            item_id = len(self.items) + 1
            self.items.append(Item(weight, value))
            self.items_tree.insert("", "end", values=(item_id, weight, value))
            
            # Clear entry fields
            self.weight_var.set("")
            self.value_var.set("")
            
        except ValueError:
            messagebox.showerror("Error", "Weight and value must be valid numbers.")
    
    def remove_item(self):
        selected_items = self.items_tree.selection()
        if not selected_items:
            return
            
        # Remove from treeview
        for item_id in selected_items:
            self.items_tree.delete(item_id)
        
        # Rebuild items list from treeview
        self.items = []
        for i, item in enumerate(self.items_tree.get_children()):
            values = self.items_tree.item(item, "values")
            self.items.append(Item(int(values[1]), int(values[2])))
            # Update ID to maintain sequence
            self.items_tree.item(item, values=(i+1, values[1], values[2]))
    
    def clear_items(self):
        for item in self.items_tree.get_children():
            self.items_tree.delete(item)
        self.items = []
    
    def fill_example_data(self):
        self.clear_items()
        
        # Generate random number of items (between 5 and 15)
        num_items = random.randint(5, 15)
        
        # Generate random items with weights between 1-50 and values between 10-200
        for i in range(num_items):
            weight = random.randint(1, 50)
            # Make value somewhat correlated with weight but with randomness
            base_value = weight * random.randint(3, 8)
            # Add some random variation
            value = max(10, base_value + random.randint(-20, 20))
            
            self.items.append(Item(weight, value))
            self.items_tree.insert("", "end", values=(i+1, weight, value))
            
        self.log_text.delete(1.0, tk.END)
        self.log_text.insert(tk.END, f"Generated {num_items} random items\n")
    
    def run_algorithm(self):
        if not self.items:
            messagebox.showerror("Error", "No items added. Please add items first.")
            return
        
        try:
            capacity = int(self.capacity_var.get())
            if capacity <= 0:
                messagebox.showerror("Error", "Capacity must be positive.")
                return
        except ValueError:
            messagebox.showerror("Error", "Capacity must be a valid number.")
            return
        
        # Update global variables
        global gui_items, gui_capacity, gui_mode
        gui_items = self.items.copy()
        gui_capacity = capacity
        gui_mode = self.problem_type.get()
        
        # Clear previous data
        self.log_text.delete(1.0, tk.END)
        self.solution_text.delete(1.0, tk.END)
        self.solution_canvas.delete("all")
        
        # Switch to visualization tab
        self.tab_control.select(1)
        
        # Disable run button, enable stop button
        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        
        # Start algorithm in a separate thread
        self.algorithm_thread = threading.Thread(
            target=self.run_algorithm_thread,
            args=(self.items.copy(), capacity, self.problem_type.get())
        )
        self.algorithm_thread.daemon = True
        self.algorithm_thread.start()
        
        # Start update timer
        self.start_update_timer()
    
    def run_algorithm_thread(self, items, capacity, mode):
        self.log("Starting genetic algorithm for " + mode + " knapsack problem...")
        solution, fitness = genetic_algorithm_with_gui(items, capacity, mode, update_interval=0.1)
        self.log(f"Algorithm completed. Best fitness: {fitness}")
        
        # Switch to results tab when done
        self.root.after(0, lambda: self.tab_control.select(2))
        
        # Display results
        self.root.after(0, lambda: self.display_results(items, solution, capacity, fitness))
        
        # Reset buttons
        self.root.after(0, lambda: self.run_btn.config(state="normal"))
        self.root.after(0, lambda: self.stop_btn.config(state="disabled"))
    
    def stop_algorithm(self):
        global gui_running
        gui_running = False
        
        self.log("Algorithm stopped by user.")
        self.run_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        
        if self.update_timer:
            self.root.after_cancel(self.update_timer)
            self.update_timer = None
    
    def start_update_timer(self):
        self.update_gui()
    
    def update_gui(self):
        if not gui_running and self.update_timer:
            self.root.after_cancel(self.update_timer)
            self.update_timer = None
            return
        
        # Update progress
        if GENERATIONS > 0:  # Avoid division by zero
            progress = min(100, (gui_current_generation / GENERATIONS) * 100)
            self.progress_var.set(progress)
        
        # Update labels
        self.generation_label.config(text=f"Generation: {gui_current_generation}")
        self.best_fitness_label.config(text=f"Best Fitness: {gui_best_fitness}")
        
        # Update graph if we have data
        if gui_generation_data:
            self.update_graph()
        
        # Schedule next update (more frequent updates for smoother visualization)
        self.update_timer = self.root.after(50, self.update_gui)
    
    def update_graph(self):
        if not gui_generation_data:
            return
        
        # Extract data for plotting
        generations = [g for g, _, _ in gui_generation_data]
        best_fitness = [bf for _, bf, _ in gui_generation_data]
        avg_fitness = [af for _, _, af in gui_generation_data]
        
        # Update the data without recreating plots
        self.best_fitness_line.set_data(generations, best_fitness)
        self.avg_fitness_line.set_data(generations, avg_fitness)
        
        # Adjust axes limits dynamically
        if generations:
            self.ax.set_xlim(0, max(generations) * 1.05)
            max_y = max(max(best_fitness + [1]) if best_fitness else 1, 
                        max(avg_fitness + [1]) if avg_fitness else 1)
            self.ax.set_ylim(0, max_y * 1.1)
        
        # Redraw canvas
        self.canvas.draw_idle()
    
    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
    
    def display_results(self, items, solution, capacity, fitness):
        if not solution:
            self.solution_text.insert(tk.END, "No valid solution found.")
            return
        
        total_weight = sum(solution[i] * items[i].weight for i in range(len(solution)))
        
        result_text = f"Maximum value: {fitness}\n"
        result_text += f"Knapsack capacity: {capacity}\n"
        result_text += f"Total weight: {total_weight}/{capacity}\n\n"
        result_text += "Items in knapsack:\n"
        
        for i in range(len(solution)):
            if solution[i] > 0:
                result_text += f"Item {i+1}: {solution[i]} × (Weight: {items[i].weight}, Value: {items[i].value})\n"
        
        self.solution_text.insert(tk.END, result_text)
        
        # Draw visual representation
        self.draw_knapsack_visualization(items, solution, capacity)
    
    def draw_knapsack_visualization(self, items, solution, capacity):
        self.solution_canvas.delete("all")
        
        # Draw knapsack
        knapsack_width = 500
        knapsack_height = 200
        knapsack_x = 50
        knapsack_y = 50
        
        # Calculate total weight and value
        total_weight = sum(solution[i] * items[i].weight for i in range(len(solution)))
        total_value = sum(solution[i] * items[i].value for i in range(len(solution)))
        
        # Draw knapsack outline
        self.solution_canvas.create_rectangle(
            knapsack_x, knapsack_y, 
            knapsack_x + knapsack_width, knapsack_y + knapsack_height, 
            outline="black", width=2
        )
        
        # Draw fill level based on capacity usage
        fill_width = (total_weight / capacity) * knapsack_width
        self.solution_canvas.create_rectangle(
            knapsack_x, knapsack_y,
            knapsack_x + fill_width, knapsack_y + knapsack_height,
            fill="lightblue", outline=""
        )
        
        # Draw capacity text
        self.solution_canvas.create_text(
            knapsack_x + knapsack_width // 2, knapsack_y + knapsack_height + 20,
            text=f"Capacity Used: {total_weight}/{capacity} ({total_weight/capacity*100:.1f}%)",
            fill="black", font=("Arial", 10)
        )
        
        # Draw value text
        self.solution_canvas.create_text(
            knapsack_x + knapsack_width // 2, knapsack_y + knapsack_height + 40,
            text=f"Total Value: {total_value}",
            fill="black", font=("Arial", 10, "bold")
        )
        
        # Draw items
        item_colors = ["#FF5733", "#33FF57", "#3357FF", "#F3FF33", "#FF33F3", "#33FFF3", "#F333FF"]
        
        items_included = [(i, items[i], solution[i]) for i in range(len(solution)) if solution[i] > 0]
        
        legend_y = knapsack_y + knapsack_height + 70
        legend_x = knapsack_x
        
        for idx, (i, item, count) in enumerate(items_included):
            color = item_colors[idx % len(item_colors)]
            
            # Add to legend
            self.solution_canvas.create_rectangle(
                legend_x, legend_y, legend_x + 20, legend_y + 20,
                fill=color, outline="black"
            )
            self.solution_canvas.create_text(
                legend_x + 25, legend_y + 10,
                text=f"Item {i+1}: {count} × (W: {item.weight}, V: {item.value})",
                anchor="w", font=("Arial", 9)
            )
            
            legend_y += 25
            if legend_y > knapsack_y + knapsack_height + 220:
                legend_y = knapsack_y + knapsack_height + 70
                legend_x += 250

def main():
    root = tk.Tk()
    app = KnapsackGUI(root)
    root.mainloop()

def cli_main():
    # Get user input for items and capacity
    print("Knapsack Problem Solver using Genetic Algorithm")
    print("-----------------------------------------------")
    
    # Get knapsack capacity
    while True:
        try:
            capacity = int(input("Enter knapsack capacity: "))
            if capacity <= 0:
                print("Capacity must be positive.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")
    
    # Get number of items
    while True:
        try:
            num_items = int(input("Enter number of items: "))
            if num_items <= 0:
                print("Number of items must be positive.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")
    
    # Input weights and values for each item
    user_items = []
    for i in range(num_items):
        print(f"\nItem {i+1}:")
        while True:
            try:
                weight = int(input(f"  Weight: "))
                if weight < 0:
                    print("  Weight must be non-negative.")
                    continue
                break
            except ValueError:
                print("  Please enter a valid number.")
        
        while True:
            try:
                value = int(input(f"  Value: "))
                if value < 0:
                    print("  Value must be non-negative.")
                    continue
                break
            except ValueError:
                print("  Please enter a valid number.")
        
        user_items.append(Item(weight, value))
    
    # Show summary of items
    print("\nItems Summary:")
    print("--------------")
    for i, item in enumerate(user_items):
        print(f"Item {i+1}: Weight = {item.weight}, Value = {item.value}")
    print(f"Knapsack capacity: {capacity}")
    
    # Allow user to choose problem type
    print("\nSelect problem type:")
    print("1. 0-1 Knapsack (each item can be used at most once)")
    print("2. Unbounded Knapsack (each item can be used multiple times)")
    
    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice == '1':
            mode = '0-1'
            break
        elif choice == '2':
            mode = 'unbounded'
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")
    
    print(f"\nRunning genetic algorithm for {mode} knapsack problem...")
    best_solution, best_fitness = genetic_algorithm(user_items, capacity, mode)
    
    print("\nBest solution found:")
    print_solution(user_items, best_solution, capacity, best_fitness)

if __name__ == "__main__":
    # Command-line arguments could be used to choose between GUI and CLI
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        cli_main()
    else:
        main()
