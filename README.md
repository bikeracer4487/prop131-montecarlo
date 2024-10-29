# Election Monte Carlo Simulator

Welcome to the **Proposition 131 Primary Monte Carlo Simulator**, a powerful and user-friendly tool designed to simulate primary outcomes under Prop 131 based on various configurable parameters. Whether you're a political analyst, educator, or enthusiast, this simulator provides valuable insights into how different factors might influence primary results under Prop 131.

## Understanding the Interface

Upon launching, you'll be greeted with a sleek, dark-themed interface powered by `customtkinter`. The main components of the interface are organized for easy navigation and configuration.

### 1. Number of Simulated Elections

- **Description:** Defines how many election simulations the tool will run.
- **Options:**
  - `500`
  - `1000`
  - `5000`
  - `10000` (Default)
  - `50000`
  - `100000`
- **Usage:** Select a value based on the desired accuracy and computational resources.

### 2. Size of Voting Population

- **Description:** Sets the total number of voters in each simulated election.
- **Options:**
  - `50`
  - `100`
  - `500`
  - `1000` (Default)
  - `5000`
  - `10000`
- **Usage:** Adjust according to the scale of the election scenario you're modeling.

### 3. Distribution of Voting Electorate

- **Description:** Determines the percentage of Red and Blue voters in the population.
- **Components:**
  - **Slider:** Allows manual adjustment in 5% increments.
  - **Buttons:**
    - `+5% Red`: Increases Red voter percentage by 5%.
    - `+5% Blue`: Increases Blue voter percentage by 5%.
    - `80% Red`: Sets Red voters to 80%.
    - `80% Blue`: Sets Blue voters to 80%.
    - `Reset to 50/50`: Resets distribution to an even split.
  - **Display Label:** Shows the current distribution (e.g., "50% Red").
- **Usage:** Customize voter distribution to reflect different political landscapes.

### 4. Number and Type of Republican Candidates

- **Number of Republican Candidates:**
  - **Description:** Select how many Republican candidates are running.
  - **Options:** `0` to `6` (Default: `2`)
- **Republican Candidate Type:**
  - **Description:** Define the ideological composition of Republican candidates.
  - **Options:**
    - `All Extreme`: All candidates have extreme positions.
    - `All Moderate`: All candidates have moderate positions.
    - `Even Mix`: A balanced mix of extreme and moderate candidates.
- **Usage:** Model different party strategies and candidate profiles.

### 5. Number and Type of Democrat Candidates

- **Number of Democrat Candidates:**
  - **Description:** Select how many Democrat candidates are running.
  - **Options:** `0` to `6` (Default: `2`)
- **Democrat Candidate Type:**
  - **Description:** Define the ideological composition of Democrat candidates.
  - **Options:**
    - `All Extreme`: All candidates have extreme positions.
    - `All Moderate`: All candidates have moderate positions.
    - `Even Mix`: A balanced mix of extreme and moderate candidates.
- **Usage:** Similar to Republican candidates, adjust to simulate different party dynamics.

### 6. Number of 3rd Party Candidates

- **Description:** Determine the number of Third Party candidates participating.
- **Options:** `0` to `6` (Default: `1`)
- **Usage:** Introduce additional variables that can influence the election outcome.

### 7. Funding Disparity Options

- **Republican Funding Disparity:**
  - **Checkbox:** `Assume funding disparity` (Checked by default)
  - **Checkbox:** `Assume heavy funding disparity` (Unchecked by default)
- **Democrat Funding Disparity:**
  - **Checkbox:** `Assume funding disparity` (Checked by default)
  - **Checkbox:** `Assume heavy funding disparity` (Unchecked by default)
- **Description:** Simulate the impact of financial resources on campaign effectiveness.
- **Behavior:**
  - **Standard Funding Disparity:** Moderate influence based on typical campaign resources.
  - **Heavy Funding Disparity:** Significant influence, simulating scenarios where certain candidates have substantial financial advantages.
  - **Note:** Selecting "Heavy funding disparity" disables the standard disparity option to prevent conflicting settings.

### 8. Simulation Controls

- **Run Elections Button:**
  - **Description:** Initiates the simulation based on the configured parameters.
- **Cancel Simulation Button:**
  - **Description:** Stops the ongoing simulation.
  - **State:** Disabled by default and enabled only during an active simulation.

### 9. Results Display

- **Description:** Shows the probability distribution of various election outcomes based on the simulations.
- **Features:**
  - **Formatted Output:** Lists outcomes like "2 Republicans, 1 Democrat: 45.67%".
- **Usage:** Analyze which party combinations are most likely under the given parameters.


---

## Simulation Mechanics

The **Election Monte Carlo Simulator** leverages probabilistic models and random sampling to forecast election outcomes. Here's a high-level overview of how the simulation operates:

1. **Candidate Generation:**
   - Based on the number and type of candidates selected for each party, the simulator creates a pool of candidates with defined ideological positions.

2. **Voter Distribution:**
   - The electorate is divided into Red and Blue voters according to the specified percentages.
   - Each voter's allegiance influences their voting behavior.

3. **Funding Disparity Impact:**
   - Funding levels affect the probability of candidates receiving votes.
   - **Standard Funding Disparity:** Moderate influence based on typical campaign resources. Utilizes a linear voter decay model to distribute votes between like-candidates.
   - **Heavy Funding Disparity:** Significant influence, simulating scenarios where certain candidates have substantial financial advantages. Utilizes a quadratic voter decay model to distribute votes between like-candidates.
   - The algorithms for linear and quadratic voter decay are as follows:
     
     Given:
     - $n$: total number of candidates in a group (ex.: total number of Republican candidates)
     - $k$: candidate number (where 1 $\le$ $k$ $\le$ $n$)
     
     Linear Decay: Define the vote percentage for the $k$-th candidate as:
     
     $P_{k}$ = $\frac{2(n - k + 1)}{n(n + 1)}$

     Quadratic Decay: Define the vote percentage for the $k$-th candidate as:

     $P_{k}$ = $\frac{(n - k + 1)^2}{\sum_{j=1}^{n} j^2}$

  - **No Funding Disparity** (all boxes unchecked): No funding influence, votes are distributed evenly among like-candidates.

4. **Vote Allocation:**
   - Voters cast their votes probabilistically based on candidate profiles and funding disparities.
   - The simulator uses decay functions (linear or quadratic) to model the diminishing influence of candidates as their number increases.

5. **Outcome Determination:**
   - For each simulated election, the top four candidates with the highest vote counts are identified.
   - The parties of these candidates are tallied to determine the composition of the winning combination.

6. **Result Aggregation:**
   - The simulator aggregates the outcomes across all simulations to calculate the probability percentages for each possible party combination.
