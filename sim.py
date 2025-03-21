import pandas as pd
import numpy as np
from collections import defaultdict
import random
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns

import cProfile


class TournamentSimulator:
    def __init__(
        self,
        teams_data_path: str,
        bracket_path: str = None,
        num_simulations: int = 10000,
    ):
        """
        Initialize the tournament simulator with team data and bracket structure

        Args:
            teams_data_path: Path to the CSV file containing team data
            bracket_path: Optional path to a file defining the tournament bracket structure
            num_simulations: Number of Monte Carlo simulations to run
        """
        self.teams_data = None
        self.teams_data_path = teams_data_path
        self.bracket_path = bracket_path
        self.num_simulations = num_simulations
        self.bracket = None
        self.results = defaultdict(int)
        self.final_four_appearances = defaultdict(int)
        self.elite_eight_appearances = defaultdict(int)
        self.sweet_sixteen_appearances = defaultdict(int)
        self.regions = None

    def load_data(self) -> None:
        """Load team data from CSV file and preprocess it"""
        self.teams_data = pd.read_csv(self.teams_data_path)

    def load_bracket(self) -> None:
        """Load the tournament bracket structure"""
        # In a real implementation, this might load from a file
        # For now, we'll hard-code a simplified NCAA tournament structure

        # This is a simplified representation of the NCAA tournament bracket
        # Each region has 16 teams arranged by seed
        regions = ["East", "West", "South", "Midwest"]
        self.bracket = {}

        for region in regions:
            self.bracket[region] = []
            # Create the matchups in a region (1 vs 16, 8 vs 9, etc.)
            matchups = [
                (1, 16),
                (8, 9),
                (5, 12),
                (4, 13),
                (6, 11),
                (3, 14),
                (7, 10),
                (2, 15),
            ]

            for seed1, seed2 in matchups:
                # In a real implementation, you would assign actual teams here
                # based on the tournament seeding
                team1 = self._get_team_by_seed(seed1, region)
                team2 = self._get_team_by_seed(seed2, region)
                self.bracket[region].append((team1, team2))

    def _get_team_by_seed(self, seed: int, region: str) -> str:
        """Get a team of the specified seed for the given region"""

        # cache team region and seed
        if self.regions is None:
            self.regions = {}
            for _, team in self.teams_data.iterrows():
                team_region = team["Region"]
                team_seed = team["Seed"]
                if team_region not in self.regions:
                    self.regions[team_region] = {}
                self.regions[team_region][team_seed] = team["Team"]

        return self.regions[region][seed]

    def simulate_game(self, team1: str, team2: str) -> str:
        """
        Simulate a single game between two teams

        Args:
            team1: ID of the first team
            team2: ID of the second team

        Returns:
            The ID of the winning team
        """

        # Get team data
        team1_data = self.teams_data[self.teams_data["Team"] == team1]
        team2_data = self.teams_data[self.teams_data["Team"] == team2]

        # Use placeholder data if the team isn't in our dataset
        if team1_data.empty:
            print(team1)
            raise BaseException()
        if team2_data.empty:
            print(team2)
            raise BaseException()

        # Calculate win probability based on a combination of factors
        # We'll use a weighted combination of NetRtg, KenPom rating, and seed
        # The specific weights would be determined through analysis in a real implementation

        # Extract the metrics we'll use for prediction
        team1_netrtg = team1_data["Sched-NetRtg"].values[0]
        team2_netrtg = team2_data["Sched-NetRtg"].values[0]
        team1_kenpom = team1_data["KenPom-NetRtg"].values[0]
        team2_kenpom = team2_data["KenPom-NetRtg"].values[0]
        team1_seed = team1_data["Seed"].values[0]
        team2_seed = team2_data["Seed"].values[0]

        # Calculate advantage metrics (positive means team1 has advantage)
        netrtg_diff = team1_netrtg - team2_netrtg
        kenpom_diff = team1_kenpom - team2_kenpom
        seed_diff = team2_seed - team1_seed  # Lower seed is better

        # Weight each rating
        netrtg_weight = 0.5
        kenpom_weight = 0.5
        seed_weight = 0.0

        # scale differences in ratings
        netrtg_scale = 5
        kenpom_scale = 10
        seed_scale = 15  # max difference

        # Composite advantage
        advantage = (
            (netrtg_diff / netrtg_scale) * netrtg_weight
            + (kenpom_diff / kenpom_scale) * kenpom_weight
            + (seed_diff / seed_scale) * seed_weight
        )

        # Logistic probability for team 1 to win
        win_prob = 1 / (1 + np.exp(-advantage * 2))

        # Games with higher seeds have more random upsets
        upset_factor = max(team1_seed, team2_seed) / 8
        randomness = np.random.normal(0, 0.1) * upset_factor

        # Clamp win prob betwee 0.05 and 0.95
        win_prob = max(0.05, min(0.95, win_prob + randomness))

        # print(f"{team1} v {team2} randomness={randomness} win_prob={win_prob}")

        # Simulate the game
        if random.random() < win_prob:
            return team1
        else:
            return team2

    def simulate_round(self, matchups: List[Tuple[str, str]]) -> List[str]:
        """
        Simulate a round of the tournament

        Args:
            matchups: List of team pairs for this round

        Returns:
            List of winners who advance to the next round
        """
        winners = []
        for team1, team2 in matchups:
            winner = self.simulate_game(team1, team2)
            winners.append(winner)
        return winners

    def simulate_region(self, region: str) -> str:
        """
        Simulate all rounds in a region

        Args:
            region: Name of the region to simulate

        Returns:
            The team that wins the region
        """
        # Start with the first round matchups for this region
        matchups = self.bracket[region]

        # Track sweet 16, elite 8 appearances for this simulation
        round_teams = [team for matchup in matchups for team in matchup]

        # First round (Round of 64)
        winners = self.simulate_round(matchups)

        # Second round (Round of 32)
        matchups = [(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)]
        winners = self.simulate_round(matchups)

        # Sweet 16
        sweet_16_teams = winners.copy()
        for team in sweet_16_teams:
            self.sweet_sixteen_appearances[team] += 1

        matchups = [(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)]
        winners = self.simulate_round(matchups)

        # Elite 8
        elite_8_teams = winners.copy()
        for team in elite_8_teams:
            self.elite_eight_appearances[team] += 1

        # Regional Final
        matchups = [(winners[0], winners[1])]
        regional_winner = self.simulate_round(matchups)[0]

        # Record Final Four appearance
        self.final_four_appearances[regional_winner] += 1

        return regional_winner

    def simulate_tournament(self) -> str:
        """
        Simulate the entire tournament once

        Returns:
            The champion team ID
        """
        # Simulate each region to get the Final Four
        final_four = []
        for region in self.bracket:
            regional_winner = self.simulate_region(region)
            final_four.append(regional_winner)

        # Simulate Final Four (semifinals)
        # Match regions in a predetermined way (e.g., East vs West, South vs Midwest)
        semifinal1 = (final_four[0], final_four[1])
        semifinal2 = (final_four[2], final_four[3])

        finalist1 = self.simulate_game(*semifinal1)
        finalist2 = self.simulate_game(*semifinal2)

        # Simulate Championship game
        champion = self.simulate_game(finalist1, finalist2)

        return champion

    def run_simulations(self) -> Dict[str, Any]:
        """
        Run multiple tournament simulations and collect statistics

        Returns:
            Dictionary with simulation results
        """
        # Initialize tracking dictionaries
        self.results = defaultdict(int)
        self.final_four_appearances = defaultdict(int)
        self.elite_eight_appearances = defaultdict(int)
        self.sweet_sixteen_appearances = defaultdict(int)

        # Run simulations
        for i in range(self.num_simulations):
            if i % 10 == 0:
                print(i)
            # Reset the bracket for each simulation
            self.load_bracket()

            # Simulate the tournament and record the champion
            champion = self.simulate_tournament()
            self.results[champion] += 1

        # Calculate probabilities
        total_sims = self.num_simulations
        champion_probs = {
            team: count / total_sims for team, count in self.results.items()
        }
        ff_probs = {
            team: count / total_sims
            for team, count in self.final_four_appearances.items()
        }
        e8_probs = {
            team: count / total_sims
            for team, count in self.elite_eight_appearances.items()
        }
        s16_probs = {
            team: count / total_sims
            for team, count in self.sweet_sixteen_appearances.items()
        }

        # Sort results by probability
        champion_probs = dict(
            sorted(champion_probs.items(), key=lambda x: x[1], reverse=True)
        )
        ff_probs = dict(sorted(ff_probs.items(), key=lambda x: x[1], reverse=True))
        e8_probs = dict(sorted(e8_probs.items(), key=lambda x: x[1], reverse=True))
        s16_probs = dict(sorted(s16_probs.items(), key=lambda x: x[1], reverse=True))

        return {
            "champion_probabilities": champion_probs,
            "final_four_probabilities": ff_probs,
            "elite_eight_probabilities": e8_probs,
            "sweet_sixteen_probabilities": s16_probs,
        }

    def visualize_results(
        self, results: Dict[str, Dict[str, float]], top_n: int = 10
    ) -> None:
        """
        Visualize the simulation results

        Args:
            results: Dictionary with simulation results
            top_n: Number of top teams to display in each visualization
        """
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Tournament Simulation Results", fontsize=16)

        # Plot championship probabilities
        self._plot_probabilities(
            axes[0, 0],
            results["champion_probabilities"],
            "Championship Probability",
            top_n,
        )

        # Plot Final Four probabilities
        self._plot_probabilities(
            axes[0, 1],
            results["final_four_probabilities"],
            "Final Four Probability",
            top_n,
        )

        # Plot Elite Eight probabilities
        self._plot_probabilities(
            axes[1, 0],
            results["elite_eight_probabilities"],
            "Elite Eight Probability",
            top_n,
        )

        # Plot Sweet Sixteen probabilities
        self._plot_probabilities(
            axes[1, 1],
            results["sweet_sixteen_probabilities"],
            "Sweet Sixteen Probability",
            top_n,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig("tournament_simulation_results.png", dpi=300, bbox_inches="tight")
        plt.show()

    def _plot_probabilities(
        self, ax, prob_dict: Dict[str, float], title: str, top_n: int
    ) -> None:
        """Helper function to plot probabilities for a given category"""
        # Get top N teams
        top_teams = list(prob_dict.keys())[:top_n]
        probs = [prob_dict[team] * 100 for team in top_teams]  # Convert to percentage

        # Create horizontal bar chart
        bars = ax.barh(
            top_teams, probs, color=sns.color_palette("Blues_r", len(top_teams))
        )

        # Add percentage labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(
                width + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{probs[i]:.1f}%",
                ha="left",
                va="center",
            )

        # Set labels and title
        ax.set_xlabel("Probability (%)")
        ax.set_title(title)

        # Set x-axis limit
        ax.set_xlim(0, max(probs) * 1.1)

        # Remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def print_simulation_summary(
    results: Dict[str, Dict[str, float]], top_n: int = 10
) -> None:
    """
    Print a summary of simulation results

    Args:
        results: Dictionary with simulation results
        top_n: Number of top teams to display
    """
    print("=" * 50)
    print("TOURNAMENT SIMULATION RESULTS")
    print("=" * 50)

    # Print championship probabilities
    print("\nCHAMPIONSHIP PROBABILITIES:")
    print("-" * 30)
    for i, (team, prob) in enumerate(
        list(results["champion_probabilities"].items())[:top_n]
    ):
        print(f"{i+1}. {team}: {prob*100:.2f}%")

    # Print Final Four probabilities
    print("\nFINAL FOUR PROBABILITIES:")
    print("-" * 30)
    for i, (team, prob) in enumerate(
        list(results["final_four_probabilities"].items())[:top_n]
    ):
        print(f"{i+1}. {team}: {prob*100:.2f}%")

    # Print Elite Eight probabilities
    print("\nELITE EIGHT PROBABILITIES:")
    print("-" * 30)
    for i, (team, prob) in enumerate(
        list(results["elite_eight_probabilities"].items())[:top_n]
    ):
        print(f"{i+1}. {team}: {prob*100:.2f}%")

    # Print Sweet Sixteen probabilities
    print("\nSWEET SIXTEEN PROBABILITIES:")
    print("-" * 30)
    for i, (team, prob) in enumerate(
        list(results["sweet_sixteen_probabilities"].items())[:top_n]
    ):
        print(f"{i+1}. {team}: {prob*100:.2f}%")


def main():
    """Main function to run the tournament simulation"""
    # Set up the simulator
    teams_data_path = "kenpom.csv"
    simulator = TournamentSimulator(
        teams_data_path=teams_data_path, num_simulations=100
    )

    # Load data and bracket
    simulator.load_data()
    simulator.load_bracket()

    # Run simulations
    results = simulator.run_simulations()

    # Print and visualize results
    print_simulation_summary(results, 10)
    # simulator.visualize_results(results)


if __name__ == "__main__":
    with cProfile.Profile() as pr:
        main()
        pr.print_stats(sort="cumtime")
    # main()
