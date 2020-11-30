import cProfile
from agents.agent_minimax import generate_move
from main import human_vs_agent

cProfile.run(
    "human_vs_agent(generate_move, generate_move)", "mmab"
)

import pstats

p = pstats.Stats("mmab")
p.sort_stats("tottime").print_stats(50)
