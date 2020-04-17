from puzzle import *
from a_star import solve
import datetime
import os
if __name__ == '__main__':
    # we create some start and goal states. the number of actions between them is 25 although a shorter plan of
    initial_state = State(os.linesep.join(["8 6 7", "2 5 4", "3 0 1"]))
    goal_state = State()
    puzzle = Puzzle(initial_state, goal_state)
    solution_start_time = datetime.datetime.now()
    solve(puzzle)
    print('time to solve {}'.format(datetime.datetime.now()-solution_start_time))