from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
from func_timeout import func_timeout, FunctionTimedOut
import random
import time


def smart_heuristic(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)
    opponent = env.get_robot((robot_id + 1) % 2)

    credit_diff = robot.credit - opponent.credit
    h_val = credit_diff * 30

    needs_charge = False

    dist_to_stations = [manhattan_distance(robot.position, s.position) for s in env.charge_stations]
    closest_station_dist = min(dist_to_stations)

    if robot.package is not None:
        dist_to_dest = manhattan_distance(robot.position, robot.package.destination)

        h_val += 100 / (dist_to_dest + 1)

        if robot.battery < dist_to_dest:
            needs_charge = True
            h_val -= (closest_station_dist * 15)
    else:
        available_packages = [pkg for pkg in env.packages if pkg.on_board]

        closest_package_dist = float('inf')
        for pkg in available_packages:
            dist_to_pkg = manhattan_distance(robot.position, pkg.position)
            if dist_to_pkg < closest_package_dist:
                closest_package_dist = dist_to_pkg

        if robot.battery > closest_package_dist:
            h_val += 10 / (closest_package_dist + 1)
        else:
            needs_charge = True
            h_val -= (closest_station_dist * 15)

    if closest_station_dist == 0 and needs_charge:
        h_val *= 10

    return h_val


class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 4
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        opponent_id = (agent_id + 1) % 2
        deadline = time.time() + time_limit - 0.2
        best_op = None
        current_depth = 1
        while time.time() < deadline and current_depth <= 7:
            try:
                best_op, val = func_timeout(deadline - time.time(), self.minimax, args=(env, agent_id, opponent_id, current_depth, True))
                current_depth += 1
            except FunctionTimedOut:
                break
        return best_op
    def minimax(self, env, robot_id, opponent_id, depth, isMax):
        if depth == 0:
            return None, smart_heuristic(env, robot_id)
        if isMax:
            return self.maximizer(env, robot_id, opponent_id, depth)
        else:
            return self.minimizer(env, robot_id, opponent_id, depth)

    def maximizer(self, env, robot_id, opponent_id, depth):
        operators , children = self.successors(env, robot_id)
        max_val = float('-inf')
        best_op = None
        for i in range(len(operators)):
            op, val = self.minimax(children[i], robot_id, opponent_id, depth - 1, False)
            if val > max_val:
                max_val = val
                best_op = operators[i]
        return best_op, max_val

    def minimizer(self, env, robot_id, opponent_id, depth):
        operators, children = self.successors(env, opponent_id)
        min_val = float('inf')
        best_op = None
        for i in range(len(operators)):
            op, val = self.minimax(children[i], robot_id, opponent_id, depth - 1, True)
            if val < min_val:
                min_val = val
                best_op = operators[i]
        return best_op, min_val


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        opponent_id = (agent_id + 1) % 2
        deadline = time.time() + time_limit - 0.2
        best_op = None
        current_depth = 1
        while time.time() < deadline and current_depth <= 7:
            try:
                best_op, val = func_timeout(deadline - time.time(), self.minimax, args=(env, agent_id, opponent_id, current_depth, True, float('-inf'), float('inf')))
                current_depth += 1
            except FunctionTimedOut:
                break
        return best_op

    def minimax(self, env, robot_id, opponent_id, depth, isMax, alpha, beta):
        if depth == 0:
            return None, smart_heuristic(env, robot_id)
        if isMax:
            return self.maximizer(env, robot_id, opponent_id, depth, alpha, beta)
        else:
            return self.minimizer(env, robot_id, opponent_id, depth, alpha, beta)

    def maximizer(self, env, robot_id, opponent_id, depth, alpha, beta):
        operators, children = self.successors(env, robot_id)
        max_val = float('-inf')
        best_op = None
        for i in range(len(operators)):
            op, val = self.minimax(children[i], robot_id, opponent_id, depth - 1, False, alpha, beta)
            if val > max_val:
                max_val = val
                best_op = operators[i]
            if max_val > alpha:
                alpha = max_val
            if max_val >= beta:
                return best_op, float('inf')
        return best_op, max_val

    def minimizer(self, env, robot_id, opponent_id, depth, alpha, beta):
        operators, children = self.successors(env, opponent_id)
        min_val = float('inf')
        best_op = None
        for i in range(len(operators)):
            op, val = self.minimax(children[i], robot_id, opponent_id, depth - 1, True, alpha, beta)
            if val < min_val:
                min_val = val
                best_op = operators[i]
            if min_val < beta:
                beta = min_val
            if min_val <= alpha:
                return best_op, float('-inf')
        return best_op, min_val


class AgentExpectimax(Agent):
    # TODO: section d : 3
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        opponent_id = (agent_id + 1) % 2
        deadline = time.time() + time_limit - 0.2
        best_op = None
        current_depth = 1
        while time.time() < deadline and current_depth <= 7:
            try:
                best_op, val = func_timeout(deadline - time.time(), self.expectimax, args=(env, agent_id, opponent_id, current_depth, True))
                current_depth += 1
            except FunctionTimedOut:
                break
        return best_op

    def expectimax(self, env, robot_id, opponent_id, depth, isMax):
        if depth == 0:
            return None, smart_heuristic(env, robot_id)
        if isMax:
            return self.maximizer(env, robot_id, opponent_id, depth)
        else:
            return self.expect(env, robot_id, opponent_id, depth)

    def maximizer(self, env, robot_id, opponent_id, depth):
        operators, children = self.successors(env, robot_id)
        max_val = float('-inf')
        best_op = None
        for i in range(len(operators)):
            op, val = self.expectimax(children[i], robot_id, opponent_id, depth - 1, False)
            if val > max_val:
                max_val = val
                best_op = operators[i]
        return best_op, max_val

    def expect(self, env, robot_id, opponent_id, depth):
        operators, children = self.successors(env, opponent_id)
        sum = 0
        count = 0
        for i in range(len(operators)):
            op, val = self.expectimax(children[i], robot_id, opponent_id, depth - 1, True)
            if op == "pick up" or op == "move west":
                count += 3
                sum += 3*val
            else:
                count += 1
                sum += val
        return None, sum/count


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)


