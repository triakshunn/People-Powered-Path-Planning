#!/usr/bin/env python3
import numpy as np
import time
import argparse
from scipy.optimize import minimize
from shapely.geometry import Point, LineString
from typing import Tuple, Optional, List, Dict

try:
    from simulator import CrowdSimulator, Agent, SocialForceModel, create_test_scenario
except ImportError:
    print("Error: 'simulator.py' not found. Please ensure it's in the same directory.")
    exit()

class MPCLocalPlanner:
    """
    Core MPC optimizer.
    """
    def __init__(self, simulation: CrowdSimulator, horizon: int, dt: float, max_speed: float, static_obstacles=None, use_sm_cost=False, wg= 1.0, ws = 1.0, wd=1.0, wps = 0.0):
        self.simulation = simulation
        self.robot = simulation.robot
        self.horizon = horizon
        self.dt = dt
        self.max_speed = max_speed
        self.static_obs = static_obstacles or []
        self.goal = np.array(self.robot.get_goal())
        self.start_pos = np.array(self.robot.get_position())
        self.ps_sigma_front = 1.2
        self.ps_sigma_side  = 0.7
        self.ps_sigma_rear  = 0.5
        self.negative_momentum_const = 100
        self.use_sm_cost = use_sm_cost
        self.wg = wg
        self.ws = ws
        self.wd = wd
        self.wps = wps

    def objective_function(self, u_sequence: np.ndarray, current_state: np.ndarray) -> float:
        u_sequence = u_sequence.reshape(self.horizon, 2)
        trajectory = self._simulate_robot_trajectory(current_state, u_sequence)
        total_cost = 0.0
        for t, state_t in enumerate(trajectory):
            if self.use_sm_cost:
                goal_cost = self._goal_cost(state_t)
                if t > 0:
                    sm_cost = 10 * self._social_momentum_cost(state_t, trajectory[t-1], u_sequence[t], t)
                    total_cost += 0.3 * goal_cost + 0.7 * sm_cost
                else:
                    total_cost += goal_cost
            else:
                total_cost += self.wg * self._goal_cost(state_t)
                total_cost += self.ws * self._static_obstacle_cost(state_t)
                total_cost += self.wd * self._dynamic_obstacle_cost(state_t, t)
                total_cost += self.wps * self._personal_space_cost(state_t, t)

        return total_cost

    def _simulate_robot_trajectory(self, initial_state: np.ndarray, controls: np.ndarray) -> np.ndarray:
        trajectory = np.zeros((self.horizon, 2))
        state = initial_state
        #TODO Task 1.1 
        # iterate over the horizon (controls array):
        for t in range(0, len(controls)):
            u_t= controls[t]
            new_state = state + u_t * self.dt
            state = new_state
            trajectory[t] = state

        # Read u_t from `controls[t]`.
        # Update state with simple euler update ut*dt
        # Store into trajectory[t] = state

        return trajectory

    def _goal_cost(self, state: np.ndarray) -> float:
        # TODO Task 1.1
        # Compute Euclidean distance between `state` and `g` (numerator)
        # Compute Euclidean distance between `self.start_pos` and `g` as the denominator =  ||self.goal-self.start_pos|| + 1e-6
        goal_cost = 0

        start_x = self.start_pos[0]
        start_y = self.start_pos[1]
        curr_x = state[0]
        curr_y = state[1]
        goal_x = self.goal[0]
        goal_y = self.goal[1]

        numerator = np.sqrt((curr_x - goal_x)**2 + (curr_y - goal_y)**2)
        denominator = np.sqrt((start_x - goal_x)**2 + (start_y - goal_y)**2) + 1e-6

        goal_cost = numerator / denominator

        return goal_cost

    def _dynamic_obstacle_cost(self, state: np.ndarray, t: int) -> float:
        cost = 0.0
        robot_pos = state
        for human in self.simulation.humans:
            # TODO Task 1.3
            # For each human h at planning step t
            # Predict **next-step** human position with a constant-velocity model use human.get_position() and human.get_velocity()
            predicted_position = human.get_position() + human.get_velocity() * self.dt

            # Compute distance from the robot_pos 
            distance = np.sqrt((predicted_position[0] - robot_pos[0])**2 + (predicted_position[1] - robot_pos[1])**2)

            # Define safe separation: d_safe = r_robot + r_human + margin (1.5)
            margin = 1.5
            d_safe = self.robot.radius + human.radius + margin

            # Accumulate cost based on the equation in notebook (alpha = 1, delta = 1e-6):
            # Return total cost over all humans.
            alpha = 1
            delta = 1e-6

            if(distance < d_safe):
                cost += alpha * (d_safe - distance) / (distance + delta)
        return cost

    def _static_obstacle_cost(self, state: np.ndarray) -> float:
        if not self.static_obs:
            return 0.0
        cost = 0.0
        robot_point = Point(state)

        for obs in self.static_obs:
            # TODO Task 1.2
            #  Each obstacle is a line segment with endpoints `obs.p1` and `obs.p2` 
            #  Use LineString from shapely to create a line object from the obstacle
            line_segment = LineString([obs.p1, obs.p2])

            #  Compute Euclidean distance `d` from the robot point to this segment using LineString.distance(point) on the obstacle line and robot_point
            d = line_segment.distance(robot_point)

            #  Collision check: If `d < self.robot.radius`: immediately return `C_collide (1e6)`.
            if(d < self.robot.radius):
                return 1e6

            #  Long-range repulsion: Define `d_safe = self.robot.radius + 1.5` (meters).
            d_safe = self.robot.radius + 1.5


            #  If `d < d_safe`:
            #  Accumulate cost based on the equation in notebook (gamma = 1, delta = 1e-3)
            if(d < d_safe):
                gamma = 1
                delta = 1e-3
                cost += gamma * (d_safe - d) / (d + delta)

        return cost
    
    def _personal_space_cost(self, state_t: np.ndarray, t: int) -> float:
        """
        Kirby-style asymmetric Gaussian personal space around each human.
        - Oriented by the human's heading.
        - Asymmetric along forward (front) vs backward (rear) axis.
        - Optional dynamic elongation with human speed.

        Returns a nonnegative scalar cost (sum over humans).
        """
        eps = 1e-6
        cost = 0.0
        p_r = np.asarray(state_t, dtype=float)

        for human in getattr(self.simulation, "humans", []):
            p_h0 = np.asarray(human.get_position(), dtype=float)
            v_h  = np.asarray(human.get_velocity(), dtype=float)
            # TODO Task 3.1
            # Predict human position at step t (constant-velocity):
            p_ht=p_h0+v_h*t*self.dt
            # Compute robot offset in human's **body frame** (x_b, y_b):
            # Calculate the Heading direction of the  human from vh (if nearly static, keep heading 0 by default
            theta=np.arctan2(v_h[1],v_h[0]) if np.linalg.norm(v_h)>eps else 0
            # Vector robot relative to human in world frame (dx,dy)
            d_world=p_r-p_ht
            # Rotate into human body frame: (x_b, y_b) = R(-theta) * d_world where R is the 2D rotation matrix
            c, s = np.cos(-theta), np.sin(-theta)
            R = np.array([[c, -s], [s, c]])
            d_body = R @ d_world
            # Select asymmetric sigmas, use self.ps_sigma_front,self.ps_sigma_rear, self.ps_sigma_side
            x,y=d_body
            sigma_front=self.ps_sigma_front
            sigma_rear=self.ps_sigma_rear
            sigma_side=self.ps_sigma_side
            # sigma_x = sigma_front if x >= 0 else sigma_rear
            if x >= 0:
                sigma_x = sigma_front
            else:
                sigma_x = sigma_rear
            # sigma_y = sigma_side
            sigma_y = sigma_side
            # calculate per-human contribution (unnormalized 2D Gaussian) based on the equation from the notebook add eps(1e-6) to sigma_x^2 and sigma_y^2 for stability
            cost += float(0)
            cost+=np.exp(-0.5*(x**2/(sigma_x**2+eps)+y**2/(sigma_y**2+eps)))

        return cost
    def get_interacting_agents(self, state: np.ndarray):
        interacting_agents, agent_weights = [], []
        robot_theta = np.arctan2(self.robot.gy - self.robot.py, self.robot.gx - self.robot.px)
        robot_heading_degrees = np.degrees(robot_theta)

        FOV = 80.0  # degrees
        MAX_RANGE = 7.0
        eps = 1e-6

        for human in self.simulation.humans:
            direction_to_agent = np.degrees(np.arctan2(human.py - state[1], human.px - state[0]))
            distance_to_agent  = float(np.hypot(human.py - state[1], human.px - state[0]))

            rel_angle = (direction_to_agent - robot_heading_degrees + 180.0) % 360.0 - 180.0
            if (-FOV <= rel_angle <= FOV) and (distance_to_agent < MAX_RANGE):
                interacting_agents.append(human)
                agent_weights.append(min(1.0 / (distance_to_agent + eps), 2.0))

        return interacting_agents, agent_weights

    def _social_momentum_cost(self,state,prev_state, u, t):
        cost = 0.0
        interacting_agents, agent_weights = self.get_interacting_agents(state)
        if interacting_agents:
            for i, agent in enumerate(interacting_agents):
                # TODO Task 4.1
                # predict human positions (constant velocity) for timestep t
                h_i_t = agent.get_position() + agent.get_velocity() * t * self.dt
                h_i_t_1 = agent.get_position() + agent.get_velocity() * (t+1) * self.dt

                # find the midpoint of the human and the previous robot state (prev_state)
                # find the relative vector from robot-to-center (r_ac) and human-to-center (r_bc)
                r_t_1 = prev_state
                r_t = state
                c_i_t = .5 * (r_t_1 + h_i_t)
                c_i_t_1 = .5 * (state + h_i_t_1)

                a_i_t = r_t_1 - c_i_t
                b_i_t = h_i_t - c_i_t

                a_hat_i_t_1 = r_t - c_i_t_1
                b_hat_i_t_1 = h_i_t_1 - c_i_t_1
                
                # calculate the social momentum scalars using simulators robot.get_velocity() and agent.get_velocity() for human
                # Repeat the process for the predicted timestep t+1 by calculating predicted humans poses for t+1
                def cross(x, y):
                    return x[0]*y[1] - x[1]*y[0]
                
                v_r = self.robot.get_velocity()
                v_i_h = agent.get_velocity()
                L_i_t = cross(a_i_t, v_r) + cross(b_i_t, v_i_h)

                u_t = u
                L_hat_i_t_1 = cross(a_hat_i_t_1, u_t) + cross(b_hat_i_t_1, v_i_h)
                
                # calculate midpoint and relative vectors again but now with current robot state (state)
                # calculate the t+1 social momentum scalars  using u[0],u[1] for robot and agent.get_velocity() for human
                # calculate the Per-human cost update based on the equation mentioned in the notebook where C_neg = self.negative_momentum_const
                C_neg = self.negative_momentum_const
                w_i = self.get_interacting_agents(state)[1][i]


                if(L_i_t * L_hat_i_t_1 > 0):
                    J_i_t = -1 * w_i * abs(L_hat_i_t_1)
                else:
                    J_i_t = C_neg

                # # accumulate total social momentum cost for all interactiong agents
                cost += J_i_t
        else:
            cost = 0
        return cost

    def plan(self) -> np.ndarray:
        current_state = np.array(self.robot.get_position())
        num_controls = self.horizon * 2
        goal_direction = self.goal - current_state
        dist_to_goal = np.linalg.norm(goal_direction)
        if dist_to_goal > 1e-6:
            desired_velocity = (goal_direction / dist_to_goal) * self.max_speed
        else:
            desired_velocity = np.zeros(2)
        
        initial_guess = np.tile(desired_velocity, self.horizon)
        
        bounds = [(-self.max_speed, self.max_speed)] * num_controls
        result = minimize(
                self.objective_function,
                initial_guess,
                args=(current_state,),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 25, 'ftol': 1e-2, 'disp': False}
        )
        return result.x.reshape(self.horizon, 2)


class RobotMPCController:
    """
    A wrapper for the MPC planner that acts as the robot's policy.
    """
    def __init__(self, horizon=8, dt=0.25, max_speed=1.5, use_sm_cost=False, wg= 1.0, ws = 1.0, wd=1.0, wps = 0.0):
        self.horizon = horizon
        self.dt = dt
        self.max_speed = max_speed
        self.use_sm_cost = use_sm_cost
        self.wg = wg
        self.ws = ws
        self.wd = wd
        self.wps = wps
        print(f"MPC Controller initialized with Horizon={horizon}, Timestep={dt}")

    def predict(self, agent: Agent, other_agents: List[Agent], obstacles: List['Obstacle'] = []):
        """
        Aligns with SocialForceModel.predict(agent, other_agents, obstacles).
        Called automatically by CrowdSimulator.step().
        """
        simulation = type("DummySim", (), {})()
        simulation.robot = agent
        simulation.humans = [a for a in other_agents if a.agent_type == "human"]
        simulation.obstacles = obstacles

        planner = MPCLocalPlanner(
            simulation=simulation,
            horizon=self.horizon,
            dt=self.dt,
            max_speed=self.max_speed,
            static_obstacles=obstacles,
            use_sm_cost= self.use_sm_cost,
            ws = self.ws,
            wg = self.wg,
            wd = self.wd,
            wps = self.wps
        )
        optimal_controls = planner.plan()
        vx, vy = optimal_controls[0]
        return (vx, vy)

def run_mpc_demo(args):
    """
    Main function to set up and run the MPC demo.
    """
    sim = CrowdSimulator(time_step=0.25, max_steps=args.max_steps)
    human_starts, human_goals = create_test_scenario(args.scenario)
    robot = sim.add_robot(x=0, y=-5, gx=0, gy=5, v_pref=1.2)
    for start, goal in zip(human_starts, human_goals):
        human = sim.add_human(x=start[0], y=start[1], gx=goal[0], gy=goal[1], v_pref=1.0)
        sim.set_human_policy(human.id, SocialForceModel(v0=0.8, tau = 0.6, A = 5.0, B = 0.05, max_speed=1.0))

    robot_controller = RobotMPCController(horizon=args.horizon, dt=sim.time_step)
    
    print(f"Starting MPC demo: Scenario '{args.scenario}', {len(sim.humans)} humans, Horizon={args.horizon}.")
    start_time = time.time()
    
    for step in range(args.max_steps):
        if sim.done: break
        robot_action = robot_controller.predict(robot, sim)
        sim.step(action=robot_action)
        print(f"Step {step+1}/{args.max_steps} -> Robot Pos: ({robot.px:.2f}, {robot.py:.2f}), Goal Dist: {robot.get_distance_to_goal():.2f}m")

    end_time = time.time()
    print("\n--- Simulation Complete ---")
    print(f"Status: {sim.info.get('status', 'Finished')}")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    if sim.current_step > 0:
        print(f"Average time per step: {(end_time - start_time) / sim.current_step:.3f} seconds")

    if args.visualize:
        print("\nCreating visualization...")
        sim.visualize_simulation(output_file=args.output_file, show_plot=not args.output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Predictive Control (MPC) Demo for Robot Navigation.")
    parser.add_argument('--scenario', type=str, default='easy', choices=['easy', 'medium', 'hard'], help='Simulation scenario for human agents.')
    parser.add_argument('--horizon', type=int, default=8, help='MPC planning horizon.')
    parser.add_argument('--max_steps', type=int, default=120, help='Maximum simulation steps.')
    parser.add_argument('--visualize', action='store_true', help='Show animated visualization of the results.')
    parser.add_argument('--output_file', type=str, default=None, help='Save animation to file (e.g., mpc_demo.mp4).')
    
    args = parser.parse_args()
    run_mpc_demo(args)