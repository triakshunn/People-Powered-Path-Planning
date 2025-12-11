#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from typing import List, Dict, Tuple, Optional
from IPython.display import HTML
import time
import logging
import json
import os
from datetime import datetime

class Agent: pass
class MultiHumanTracker: pass


class stateutils:
    @staticmethod
    def normalize(vectors: np.ndarray):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms_safe = np.where(norms == 0, 1, norms)
        return vectors / norms_safe, norms.flatten()

class Agent:
    """
    Base class for agents (humans and robots) in the simulation.
    """
    def __init__(self, agent_id: int, x: float, y: float, gx, gy,
                 vx: float = 0.0, vy: float = 0.0, radius: float = 0.3,
                 v_pref: float = 1.0, agent_type: str = 'human', wait_times: list[float]=[]):
        self.id = agent_id
        self.px, self.py = x, y
        self.gx, self.gy = gx, gy
        self.goal_index = 0
        self.vx, self.vy = vx, vy
        self.radius = radius
        self.v_pref = v_pref
        self.agent_type = agent_type
        self.position_history = [(x, y)]
        self.velocity_history = [(vx, vy)]
        self.preferred_velocity = np.array([0.0, 0.0])
        
        # Wait times for each goal (in seconds) - only done if we gave a list of subgoals
        if(isinstance(gx, list)):
            self.wait_times = wait_times if wait_times else [0.0] * len(gx)
            # Ensure wait_times matches number of goals
            while len(self.wait_times) < len(gx):
                self.wait_times.append(0.0)
        else:
            self.wait_times = 0
        
        # Time spent waiting at current goal
        self.current_wait_time = 0.0

        self.is_waiting = False

    def get_position(self) -> Tuple[float, float]:
        return np.array([self.px, self.py])

    def get_velocity(self) -> Tuple[float, float]:
        return np.array([self.vx, self.vy])

    def get_goal(self) -> Tuple[float, float]:
        if(isinstance(self.gx, list)):
            return np.array([self.gx[self.goal_index], self.gy[self.goal_index]])
        else:
            return np.array([self.gx, self.gy])

    def set_position(self, x: float, y: float):
        self.px, self.py = x, y
        self.position_history.append((x, y))

    def set_velocity(self, vx: float, vy: float):
        self.vx, self.vy = vx, vy
        self.velocity_history.append((vx, vy))

    def get_distance_to_goal(self) -> float:
        if(isinstance(self.gx, list)):
            return np.sqrt((self.px - self.gx[self.goal_index])**2 + (self.py - self.gy[self.goal_index])**2)
        else:
            return np.sqrt((self.px - self.gx)**2 + (self.py - self.gy)**2)

    def is_at_goal(self, threshold: float = 0.5) -> bool:
        return self.get_distance_to_goal() < threshold
    
    def should_advance_to_next_goal(self, time_step: float) -> bool:
        """Check if agent has waited long enough at current goal to advance."""
        if not self.is_at_goal():
            self.current_wait_time = 0.0
            self.is_waiting = False
            return False
        
        if not self.is_waiting:
            self.is_waiting = True
            self.current_wait_time = 0.0
        
        self.current_wait_time += time_step

        if(isinstance(self.gx, list)):
            return self.current_wait_time >= self.wait_times[self.goal_index]
        else:
            return True # there's no wait times if you don't have subgoals
    
    def get_preferred_velocity(self) -> np.ndarray:
        """
        Returns the agent's preferred velocity (w), required by the Social Force Model.
        """
        return self.preferred_velocity

    def set_preferred_velocity(self, w: np.ndarray):
        """
        Updates the agent's preferred velocity (w).
        """
        self.preferred_velocity = w

    
class Obstacle:
    """Represents a line segment obstacle."""
    def __init__(self, p1, p2):
        self.p1 = np.array(p1, dtype=float)
        self.p2 = np.array(p2, dtype=float)

class GoalPolicy:
    """
    A simple policy that directs an agent towards its goal at a constant speed,
    ignoring other agents and obstacles.
    """
    def __init__(self, max_speed=1.0):
        self.max_speed = max_speed

    def predict(self, agent, other_agents, obstacles=[]):
        """
        Calculates the velocity vector to move directly to the goal.
        """
        direction_to_goal = agent.get_goal() - agent.get_position()
        distance = np.linalg.norm(direction_to_goal)

        if distance < 1e-6:
            return (0.0, 0.0)

        normalized_direction = direction_to_goal / distance
        desired_velocity = self.max_speed * normalized_direction
        return (desired_velocity[0], desired_velocity[1])

class SocialForceModel:
    """
    Implementation of the Social Force Model.
    """

    def __init__(self, v0: float = 1.0, tau: float = 0.2, A: float = 1000.0, B: float = 0.15,
                 max_speed: float = 2.0, radius: float = 0.3,
                 interaction_range: float = 10.0, time_step: float = 0.25,
                 delta_t: float = 0.2, field_of_view_angle_deg: float = 200.0,
                 c_perception: float = 0.5):  ### interacrion_range_original:3
        self.v0 = v0                   
        self.tau = tau                 
        self.A = A                    
        self.B = B                     
        self.radius = radius          
        self.max_speed = max_speed
        self.interaction_range = interaction_range
        self.time_step = time_step

        self.delta_t = delta_t        
        self.fov_cos_angle = np.cos(np.deg2rad(field_of_view_angle_deg) / 2.0)
        self.c_perception = c_perception 
        self.last_predicted_w = np.array([0.0, 0.0])

    def _compute_desired_acceleration(self, position, goal, current_velocity, desired_speed):
        direction_to_goal = np.array(goal) - np.array(position)
        distance = np.linalg.norm(direction_to_goal)

        if distance < 1e-6:
            return np.array([0.0, 0.0])

        e_i = direction_to_goal / distance
        desired_velocity = desired_speed * e_i
        desired_accleration = (desired_velocity - np.array(current_velocity)) / self.tau
        return desired_accleration
    
    def _compute_social_acceleration(self, agent_pos, agent_desired_dir, other_agents: List['Agent']):
        social_acceleration = np.array([0.0, 0.0])
        for beta in other_agents:
            pos_beta = beta.get_position()
            if np.allclose(agent_pos, pos_beta):
                continue
            
            r_alpha_beta = agent_pos - pos_beta
            d_alpha_beta = np.linalg.norm(r_alpha_beta)
            if d_alpha_beta > self.interaction_range:
                continue

            vel_beta = beta.get_velocity()
            v_beta_norm = np.linalg.norm(vel_beta)
            e_beta = vel_beta / v_beta_norm if v_beta_norm > 1e-4 else np.array([0., 0.])
            
            r_prime = r_alpha_beta - v_beta_norm * self.delta_t * e_beta 
            b = 0.5 * np.sqrt(max(0, (d_alpha_beta + np.linalg.norm(r_prime))**2 - (v_beta_norm * self.delta_t)**2))

            force_mag = self.A * np.exp(-b / self.B)
            n_alpha_beta = r_alpha_beta / d_alpha_beta
            f_alpha_beta = force_mag * n_alpha_beta

            cos_phi = np.dot(agent_desired_dir, -n_alpha_beta)
            if cos_phi < self.fov_cos_angle:
                f_alpha_beta *= self.c_perception

            social_acceleration += f_alpha_beta
            
        return social_acceleration
    
    def _compute_obstacle_acceleration(self, position, agent_desired_dir, obstacles: List['Obstacle']):
        """
        Computes repulsive 'forces' from obstacles.
        """
        obstacle_acceleration = np.array([0.0, 0.0])
        for wall in obstacles:
            p1, p2 = wall.p1, wall.p2
            line_vec = p2 - p1
            line_len_sq = np.dot(line_vec, line_vec)
            
            if line_len_sq < 1e-6: continue
            
            t = np.dot(position - p1, line_vec) / line_len_sq
            t = np.clip(t, 0, 1)
            closest_point = p1 + t * line_vec

            d_iB_vec = position - closest_point
            d_iB = np.linalg.norm(d_iB_vec)

            if d_iB > self.interaction_range or d_iB < 1e-6:
                continue
            
            n_iB = d_iB_vec / d_iB
            force_mag = self.A * np.exp(-d_iB / self.B)
            f_iB = force_mag * n_iB
            if np.dot(agent_desired_dir, -n_iB) < self.fov_cos_angle:
                f_iB *= self.c_perception

            obstacle_acceleration += f_iB
        return obstacle_acceleration
    
    def _compute_group_coherence_force(self, agent: Agent, all_agents: List[Agent], groups: List[List[int]]):
        total_force = np.array([0.0, 0.0])
        return total_force * 3.0  
    
    def _compute_group_repulsive_force(self, agent: Agent, all_agents: List[Agent], groups: List[List[int]], threshold=0.55):

        total_force = np.array([0.0, 0.0])
        return total_force * 2
    
    def get_last_predicted_preferred_velocity(self) -> np.ndarray:
        return self.last_predicted_w

    def predict(self, agent: 'Agent', other_agents: List['Agent'], obstacles: List['Obstacle'] = [], groups: List[List[int]] = []) -> Tuple[float, float]:
        position = agent.get_position()
        goal = agent.get_goal()
        actual_velocity = agent.get_velocity()
        preferred_velocity = agent.get_preferred_velocity()
        desired_speed = self.v0

        direction_to_goal = goal - position
        dist_to_goal = np.linalg.norm(direction_to_goal)
        agent_desired_dir = direction_to_goal / dist_to_goal if dist_to_goal > 1e-6 else np.array([0.,0.])

        desired_acceleration = self._compute_desired_acceleration(position, goal, actual_velocity, desired_speed)
        social_acceleration = self._compute_social_acceleration(position, agent_desired_dir, other_agents)
        obstacle_acceleration = self._compute_obstacle_acceleration(position, agent_desired_dir, obstacles)

        group_coherence = self._compute_group_coherence_force(agent, other_agents, groups)
        group_repulsive = self._compute_group_repulsive_force(agent, other_agents, groups)
        
        total_acceleration = desired_acceleration + social_acceleration + obstacle_acceleration + group_repulsive +  group_coherence 
        noise_magnitude = 0.01 
        noise = noise_magnitude * self.A * (np.random.rand(2) - 0.5)
        total_acceleration+= noise

        new_preferred_velocity = preferred_velocity + total_acceleration * self.time_step
        self.last_predicted_w = new_preferred_velocity

        speed_w = np.linalg.norm(new_preferred_velocity)
        if speed_w > self.max_speed:
            new_actual_velocity = (new_preferred_velocity / speed_w) * self.max_speed
        else:
            new_actual_velocity = new_preferred_velocity

        return new_actual_velocity[0], new_actual_velocity[1]

class CrowdSimulator:
    """
    Simulates the physical movement of agents in a crowd.
    """
    def __init__(self, time_step: float = 0.25, max_steps: int = 200):
        self.time_step = time_step
        self.max_steps = max_steps
        self.current_step = 0
        self.robot: Optional[Agent] = None
        self.obstacles: List[Obstacle] = []
        self.humans: List[Agent] = []
        self.groups: List[List[int]] = []
        self.all_agents: List[Agent] = []
        self.robot_policy: Optional[SocialForceModel] = None
        self.human_policies: Dict[int, SocialForceModel] = {}
        self.done = False
        self.info = {}
        self.leader_ID= []

    def add_robot(self, *args, **kwargs) -> Agent:
        self.robot = Agent(agent_id=0, agent_type='robot', *args, **kwargs)
        self.all_agents.append(self.robot)
        return self.robot

    def add_human(self, *args, **kwargs) -> Agent:
        human_id = len(self.humans) + 1
        human = Agent(agent_id=human_id, agent_type='human', *args, **kwargs)
        self.humans.append(human)
        self.all_agents.append(human)
        return human
    
    def add_obstacle(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> Obstacle:
        obstacle = Obstacle(p1, p2)
        self.obstacles.append(obstacle)
        return obstacle

    def set_robot_policy(self, policy: SocialForceModel):
        self.robot_policy = policy
        self.robot_policy.time_step = self.time_step

    def set_human_policy(self, human_id: int, policy: SocialForceModel):
        self.human_policies[human_id] = policy
        self.human_policies[human_id].time_step = self.time_step
        
    def add_group(self, agent_ids: List[int]):
        """Register a group by list of agent IDs."""
        self.groups.append(agent_ids)

    def has_group(self):
        return len(self.groups) > 0
    
    def step(self, action: Optional[Tuple[float, float]] = None):
        """
        Advances the simulation by one step.
        """
        if self.done:
            return None, 0, self.done, self.info

        next_velocities = {}
        next_preferred_velocities = {}
        for agent in self.all_agents:
            if(isinstance(agent.gx, list)): # if we have multiple subgoals, enable waiting
                # If waiting at goal, set velocity to zero
                if agent.is_at_goal() and not agent.should_advance_to_next_goal(self.time_step):
                    next_velocities[agent.id] = (0.0, 0.0)
                    continue
                
                if agent.is_at_goal() and agent.goal_index >= len(agent.gx) - 1:
                    next_velocities[agent.id] = (0.0, 0.0)
                    continue
            
            # ...existing code...
            if agent.agent_type == 'robot':
                if action is not None:
                    next_velocities[agent.id] = action
                elif self.robot_policy:
                    if isinstance(self.robot_policy, SocialForceModel):
                        vx, vy = self.robot_policy.predict(agent, self.all_agents, obstacles=self.obstacles)
                        next_velocities[agent.id] = (vx, vy)
                        new_w = self.robot_policy.get_last_predicted_preferred_velocity()
                        next_preferred_velocities[agent.id] = new_w
                    else:
                        next_velocities[agent.id] = self.robot_policy.predict(agent, self.all_agents, obstacles=self.obstacles)
            
            elif agent.agent_type == 'human':
                if agent.id in self.human_policies:
                    policy = self.human_policies[agent.id]
                    if isinstance(policy, SocialForceModel):
                        vx, vy = policy.predict(agent, self.all_agents, obstacles=self.obstacles, groups=self.groups)
                        next_velocities[agent.id] = (vx, vy)
                        new_w = policy.get_last_predicted_preferred_velocity()
                        next_preferred_velocities[agent.id] = new_w
                    else:
                        next_velocities[agent.id] = policy.predict(agent, self.all_agents, obstacles=self.obstacles)

        for agent in self.all_agents:
            if agent.id in next_velocities:
                vx, vy = next_velocities[agent.id]
                
                agent.set_velocity(vx, vy)
                agent.set_position(agent.px + vx * self.time_step, agent.py + vy * self.time_step)
                if agent.id in next_preferred_velocities:
                    agent.set_preferred_velocity(next_preferred_velocities[agent.id])

        self.current_step += 1
        reward = 0

        # Increment to next goal if the human has completed waiting
        for agent in self.all_agents:
            if isinstance(agent.gx, list) and agent.should_advance_to_next_goal(self.time_step) and agent.goal_index < len(agent.gx) - 1:
                agent.goal_index += 1
                agent.current_wait_time = 0.0
                agent.is_waiting = False
         # check if robot has completed
        for agent in self.all_agents:
            if(agent.agent_type == 'robot'):
                if(isinstance(agent.gx, list) and agent.is_at_goal() and agent.goal_index >= len(agent.gx) - 1):
                    self.done, self.info['status'], reward = True, 'RobotDone', 10
                if(not isinstance(agent.gx, list) and agent.is_at_goal()):
                    self.done, self.info['status'], reward = True, 'RobotDone', 10
        # Check for success (either there is one goal, in which case check that, otherwise check that it's finished all subgoals)
        if all((not isinstance(agent.gx, list) and agent.is_at_goal()) or (isinstance(agent.gx, list) and agent.is_at_goal() and agent.goal_index >= len(agent.gx) - 1) for agent in self.all_agents):
            self.done, self.info['status'], reward = True, 'AllAtGoal', 10
        # Check for timeout
        elif self.current_step >= self.max_steps:
            self.done, self.info['status'] = True, 'Timeout'
        # Check for collisions
        elif self._check_collisions():
            self.done, self.info['status'], reward = True, 'Collision', -20
            
        return None, reward, self.done, self.info
    

    def reset(self):
        """
        Resets the simulation to the initial state.
        Keeps obstacles, groups, and policies intact.
        """
        self.current_step = 0
        self.done = False
        self.info = {}
        for agent in self.all_agents:
            agent.px, agent.py = agent.position_history[0]
            agent.vx, agent.vy = agent.velocity_history[0]
            agent.position_history = [agent.position_history[0]]
            agent.velocity_history = [agent.velocity_history[0]]
            agent.preferred_velocity = np.array([0.0, 0.0])
            agent.goal_index = 0
            agent.current_wait_time = 0.0
            agent.is_waiting = False
        return None



    def _check_collisions(self) -> bool:
        if not self.robot: return False
        for human in self.humans:
            dist = np.sqrt((self.robot.px - human.px)**2 + (self.robot.py - human.py)**2)
            if dist < (self.robot.radius + human.radius):
                return True
        return False

    def visualize_simulation(self, tracker: Optional[MultiHumanTracker] = None, output_file: Optional[str] = None, show_plot: bool = True):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(-12,12)   ### -15,15 for convention scenario
        ax.set_ylim(-12,12)   #### -15,15 for convention scenario
        ax.set_aspect('equal')
        ax.set_title("Crowd Simulator")
        ax.grid(True, linestyle='--', alpha=0.6)
 

        for wall in self.obstacles:
            ax.plot([wall.p1[0], wall.p2[0]], [wall.p1[1], wall.p2[1]], color='black', linewidth=3, zorder=2)

        human_colors = plt.cm.viridis(np.linspace(0, 1, len(self.humans)))
        # human_colors = ['black'] * len(self.humans)

        if self.robot:
            if(isinstance(self.robot.gx, list)):
                ax.plot(self.robot.gx[self.robot.goal_index], self.robot.gy[self.robot.goal_index], 'k*', markersize=15, label="Robot Goal")
            else:
                ax.plot(self.robot.gx, self.robot.gy, 'k*', markersize=15, label="Robot Goal")
        for i, human in enumerate(self.humans):
            # draw all the goals for each human
            for goal_index in range(0, len(human.gx)):
                ax.plot(human.gx[goal_index], human.gy[goal_index], '*', color=human_colors[i], markersize=1)

        robot_artist = Circle((0,0), self.robot.radius, color='red', zorder=10, label="Robot") 
        ax.add_patch(robot_artist)
        human_artists = [Circle((0,0), h.radius, color=c, zorder=8, label=f"Human {h.id}") for i, (h, c) in enumerate(zip(self.humans, human_colors))] # sebastian right here is where colors change
        for artist in human_artists:
            ax.add_patch(artist)

        est_artists, particle_artists = [], []
        if tracker:
            est_artists = [ax.plot([], [], 'x', color='red', markersize=8, zorder=9, label="Estimate")[0] for _ in self.humans]
            particle_artists = [ax.plot([], [], '.', color=c, markersize=1, alpha=0.5, zorder=5)[0] for c in human_colors]

        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        # ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper right')

        def update(frame):
            robot_artist.center = self.robot.position_history[frame]
            for i, human in enumerate(self.humans):
                human_artists[i].center = human.position_history[frame]
                if(frame<len(self.leader_ID) and frame>0 and human_artists[i].get_label() == "Human " + str(self.leader_ID[frame])): # if the human is the leader we're following sebastian
                    human_artists[i].set_color("yellow")
                else:
                    human_artists[i].set_color("black")

                if tracker and frame > 0 and frame <= len(tracker.tracking_history[human.id]['estimated']):
                    est_pos = tracker.tracking_history[human.id]['estimated'][frame-1]
                    est_artists[i].set_data([est_pos[0]], [est_pos[1]])
                    if 'particles' in tracker.tracking_history[human.id] and frame-1 < len(tracker.tracking_history[human.id]['particles']):
                        particles = tracker.tracking_history[human.id]['particles'][frame-1]
                        particle_artists[i].set_data(particles[:, 0], particles[:, 1])

            # print("IN CROWD SIM, leaderID is: ", self.leader_ID)
            print("THIS IS FRAME: ", frame)
            print(len(self.leader_ID))
            if(frame<len(self.leader_ID) and frame>0):   ### issue: we sometimes get self.LEADER_ID shorter than num frames? or an extra frame for some reason? Bandaid solution implemented right now
                time_text.set_text(f'Time: {frame * self.time_step:.2f}s' + '\nLeader: ' + str(self.leader_ID[frame-1]))
            else:
                time_text.set_text(f'Time: {frame * self.time_step:.2f}s' + '\nLeader: None Chosen')
            return [robot_artist] + human_artists + est_artists + particle_artists + [time_text]
            # time_text.set_text(f'Time: {frame * self.time_step:.2f}s' + '\nLeader: ' + str(self.leader_ID[frame-1]))
            # return [robot_artist] + human_artists + est_artists + particle_artists + [time_text]

        num_frames = len(self.robot.position_history)
        max_frames = max(len(agent.position_history) for agent in self.all_agents)
        for agent in self.all_agents:
            while len(agent.position_history) < max_frames:
                agent.position_history.append(agent.position_history[-1])
                agent.velocity_history.append(agent.velocity_history[-1])
        num_frames = max_frames
        anim = animation.FuncAnimation(fig, update, frames=num_frames, interval=150, blit=True)
        
        if output_file:
            print(f"Saving animation to {output_file}...")
            anim.save(output_file, writer='ffmpeg', fps=12)
            print("Save complete.")
        if show_plot:
            return HTML(anim.to_jshtml())
        

    def calculate_path_smoothness(self, window_size=10):  ### this was renamed to Path Roughness in the paper
        print("calculating path smoothness")
        total_rotation = 0
        positions = np.array(self.robot.position_history)
        velocities = np.array(self.robot.velocity_history)
        skips = 0
        for i in range(0, len(positions) - window_size):
            print("iteration: ", i)
            # get current velocity (not instantaneous)
            for j in range(0, window_size // 2):
                curr_velocities = velocities[i + j]
            curr_velocity = curr_velocities / (window_size / 2)

            # if it's not moving
            if(np.linalg.norm(curr_velocity) < 1e-6):
                print("velocity skipped")
                skips += 1
                continue

            # get ideal average velocity vector
            curr_position = positions[i]
            path_ahead_position = positions[i+window_size]
            path_ahead_vector = path_ahead_position - curr_position

            # normalize each vector
            curr_velocity = curr_velocity / np.linalg.norm(curr_velocity)
            path_ahead_vector = path_ahead_vector / np.linalg.norm(path_ahead_vector)

            # get the angle of each vector
            print(f"{curr_velocity=}")
            print(f"{path_ahead_vector=}")
            curr_angle = np.arctan2(curr_velocity[1], curr_velocity[0])
            path_ahead_angle = np.arctan2(path_ahead_vector[1], path_ahead_vector[0])
            print(f"{curr_angle=}")
            print(f"{path_ahead_angle=}")

            # get the difference in angles
            rotation = abs(curr_angle - path_ahead_angle)
            print(f"{rotation=}")

            total_rotation += rotation

        return total_rotation / (len(positions) - window_size - skips)

    def calculate_change_in_human_acceleration(self, bubble_radius = 3.0): ### renamed to Privacy Invasion Metric in the paper
        print("running calculating change in human_acc")
        total_change_in_acceleration = 0

        # iterate through every time step
        num_privacy_invaded = 0
        for frame in range(len(self.all_agents[0].velocity_history) - 3):
            print("frame: ", frame)
            human_velocities = []

            # get a list of human's velocities when they were close to the robot
            robot_position = np.array(self.robot.position_history[frame])
            print(f"{robot_position=}")
            for human in self.all_agents[1:]:
                human_position = np.array(human.position_history[frame])
                print(f"{human_position=}")
                distance_bw_human_robot = np.linalg.norm(robot_position - human_position)
                print(f"{distance_bw_human_robot=}")

                # if close, save its velocities
                if(distance_bw_human_robot < bubble_radius):
                    human_velocities.append(np.array(human.velocity_history[frame: frame+3]))


            num_privacy_invaded += len(human_velocities) # the same human can get recounted
                
            # print(f"{human_velocities=}")
            # # now for each human, see its change in acceleration
            # change_in_acceleration = 0
            # for human_velocity_set in human_velocities:
            #     acc_1 = (human_velocity_set[1] - human_velocity_set[0]) / self.time_step
            #     acc_2 = (human_velocity_set[2] - human_velocity_set[1]) / self.time_step
            #     print(f"{acc_1=}")
            #     delta_a = (acc_2 - acc_1) / self.time_step
            #     change_in_acceleration += np.linalg.norm(delta_a)

            # # sum all the changes of acceleration for all people in the bubble in this frame
            # total_change_in_acceleration += change_in_acceleration

        # return total_change_in_acceleration
        return num_privacy_invaded

    def calculate_metrics(self) -> Dict:
        """
        Calculates simulation metrics after a run is complete.
        Includes both per-agent and global measures, as well as
        robot-focused navigation metrics (time-to-goal, collisions, speed, etc.).
        """
        if not self.done:
            logging.warning("Cannot calculate metrics until the simulation is done.")
            return {}

        metrics = {
            'global': {},
            'per_agent': {agent.id: {} for agent in self.all_agents},
            'robot': {}
        }

        # ----------------
        # Robot-focused metrics
        # ----------------
        robot = self.robot
        metrics['robot']['time_to_goal'] = self.current_step
        metrics['robot']['collision'] = (self.info.get('status') == 'Collision')

        positions = np.array(robot.position_history)
        if len(positions) > 1:
            path_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
            straight_dist = np.linalg.norm(positions[-1] - positions[0])
            metrics['robot']['path_efficiency'] = path_length / (straight_dist + 1e-6)
        else:
            metrics['robot']['path_efficiency'] = float('inf')

        velocities = np.array(robot.velocity_history)
        if len(velocities) > 0:
            speeds = np.linalg.norm(velocities, axis=1)
            metrics['robot']['avg_speed'] = np.mean(speeds)
        else:
            metrics['robot']['avg_speed'] = 0.0

        metrics['robot']['path_smoothness'] = self.calculate_path_smoothness()
        metrics['robot']['change_in_human_acc'] = self.calculate_change_in_human_acceleration()



        # Minimum distance to humans
        min_dist = float('inf')
        if self.humans:
            for step in range(len(self.humans[0].position_history)):
                robot_pos = positions[min(step, len(positions)-1)]
                for h in self.humans:
                    human_pos = np.array(h.position_history[min(step, len(h.position_history)-1)])
                    dist = np.linalg.norm(robot_pos - human_pos)
                    min_dist = min(min_dist, dist)
        metrics['robot']['min_human_dist'] = min_dist
        # ----------------
        # Per-agent metrics
        # ----------------
        for agent in self.all_agents:
            # Average acceleration
            accel_magnitudes = []
            for i in range(1, len(agent.velocity_history)):
                v_curr = np.array(agent.velocity_history[i])
                v_prev = np.array(agent.velocity_history[i-1])
                accel = (v_curr - v_prev) / self.time_step
                accel_magnitudes.append(np.linalg.norm(accel))
            avg_accel = np.mean(accel_magnitudes) if accel_magnitudes else 0.0
            metrics['per_agent'][agent.id]['avg_acceleration'] = avg_accel

            # Path efficiency
            path_length = 0.0
            for i in range(1, len(agent.position_history)):
                p_curr = np.array(agent.position_history[i])
                p_prev = np.array(agent.position_history[i-1])
                path_length += np.linalg.norm(p_curr - p_prev)
            start_pos = np.array(agent.position_history[0])
            end_pos = np.array(agent.position_history[-1])
            straight_line_dist = np.linalg.norm(end_pos - start_pos)
            efficiency = path_length / straight_line_dist if straight_line_dist > 1e-6 else float('inf')
            metrics['per_agent'][agent.id]['path_efficiency'] = efficiency

        # ----------------
        # Global metrics
        # ----------------
        min_dist_overall = float('inf')
        num_steps = len(self.all_agents[0].position_history)
        for t in range(num_steps):
            for i in range(len(self.all_agents)):
                for j in range(i + 1, len(self.all_agents)):
                    pos1 = np.array(self.all_agents[i].position_history[t])
                    pos2 = np.array(self.all_agents[j].position_history[t])
                    dist = np.linalg.norm(pos1 - pos2)
                    if dist < min_dist_overall:
                        min_dist_overall = dist
        metrics['global']['min_inter_agent_distance'] = min_dist_overall

        # cache and return
        self.metrics = metrics
        return metrics



def create_test_scenario(scenario_type: str = 'scenario1') -> Tuple[List[Tuple], List[Tuple], List[Tuple]]:
    """
    Returns human start/goal positions and obstacles for some set scenarios.
    """
    if scenario_type == 'scenario1':
        human_starts = [
            (-5.0, -1.0),   
            (5.0,  -1.0),   
            (-2.0,  -4.0)   
        ]
        human_goals  = [
            (-5.0, 1.0),   
            (5.0, 1.0),  
            (2.0,  -4.0)     
        ]
        obstacles = []

    elif scenario_type == 'scenario2':
        human_starts = [
            (-4.0, 0.0), (4.0, 0.0), (0.0, 4.0), (0.0, -4.0),
            (-2.8, 2.8), (2.8, 2.8), (-2.8, -2.8), (2.8, -2.8)
        ]
        human_goals  = [
            (4.0, 0.0), (-4.0, 0.0), (0.0, -4.0), (0.0, 4.0),
            (2.8, -2.8), (-2.8, -2.8), (2.8, 2.8), (-2.8, 2.8)
        ]
        obstacles = []
    elif scenario_type == 'scenario3':
        human_starts = [(-5, 1), (5, -1), (-5, -1), (5, 1), (0, -5), (0, 5)]
        human_goals = [(5, 1), (-5, -1), (5, -1), (-5, 1), (0, 5), (0, -5)]
        obstacles = []
    elif scenario_type == 'scenario4':
        human_starts = [(-2.0, -2.0), (2.0, 2.0)]
        human_goals  = [(2.0, -2.0), (-2.0, 2.0)]

        obstacles = [
            ((-1.0, 0.0), (1.0, 0.0)), 
        ]
    elif scenario_type == 'scenario5':      
        human_starts = [(-2.5, -1.80), ( 2.5, 0.80), (-2.5, 0.92)]
        human_goals  = [( 2.5, -1.80), (-2.5, 0.80), ( 2.5, 0.92)]

        obstacles = [
            ((2.0,  2.00), (4.0,  2.00)),
        ]
    elif scenario_type == 'junction':
        # Junction environment: plus-sign shaped road with 4 arms
        road_width = 2  # Half-width of each road
        road_length = 6.0  # Length of each road arm from center
        
        # Human starts and goals positioned at the ends of each road arm
        human_starts = [
            #(-road_length, 1.75),  # Left arm
            #(-road_length*0.5, -0.5),
            (-road_length*0.75, -1.5),
            (road_length*0.25, 1.6), # Right arm
            (road_length*0.125, -1.75),
            (road_length, 1),
            (road_length, -0.25),   
            (0.25, -road_length),      # Bottom 
            (-0.25, -road_length*0.8),     
            #(0.75, -road_length*0.5),
            (0.75, road_length),      ## Top
            (-0.75, 0.5*road_length),     
            (1.0, road_length),        
        ]
        human_goals = [
            #(0.25, -road_length),   # Left to bottom
            #(-1.75, road_length),  # Right to top
            (-1, -road_length*0.7),   # Left to bottom
            (-road_length*0.9, 1.75),  # Right to left
            (0.5, -road_length*0.8),   # Right to bottom
            (-0.25, 0.9*road_length),   # Right to top
            (-road_length*0.85, -1.5),   # Bottom to left
            (1.5, road_length),  # Bottom to top
            #(road_length*0.95, -1.75),  # Bottom to right
            (-road_length, -0.5), # Top to left
            (road_length*0.755, 1.6),   # Top to right
            (0.75, -road_length*0.9),  # Top to bottom

        ]
        
        # Create obstacles forming the road boundaries
        obstacles = [
            # Top-left quadrant obstacles
            ((-road_length, road_width), (-road_width, road_width)),  # Top horizontal
            ((-road_width, road_width), (-road_width, road_length)),  # Left vertical
            
            # Top-right quadrant obstacles
            ((road_width, road_width), (road_length, road_width)),    # Top horizontal
            ((road_width, road_width), (road_width, road_length)),    # Right vertical
            
            # Bottom-right quadrant obstacles
            ((road_width, -road_width), (road_length, -road_width)),  # Bottom horizontal
            ((road_width, -road_width), (road_width, -road_length)),  # Right vertical
            
            # Bottom-left quadrant obstacles
            ((-road_length, -road_width), (-road_width, -road_width)),# Bottom horizontal
            ((-road_width, -road_width), (-road_width, -road_length)),# Left vertical
        ]
    else:
        human_starts = [(-5.0, 1.5), (5.0, -1.5), (0.0, 5.0)]
        human_goals  = [(5.0, 1.5), (-5.0, -1.5), (0.0, -5.0)]
        obstacles = []

    return human_starts, human_goals, obstacles


# =============================================================================
# N-Simulation Runner Functions
# =============================================================================

# Global variable to store all run results
all_runs_data = []


def run_single_simulation(setup_func, iteration_num):
    """Run a single simulation and return its data."""
    print(f"\n{'='*60}")
    print(f"Running iteration {iteration_num}")
    print(f"{'='*60}")
    
    # Setup fresh simulation
    sim = setup_func()
    
    # Run simulation
    for step in range(sim.max_steps):
        _, _, done, _ = sim.step()
        if done:
            break
    
    # Calculate metrics
    metrics = sim.calculate_metrics()
    
    # Store run data (only robot metrics for logging)
    run_data = {
        'iteration': iteration_num,
        'sim_params': {
            'time_step': sim.time_step,
            'max_steps': sim.max_steps,
            'total_steps': sim.current_step,
            'num_humans': len(sim.humans),
            'num_obstacles': len(sim.obstacles),
            'status': sim.info.get('status', 'Unknown')
        },
        'metrics': {
            'robot': metrics['robot']  # Only log robot metrics
        }
    }
    
    return run_data


def _aggregate_all_runs():
    """Aggregate robot metrics from all runs in global variable."""
    if not all_runs_data:
        return {}
    
    aggregated = {
        'robot': {},
        'summary': {
            'total_runs': len(all_runs_data),
            'successful_runs': 0,
            'collision_runs': 0,
            'timeout_runs': 0
        }
    }
    
    # Collect all robot metric values
    robot_metrics_dict = {}
    
    # Track time_to_goal values and max_steps for completion rate calculation
    time_to_goal_values = []
    max_steps_values = []
    
    # First pass: collect all metric keys and values (robot only)
    for run in all_runs_data:
        metrics = run['metrics']
        max_steps = run['sim_params']['max_steps']
        max_steps_values.append(max_steps)
        
        # Track time_to_goal separately for filtered averaging
        if 'time_to_goal' in metrics['robot']:
            time_to_goal_values.append((metrics['robot']['time_to_goal'], max_steps))
        
        # Count outcomes
        status = run['sim_params']['status']
        if status == 'AllAtGoal':
            aggregated['summary']['successful_runs'] += 1
        elif status == 'Collision':
            aggregated['summary']['collision_runs'] += 1
        elif status == 'Timeout':
            aggregated['summary']['timeout_runs'] += 1
        
        # Collect only robot metrics
        for key, value in metrics['robot'].items():
            if key not in robot_metrics_dict:
                robot_metrics_dict[key] = []
            if isinstance(value, (int, float)):
                robot_metrics_dict[key].append(value)
    
    # Second pass: calculate statistics (only for robot metrics)
    for key, values in robot_metrics_dict.items():
        if key == 'collision':
            # For collision, calculate rate
            aggregated['robot'][f'{key}_rate'] = sum(values) / len(values)
        elif key == 'time_to_goal':
            # For time_to_goal, only average completed runs (time < max_steps)
            completed_times = [t for t, max_s in time_to_goal_values if t < max_s]
            if completed_times:
                aggregated['robot'][f'{key}_mean'] = np.mean(completed_times)
                aggregated['robot'][f'{key}_std'] = np.std(completed_times)
                aggregated['robot'][f'{key}_min'] = np.min(completed_times)
                aggregated['robot'][f'{key}_max'] = np.max(completed_times)
            else:
                # No completed runs
                aggregated['robot'][f'{key}_mean'] = float('nan')
                aggregated['robot'][f'{key}_std'] = float('nan')
                aggregated['robot'][f'{key}_min'] = float('nan')
                aggregated['robot'][f'{key}_max'] = float('nan')
        else:
            aggregated['robot'][f'{key}_mean'] = np.mean(values)
            aggregated['robot'][f'{key}_std'] = np.std(values)
            aggregated['robot'][f'{key}_min'] = np.min(values)
            aggregated['robot'][f'{key}_max'] = np.max(values)
    
    # Calculate completion rate: proportion of runs where time_to_goal < max_steps
    completed_runs = sum(1 for t, max_s in time_to_goal_values if t < max_s)
    aggregated['robot']['completion_rate'] = completed_runs / len(all_runs_data) if all_runs_data else 0.0
    
    return aggregated


def _convert_to_serializable(obj):
    """Convert NumPy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: _convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def _log_results(aggregated, env_name, method_name, n_iterations, log_dir="logs"):
    """Log aggregated results to file."""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{env_name}_{method_name}_n{n_iterations}_{timestamp}.json"
    filepath = os.path.join(log_dir, filename)
    
    log_data = {
        'timestamp': timestamp,
        'environment': env_name,
        'method': method_name,
        'n_iterations': n_iterations,
        'aggregated_metrics': aggregated,
        'all_runs': all_runs_data  # Store all individual runs too
    }
    
    # Convert NumPy types to Python native types for JSON serialization
    log_data = _convert_to_serializable(log_data)
    
    with open(filepath, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results logged to: {filepath}")
    print(f"{'='*60}\n")
    
    # Print summary
    print("SUMMARY:")
    print(f"  Total runs: {aggregated['summary']['total_runs']}")
    print(f"  Successful: {aggregated['summary']['successful_runs']}")
    print(f"  Collisions: {aggregated['summary']['collision_runs']}")
    print(f"  Timeouts: {aggregated['summary']['timeout_runs']}")
    print("\nROBOT METRICS (averaged):")
    for key, value in aggregated['robot'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")


def run_n_simulations(n_iterations, setup_func, env_name, method_name, log_dir="logs"):
    """
    Run n simulations and aggregate results.
    
    Args:
        n_iterations: Number of simulations to run
        setup_func: Function that sets up and returns a configured simulator
        env_name: Name of environment for logging
        method_name: Name of method for logging
        log_dir: Directory to save logs
    
    Returns:
        aggregated_metrics: Dictionary with averaged metrics across all runs
    """
    global all_runs_data
    all_runs_data = []  # Reset global list
    
    # Run all iterations
    for i in range(1, n_iterations + 1):
        run_data = run_single_simulation(setup_func, i)
        all_runs_data.append(run_data)
        print(f"Iteration {i} complete. Status: {run_data['sim_params']['status']}")
    
    # Calculate aggregated metrics
    aggregated = _aggregate_all_runs()
    
    # Log results
    _log_results(aggregated, env_name, method_name, n_iterations, log_dir)
    
    return aggregated
