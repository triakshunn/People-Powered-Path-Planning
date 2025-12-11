import math
import numpy as np

class LeaderFollowerPlanner:
    def __init__(self,goal):

        self.default=False
        self.robot_range = 10
        self.robot_radius=1.0        
        self.previous_leader = None

        #################
        # Tunable Param # -----------------------------------
        #################
        # agent
        self.human_radius = 0.8  # distance to keep when following, not the radius in planner
        self.human_speed_ideal = 1.4
        self.human_speed_min = 0.6
        self.robot_speed_max = 1.4  # avg human walking speed

        # score function
        self.goal_score_steps = 10
        self.catchup_threshold = 1.5
        self.catchup_speed = 2.0 # sebastian we should changet his to the robot's max speed
        self.position_penalty = -1.75

        # weight
        self.weight_goal = 1.0
        self.weight_velocity = 0.5
        self.weight_position = 1.0
        self.current_leader_bias = 0.05

        # group identification
        self.min_distance_threshold = 1.5
        self.distance_threshold = 2.0
        self.velocity_threshold = 0.5
        
        # visibility
        self.inflate_radius = 0.5
        #-----------------------------------------------------

        self.goal=goal
        # self.obs_list=[]
        self.history_list=[]
        self.list_length=25
        self.state_buffer=[] # list 25, FullState(px, py, vx, vy, radius, gx, gy, v_pref, theta), latest at the end
        self.human_buffer=[] # list of list 25*n, list(ID, px, py, vx, vy), latest at the end (25 time steps for n humans, each element is list with ID...)
        self.time_step=0.11

        self.leader = "None Chosen"

    # SEBASTIAN THIS IS THE MAIN FUNCTION
    # human buffer must be filled with all of the people
    # @prereq
        # self.goal = [global_goal_x, global_goal_y]
    # @params
        # state: (px, py, vx, vy) -> this is the current robot position and velocities
        # human_step: 2D array which is a list of all humans, each human has an [ID, px, py, vx, vy]; n*5; ID order doesn't matter
    def getNewSubGoal(self,state,human_step):

        
        # fill the buffers to a length of 25
        # state [px py vx vy] (not a full state)
        while(len(self.state_buffer)<self.list_length): # appends enough states to fill the state_buffer
            self.state_buffer.append(state) # length 25
        while(len(self.human_buffer)<self.list_length): # appends enough human states to fill the human_buffer
            self.human_buffer.append(human_step) # length 25

        # print(f"{self.state_buffer=}")
        # print(f"{self.human_buffer=}")

        # find the global robot goal
        global_goal_x=self.goal[0]
        global_goal_y=self.goal[1]

        # finding the vector directly to the goal
        robot_goal_vector = (global_goal_x - state[0], global_goal_y - state[1])
        goal_distance = math.sqrt(robot_goal_vector[0]**2 + robot_goal_vector[1]**2)

        #########################
        # Leader Identification #-----------------------------------------------------------------
        #########################
        # ID is not in same order as human_step

        # neighbor_traj is a dictionary where each person (denoted with ID) is given a list of lists (px, py, vx, vy) so your position at every timestep
        neighbor_traj = {}
        # human buffer is timesteps*num_humans*5_parameters
        #making a list of all humans you've seen and adding in any new humans you just saw
        for timestep in self.human_buffer:
            for human in timestep:
                id, px, py, vx, vy = human
                # if the IDs of the people that you found
                if any(agent[0] == id for agent in human_step): # only add visible human (visible = any new human because we aren't using a lidar)
                    if id not in neighbor_traj:
                        neighbor_traj[id] = []
                    neighbor_traj[id].append([px, py, vx, vy])
                    
        ########################
        # Group Identification #
        ########################
        groups = []
        visited = set()

        # human_step = num_humans * 5_parameters
        for i, human_a in enumerate(human_step): # so for every human
            if human_a[0] in visited: # if we've seen this human
                continue
            # if not visited, add the human to your group and add them to visited
            group = [human_a]
            visited.add(human_a[0])

            # iterate over every human again
            for j, human_b in enumerate(human_step):
                # if we've already seen this human or if they're the first one in the group, just move on
                if human_b[0] in visited or human_a[0] == human_b[0]:
                    continue

                # look at the distance between the humans and compare their velocities
                distance = math.sqrt((human_a[1] - human_b[1])**2 + (human_a[2] - human_b[2])**2)
                velocity_similarity = math.sqrt((human_a[3] - human_b[3])**2 + (human_a[4] - human_b[4])**2)

                # if within a certain distance or velocity threshold, add them to the same group
                if (
                    (distance <= self.distance_threshold and velocity_similarity <= self.velocity_threshold)
                ):
                    group.append(human_b)
                    visited.add(human_b[0]) # don't try to put human_b inside another group

            if len(group) > 1:  # group has 2 or more humans
                groups.append(group)
        
        # so now groups is a list of lists, where each internal list has a group of humans (with at least 2 in it)

        ####################
        # Visibility Check #
        ####################
        # for every agent in a subgroup, get their that human's ID
        # so group_list never gets used 
        # group_list = [[agent[0] for agent in group] for group in groups]  
        
        # robot_pose = [state[0], state[1], 0.0] # dummy yaw, but has x and y

        # so for every human, append 0.5 to the list, so human_radius = [0.5] * num_humans
        human_radius = [self.inflate_radius for i in range(len(human_step))] # inflate by robot passing radius

        # human_scores gives the visibility of each human
        # human_scores_list = human_scoring(laser_scan, human_step, robot_pose, human_radius)
        # visible_region_edges = np.array([])

        # create a list of all of the people that I cannot see
        invisible_list = []
        # for k, score in enumerate(human_scores_list):
        #     if score < 0:
        #         invisible_list.append(human_step[k][0])
        # print(f"Invisible: {invisible_list}")

        ###########
        # 1. Goal # heading cosine similarity -1 ~ 1
        ###########
        scores_goal = {}

        # neighbor_traj = dictionary (for each person) of [px, py, vx, vy]s for each timestep
        for ped_id, trajectory in neighbor_traj.items():
            # ped_id = id of the pedestrian
            # trajectory = [px, py, vx, vy] * num_timesteps

            # we're capping the last ten of their timesteps
            num_steps = min(self.goal_score_steps, len(trajectory))

            avg_heading_vector = [0, 0]

            # compare their goal vector with that human's last velocity vector
            goal_vector = (global_goal_x - trajectory[-1][0], global_goal_y - trajectory[-1][1])


            for i in range(-1, -1 - num_steps, -1):  # Iterate backward through the trajectory
                avg_heading_vector[0] += trajectory[i][2]  # vx
                avg_heading_vector[1] += trajectory[i][3]  # vy
            avg_heading_magnitude = math.sqrt(avg_heading_vector[0]**2 + avg_heading_vector[1]**2)
            goal_magnitude = math.sqrt(goal_vector[0]**2 + goal_vector[1]**2)
            
            # redundant if condition because norm is always positive
            # print("In checking goal similarity")
            if avg_heading_magnitude > 0 and goal_magnitude > 0:
                # normalizing the vector of movement for the past (max ten) time steps
                # print("before average heading vector")
                avg_heading_vector = (avg_heading_vector[0] / avg_heading_magnitude, avg_heading_vector[1] / avg_heading_magnitude)
                # print("Normalized average heading vector:", avg_heading_vector)
                # normalizing the goal vector
                # print("before normalize")
                goal_vector = (goal_vector[0] / goal_magnitude, goal_vector[1] / goal_magnitude)
                # print("normalized average heading: ", goal_vector)
            
                # dot product of the heading and the goal
                dot_product = avg_heading_vector[0] * goal_vector[0] + avg_heading_vector[1] * goal_vector[1]

                # Only consider scores within the +/- 45-degree range
                # if going in the right direction, add how MUCH in the right direction
                if dot_product >= 0.5:  # cos(45 degrees) ≈ 0.707 # sebastian, you could tune this parameter
                    scores_goal[ped_id] = dot_product
                else:
                    # if going in such a bad direction, penalize them
                    scores_goal[ped_id] = -10
            else:
                # this never happens :(
                scores_goal[ped_id] = -10
            
        ###############
        # 2. Velocity # 0 ~ 1, 0 if very high speed, negative if too slow, -10 if stationary
        ###############
        scores_velocity = {}
        human_speeds = {}
        ideal_speed = self.human_speed_ideal
        stationary_human = [] # keep a list of all humans that aren't moving (never gets used)

        # ped_id is each pedestrian id
        # trajectory is the list of positions of that ped [px, py, vx, vy]
        for ped_id, trajectory in neighbor_traj.items():
            total_distance = 0

            # capping trajectory at 10
            steps = min(10, len(trajectory)) # avg over the last 10 steps
            for i in range(-1, -1 - steps, -1):
                # iterating through your last (up to) 10 velocities
                total_distance += math.sqrt(trajectory[i][2]**2 + trajectory[i][3]**2) # this only true when duration of steps is equal to 1 second
            # print("in velocity, before average speed")
            avg_speed = total_distance / steps
            # print("after average speed", avg_speed)
            human_speeds[ped_id] = math.sqrt(trajectory[-1][2]**2 + trajectory[-1][3]**2) # current speed

            # if the human's super slow, hate them
            if avg_speed < self.human_speed_min:
                # print("when human is slow before scores_veolocity")
                scores_velocity[ped_id] = (avg_speed - ideal_speed) / ideal_speed
                # print("after scores velocity:", scores_velocity[ped_id])
                if avg_speed < 0.1: # do not follow stationary human
                    scores_velocity[ped_id] = -10 # really don't choose them
                    stationary_human.append(ped_id)
            else: # if human is going fast enough, give them a score from 0 to 1 (where 1 is exactly the ideal speed)

                scores_velocity[ped_id] = max(0, 1 - (abs(avg_speed - ideal_speed) / ideal_speed))
                # print("human is fast enough, score velocity:", scores_velocity[ped_id])
        ###############
        # 3. Position # 0 ~ 1, -1.75 if behind robot
        ###############
        scores_position = {}
        hr_distance = {}
        hr_vector = {}

        # ped_id is the id of the pedestrian
        # trajectory is the list of [px, py, vx, vy] * num_timesteps 
        for ped_id, trajectory in neighbor_traj.items():

            # last human position
            human_pos = trajectory[-1][:2]  # (px, py)

            # difference vector from current robot position to current human position
            human_vector = (human_pos[0] - state[0], human_pos[1] - state[1])
            distance = math.sqrt(human_vector[0]**2 + human_vector[1]**2)

            # hr means human-robot
            hr_distance[ped_id] = distance
            hr_vector[ped_id] = human_vector

            # normalize the distance vector from the human and the robot
            # print("in position, before getting the human vector")
            # print(f"{human_vector=}")
            # print(f"{distance=}")
            human_vector = (human_vector[0] / distance, human_vector[1] / distance)
            # print("in position, after gettiing human_vector:", human_vector)

            # this is the ideal robot direction normalized
            robot_heading = robot_goal_vector
            r_heading_mag = math.sqrt(robot_heading[0]**2 + robot_heading[1]**2)
            # print("in position, before getting robot heading")
            robot_heading = (robot_heading[0] / r_heading_mag, robot_heading[1] / r_heading_mag)
            # print("got robot_heading: ", robot_heading)
            
            # compare the vectors of the vector the human is away from the robot and the goal of the robot
            dot_product = human_vector[0] * robot_heading[0] + human_vector[1] * robot_heading[1]
            if dot_product >= 0.5:  # within +/-60 degree # this is wrong, it's actually 15 degrees (sebastian, this is a parameter to change)
                # print("person was in a good bounds for position, before getting scores position:")
                scores_position[ped_id] = (dot_product + max(0, 1 - (distance / self.robot_range))) / 2
                # print("person was in a good bounds for position, after getting scores position:", scores_position[ped_id])
            else:  # behind
                scores_position[ped_id] = self.position_penalty

        #########
        # Total #
        #########
        total_scores = {}
        # for each pedestrian
        for ped_id in neighbor_traj.keys():
            score_goal = scores_goal.get(ped_id, -1)
            score_velocity = scores_velocity.get(ped_id, -1)
            score_position = scores_position.get(ped_id, -1)
            # filter out invisible humans
            if ped_id not in invisible_list: # invisible list is [], so it's redundant
                total_scores[ped_id] = (self.weight_goal * score_goal + 
                                        self.weight_velocity * score_velocity + 
                                        self.weight_position * score_position)
            
            # print(f"ID: {ped_id:>1} | goal: {score_goal:>5.1f} | vel: {score_velocity:>5.1f} | pos: {score_position:>5.1f}")

        # Favour current leader to avoid fluctuation
        # prefer not switching leaders
        for human in total_scores:
            if human == self.previous_leader:
                total_scores[human] += self.current_leader_bias # super tiny bias
                continue
                
        # check if we should follow leaders
        if (
            total_scores # exist neighbors
            and any(score > 0 for score in total_scores.values()) # exist good leader
            and goal_distance > 1.0 # still far from goal; if really near just use SFM
        ):
            leader_ID = max(total_scores, key=total_scores.get) # choose the best leader
            # print(f"Leader ID: {leader_ID}     distance: {goal_distance:.2f}")

            #----If leader belongs to a group-----------------------------------------
            # if leader is in a group, just choose the person in the group closest to you
            def get_closest_human_in_group(group, robot_x, robot_y):
                closest_human = None
                min_distance = float('inf')
                for human in group:
                    human_px, human_py = human[1], human[2]
                    distance = math.sqrt((robot_x - human_px)**2 + (robot_y - human_py)**2)
                    if distance < min_distance:
                        closest_human = human
                        min_distance = distance
                return closest_human

            # find the group that the leading human is in
            leader_group = None
            for group in groups:
                if any(human[0] == leader_ID for human in group):
                    leader_group = group
                    break
            if leader_group: # if we found a leader group, then use get_closest_human to find a person in that group that's close by
                closest_human = get_closest_human_in_group(leader_group, state[0], state[1])
                if ( # if it's a valid new leader (not invisible, and has positive leader score), then switch leader, otherwise, don't
                    closest_human[0] != leader_ID
                    and closest_human[0] not in invisible_list # is visible
                    and total_scores[closest_human[0]] > 0 # is ok leader
                ):
                    leader_ID = closest_human[0]
                    # print(f"Switch to closest human in group: {leader_ID}")
            #-------------------------------------------------------------------------

            self.following_id_vis = leader_ID
            self.previous_leader = leader_ID

            print("WE FOUND A LEADER: ", leader_ID)
            self.leader = leader_ID
            print(f"{self.leader=}")
            
            ###############
            # Set Subgoal #
            ###############
            '''
            - basis is human-robot vector
            - +/- 45 degree range
            - choose the one with highest min distance from human (the clearest path)
            '''

            # get the vector between the robot and the leader (this is displacement, but they treat it like velocity)
            vx = hr_vector[leader_ID][0]
            vy = hr_vector[leader_ID][1]
            
            # this is the last known position of the leader
            base_gx = neighbor_traj[leader_ID][-1][0]
            base_gy = neighbor_traj[leader_ID][-1][1]

            # get a list of valid positions surrounding the leader
            def get_candidate_positions(vx, vy, angle_range=90): # 45 # sebastian this is a tunable parameter
                positions = []
                magnitude = (vx**2 + vy**2)**0.5

                # if the robot and the human aren't on top of each other
                if magnitude != 0:
                    # for each angle in the angle range (every 2 degrees), get gx and gy that are an arc around the leader
                    for angle in range(-angle_range, angle_range + 1, 2):
                        angle_rad = math.radians(angle)
                        rotated_vx = vx * math.cos(angle_rad) - vy * math.sin(angle_rad)
                        rotated_vy = vx * math.sin(angle_rad) + vy * math.cos(angle_rad)
                        # print("getting candidate position")
                        pos_gx = base_gx - self.human_radius * rotated_vx / magnitude
                        pos_gy = base_gy - self.human_radius * rotated_vy / magnitude
                        # print("after getting candidate position: ", pos_gx, pos_gy)
                        positions.append((pos_gx, pos_gy))
                else:
                    positions.append((base_gx, base_gy))
                # positions is a list of tuples (x, y)
                return positions

            # this is returning the distance that point is from the nearest human
            # we'll want to choose a point far from other people
            def evaluate_position(goal_x, goal_y):
                min_distance = float('inf')
                for human in human_step:
                    if human[0] == leader_ID: # doesn't matter if its close to the leader because of course it's close ot the leader
                        continue
                    human_px, human_py = human[1], human[2]
                    distance = math.sqrt((human_px - goal_x)**2 + (human_py - goal_y)**2)
                    min_distance = min(min_distance, distance)
                return min_distance    

            # gets candidate positions relative to the orbot
            candidate_positions = get_candidate_positions(vx, vy) # sebastian, remember that you need to give everything relative to robot frame... sike it's not

            # take the one that maximizes distance from neighboring people
            best_position = max(candidate_positions, key=lambda pos: evaluate_position(pos[0], pos[1]))
            new_gx, new_gy = best_position
            
            # if sample pos is close to neighbor, set further
            min_safe_distance = 2.0
            min_distance = evaluate_position(new_gx, new_gy)
            if min_distance < min_safe_distance: # if the best subgoal is still unsafe, make it safer
                print('##### Push further ######')
                # base_gx is last known position of the leader
                # create a new vector that's closer to the robot than the leader
                direction_vx = new_gx - base_gx
                direction_vy = new_gy - base_gy
                direction_mag = (direction_vx**2 + direction_vy**2)**0.5
                if direction_mag > 0:
                    print("before getting scale")
                    scale = (min_safe_distance / min_distance) * 1.0
                    print("after gettting scale: ", scale)
                    new_gx = base_gx + scale * direction_vx
                    new_gy = base_gy + scale * direction_vy

            #################
            # Set New Speed #
            #################
            '''
            - move like leader when close
            - move fast to catch up leader when far
            '''
            command_v_pref=0
            if hr_distance[leader_ID] > self.catchup_threshold:
                command_v_pref = self.catchup_speed
                print("---catching up---")
            else:
                command_v_pref = human_speeds[leader_ID]

        else: # if no suitable leader, switch to default planner
            print(f"Back to default planner   distance: {goal_distance:.2f}")
            self.following_id_vis=-1
            new_gx = global_goal_x
            new_gy = global_goal_y
            command_v_pref = self.robot_speed_max

        # return new_gx, new_gy, command_v_pref

        # # sebastian we wrote this
        # displacement_vector = np.array([new_gx, new_gy])
        # displacement_vector = displacement_vector / np.linalg.norm(displacement_vector)
        # ideal_velocity = displacement_vector * command_v_pref
        # return (ideal_velocity[0], ideal_velocity[1])

        return new_gx, new_gy, command_v_pref
    
    def clear_buffer(self):
        self.state_buffer=[] # list 25, FullState(px, py, vx, vy, radius, gx, gy, v_pref, theta), latest at the end
        self.human_buffer=[] # list of list 25*n, list(ID, px, py, vx, vy), latest at the end
