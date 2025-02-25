import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import pygame
import os
import matplotlib.pyplot as plt
import csv
import sys
from torch.utils.tensorboard import SummaryWriter

# ===============================
# Global Configurations and Constants
# ===============================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Card definitions
SUITS = ['C', 'D', 'H', 'S']
RANKS = ["3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2"]
JOKER = "Joker"

# Deck: 52 cards + 1 Joker = 53 cards.
DECK_SIZE = 53

# Discrete Action Space Mapping:
#   0: "pass"
#   1-53: single plays (card index = action-1)
#   54-66: pair plays for each rank (r = action-54)
#   67-79: triplet plays (r = action-67)
#   80-92: four-of-a-kind plays (r = action-80)
#   93-212: sequence plays (length 3,4,5; enumerated by suit, length, and starting rank)
ACTION_SPACE_SIZE = 213

# Precompute the sequence mapping list.
# Each element is a tuple: (length, suit_index, start_rank_index)
sequence_actions = []
for length in [3, 4, 5]:
    for suit_index in range(len(SUITS)):
        for start in range(0, 13 - length + 1):
            sequence_actions.append((length, suit_index, start))
assert len(sequence_actions) == 120

# ===============================
# Helper Functions for Card Handling
# ===============================

def get_rank_value(rank, revolution=False):
    """Returns the numerical value for a rank. Under revolution, order is reversed."""
    if rank == "Joker":
        return 14
    if not revolution:
        return RANKS.index(rank) + 1
    else:
        return len(RANKS) - RANKS.index(rank)

def index_to_card(index):
    """Converts a card index to its string representation. Index 52 is Joker."""
    if index == 52:
        return "Joker"
    else:
        rank = RANKS[index // 4]
        suit = SUITS[index % 4]
        return f"{rank}{suit}"

def card_str_to_index(card_str):
    """Converts a card string (e.g., '3C' or 'Joker') to its index."""
    s = card_str.strip().upper()
    if s == "JOKER":
        return 52
    if len(s) < 2:
        return None
    rank = s[:-1]
    suit = s[-1]
    if rank not in RANKS or suit not in SUITS:
        return None
    return RANKS.index(rank) * 4 + SUITS.index(suit)

# ===============================
# Environment Class: DaifugoEnv
# ===============================

class DaifugoEnv:
    """
    Daifugō/Dai Hinmin environment implementing:
      - Moves: single, pair, triplet, four-of-a-kind (toggles revolution), and sequences.
      - Joker substitution for pairs/triplets/fours.
      - A fixed discrete action space (213 actions).
      - Card exchanges between rounds.
    """
    def __init__(self, num_players=5, human_player=False, visualize=False, training=True, use_revolution=True):
        self.num_players = num_players
        self.human_player = human_player  # If True, player 0 is controlled by human.
        self.visualize = visualize
        self.training = training
        self.use_revolution = use_revolution
        self.revolution = False
        
        self.ACTION_SPACE_SIZE = ACTION_SPACE_SIZE
        self.sequence_actions = sequence_actions
        
        self.deck = list(range(DECK_SIZE))
        self.hands = []
        self.last_move = None  # Dict: {"type": move_type, "cards": [card indices]}
        self.last_play_type = None
        self.pass_count = 0
        self.winner_order = []
        self.move_log = []
        self.round = 1
        self.first_move = True
        self.current_player = 0
        
        if self.visualize:
            pygame.init()
            self.screen = pygame.display.set_mode((1200, 800))
            pygame.display.set_caption("Daifugō Card Game")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 24)
            self.card_images = self.load_card_images()
        
        self.reset(first_hand=True)
    
    def reset(self, first_hand=False):
        """Resets game state: shuffles, deals, resets flags, and performs exchanges if needed."""
        self.deck = list(range(DECK_SIZE))
        random.shuffle(self.deck)
        self.hands = []
        hand_size = len(self.deck) // self.num_players
        for i in range(self.num_players):
            hand = sorted(self.deck[i * hand_size:(i + 1) * hand_size])
            self.hands.append(hand)
        remainder = len(self.deck) % self.num_players
        for i in range(remainder):
            self.hands[i].append(self.deck[-(i + 1)])
            self.hands[i].sort()
        
        self.last_move = None
        self.last_play_type = None
        self.pass_count = 0
        self.move_log = []
        self.first_move = True
        self.revolution = False
        
        self.current_player = random.randint(0, self.num_players - 1)
        if not first_hand and self.winner_order:
            self.exchange_cards()
            self.winner_order = []
        
        if self.visualize:
            self.render()
        return self.get_obs()
    
    def exchange_cards(self):
        """
        Implements card exchange between rounds:
          - Last-place (Dai Hinmin) gives two highest cards to first-place (Dai Fugō).
          - Second-last (Hinmin) gives one highest card to second-place (Fugō).
          (Assumes at least 5 players.)
        """
        if self.num_players < 5 or len(self.winner_order) < self.num_players:
            return
        dai_fugou = self.winner_order[0]
        fugou = self.winner_order[1]
        hinmin = self.winner_order[-2]
        dai_hinmin = self.winner_order[-1]
        if len(self.hands[dai_hinmin]) >= 2:
            cards_to_exchange = sorted(self.hands[dai_hinmin],
                                       key=lambda c: self.card_rank_value(c),
                                       reverse=True)[:2]
            for card in cards_to_exchange:
                self.hands[dai_hinmin].remove(card)
                self.hands[dai_fugou].append(card)
            self.hands[dai_fugou].sort()
            self.hands[dai_hinmin].sort()
        if len(self.hands[hinmin]) >= 1:
            card_to_exchange = sorted(self.hands[hinmin],
                                      key=lambda c: self.card_rank_value(c),
                                      reverse=True)[0]
            self.hands[hinmin].remove(card_to_exchange)
            self.hands[fugou].append(card_to_exchange)
            self.hands[fugou].sort()
    
    def card_rank_value(self, card_index):
        """Returns the rank value of a card (adjusted for revolution)."""
        if card_index == 52:
            return get_rank_value("Joker", self.revolution)
        else:
            rank = RANKS[card_index // 4]
            return get_rank_value(rank, self.revolution)
    
    def get_obs(self):
        """
        Returns the observation vector for the current player.
        It is a binary vector for the player's hand (length 53)
        concatenated with a 10-length encoding for the last move.
        """
        hand_obs = np.zeros(DECK_SIZE, dtype=np.float32)
        for card in self.hands[self.current_player]:
            hand_obs[card] = 1.0
        last_move_obs = np.zeros(10, dtype=np.float32)
        if self.last_move:
            type_mapping = {"single": 0, "pair": 1, "triplet": 2, "four": 3, "sequence": 4, "pass": 5}
            last_move_obs[0] = type_mapping.get(self.last_move["type"], -1)
            last_move_obs[1] = len(self.last_move["cards"])
        return np.concatenate([hand_obs, last_move_obs])
    
    def load_card_images(self):
        """Loads card images from the 'cards' directory if available."""
        card_images = {}
        card_dir = "cards"
        for rank in RANKS:
            for suit in SUITS:
                card_code = rank + suit
                image_path = os.path.join(card_dir, f"{card_code}.png")
                index = RANKS.index(rank) * 4 + SUITS.index(suit)
                if os.path.exists(image_path):
                    try:
                        image = pygame.image.load(image_path).convert_alpha()
                        image = pygame.transform.scale(image, (80, 120))
                        card_images[index] = image
                    except pygame.error as e:
                        print(f"Error loading {image_path}: {e}")
                        card_images[index] = None
                else:
                    card_images[index] = None
        joker_path = os.path.join(card_dir, "Joker.png")
        if os.path.exists(joker_path):
            try:
                image = pygame.image.load(joker_path).convert_alpha()
                image = pygame.transform.scale(image, (80, 120))
                card_images[52] = image
            except pygame.error as e:
                print(f"Error loading {joker_path}: {e}")
                card_images[52] = None
        else:
            card_images[52] = None
        return card_images

    def render(self):
        """Renders the game state using Pygame (all player hands face-up)."""
        if not self.visualize:
            return
        self.screen.fill((0, 128, 0))
        for idx, hand in enumerate(self.hands):
            pos = (50, 50 + idx * 150)
            label = f"Player {idx}"
            label_surface = self.font.render(label, True, (255, 255, 255))
            self.screen.blit(label_surface, (pos[0], pos[1] - 30))
            for i, card in enumerate(sorted(hand)):
                if self.card_images.get(card):
                    self.screen.blit(self.card_images[card], (pos[0] + i * 90, pos[1]))
                else:
                    card_text = self.font.render(index_to_card(card), True, (0, 0, 0))
                    self.screen.blit(card_text, (pos[0] + i * 90, pos[1]))
        lm_text = "Last Move: "
        if self.last_move:
            lm_text += f"{self.last_move['type']} " + " ".join(index_to_card(c) for c in self.last_move["cards"])
        else:
            lm_text += "None"
        lm_surface = self.font.render(lm_text, True, (255, 255, 0))
        self.screen.blit(lm_surface, (50, 700))
        rev_text = "Revolution ON" if self.revolution else "Normal Order"
        rev_surface = self.font.render(rev_text, True, (255, 0, 0))
        self.screen.blit(rev_surface, (900, 50))
        pygame.display.flip()
        self.clock.tick(30)
    
    # --------------------
    # Discrete Action Mapping Methods
    # --------------------
    def decode_action(self, action_index, player_index):
        """
        Decodes a discrete action index (0..212) into a move dictionary.
        Returns None if the move cannot be performed.
        """
        hand = self.hands[player_index]
        if action_index == 0:
            return {"type": "pass", "cards": []}
        if 1 <= action_index <= 53:
            card = action_index - 1
            if card in hand:
                return {"type": "single", "cards": [card]}
            return None
        if 54 <= action_index <= 66:
            r = action_index - 54
            desired_rank = RANKS[r]
            available = [c for c in hand if c != 52 and RANKS[c // 4] == desired_rank]
            if len(available) >= 2:
                return {"type": "pair", "cards": sorted(available)[:2]}
            elif len(available) == 1 and 52 in hand:
                return {"type": "pair", "cards": sorted([available[0], 52])}
            return None
        if 67 <= action_index <= 79:
            r = action_index - 67
            desired_rank = RANKS[r]
            available = [c for c in hand if c != 52 and RANKS[c // 4] == desired_rank]
            if len(available) >= 3:
                return {"type": "triplet", "cards": sorted(available)[:3]}
            elif len(available) == 2 and 52 in hand:
                return {"type": "triplet", "cards": sorted(available + [52])}
            return None
        if 80 <= action_index <= 92:
            r = action_index - 80
            desired_rank = RANKS[r]
            available = [c for c in hand if c != 52 and RANKS[c // 4] == desired_rank]
            if len(available) >= 4:
                return {"type": "four", "cards": sorted(available)[:4]}
            elif len(available) == 3 and 52 in hand:
                return {"type": "four", "cards": sorted(available + [52])}
            return None
        if 93 <= action_index < 93 + len(self.sequence_actions):
            seq_idx = action_index - 93
            length, suit_index, start = self.sequence_actions[seq_idx]
            seq_cards = []
            for offset in range(length):
                desired_rank_index = start + offset
                found = None
                for card in hand:
                    if card == 52:
                        continue
                    if (card // 4) == desired_rank_index and (card % 4) == suit_index:
                        found = card
                        break
                if found is None:
                    return None
                seq_cards.append(found)
            return {"type": "sequence", "cards": seq_cards}
        return None

    def get_valid_action_indices(self, player_index):
        """
        Returns a list of valid discrete action indices for the given player.
        """
        valid_indices = []
        for a in range(self.ACTION_SPACE_SIZE):
            move = self.decode_action(a, player_index)
            if move is None:
                continue
            if self.first_move or self.last_move is None:
                valid_indices.append(a)
            else:
                if self.is_valid_move(move):
                    valid_indices.append(a)
        return valid_indices

    def is_valid_move(self, move):
        """
        Checks if the move is valid.
        For non-pass moves (when not the first move), the move type (and sequence length)
        must match the last move and its evaluated strength must be higher.
        """
        if move["type"] == "pass":
            return not self.first_move
        if self.first_move or self.last_move is None:
            return True
        if move["type"] != self.last_move["type"]:
            return False
        if move["type"] == "sequence":
            if len(move["cards"]) != len(self.last_move["cards"]):
                return False
            suit = SUITS[move["cards"][0] % 4]
            for card in move["cards"]:
                if card == 52 or SUITS[card % 4] != suit:
                    return False
        else:
            if len(move["cards"]) != len(self.last_move["cards"]):
                return False
        return self.evaluate_move(move) > self.evaluate_move(self.last_move)
    
    def evaluate_move(self, move):
        """
        Returns a numerical strength for the move.
        For singles, pairs, triplets, and fours, strength is based on rank.
        For sequences, it is based on the highest card.
        """
        if move["type"] == "single":
            return self.card_rank_value(move["cards"][0])
        elif move["type"] in ["pair", "triplet", "four"]:
            ranks = [RANKS[c // 4] for c in move["cards"] if c != 52]
            if ranks:
                return get_rank_value(ranks[0], self.revolution)
            return 0
        elif move["type"] == "sequence":
            highest = max(move["cards"], key=lambda c: self.card_rank_value(c))
            return self.card_rank_value(highest)
        return 0

    def step(self, action_index):
        """
        Executes the move corresponding to the discrete action index.
        Returns (obs, reward, done, info). Invalid moves receive a penalty.
        """
        move = self.decode_action(action_index, self.current_player)
        if move is None or (not self.first_move and self.last_move is not None and not self.is_valid_move(move)):
            reward = -1
            obs = self.get_obs()
            return obs, reward, False, {}
        if move["type"] == "pass":
            self.pass_count += 1
            self.move_log.append(f"Player {self.current_player} passes.")
        else:
            for card in move["cards"]:
                if card in self.hands[self.current_player]:
                    self.hands[self.current_player].remove(card)
            self.last_move = move
            self.last_play_type = move["type"]
            self.pass_count = 0
            self.move_log.append(f"Player {self.current_player} plays {move['type']} with: " +
                                 " ".join(index_to_card(c) for c in move["cards"]))
            if move["type"] == "four" and self.use_revolution:
                self.revolution = not self.revolution
                self.move_log.append(f"Revolution triggered! Now: {'Revolution' if self.revolution else 'Normal'}")
        if len(self.hands[self.current_player]) == 0 and self.current_player not in self.winner_order:
            self.winner_order.append(self.current_player)
            self.move_log.append(f"Player {self.current_player} finished at rank {len(self.winner_order)}.")
        remaining = self.num_players - len(self.winner_order)
        if self.pass_count >= (remaining - 1):
            self.last_move = None
            self.last_play_type = None
            self.pass_count = 0
            self.move_log.append("Table cleared.")
        next_player = (self.current_player + 1) % self.num_players
        while next_player in self.winner_order:
            next_player = (next_player + 1) % self.num_players
        self.current_player = next_player
        self.first_move = False
        if self.visualize:
            self.render()
        done = (len(self.winner_order) == self.num_players - 1)
        info = {}
        if done:
            info["winner_order"] = self.winner_order + [self.current_player]
        reward = 1
        obs = self.get_obs()
        return obs, reward, done, info

# ===============================
# Neural Network and Replay Buffer
# ===============================

class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *args):
        self.buffer.append(Transition(*args))
    
    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# ===============================
# Human Input Functions
# ===============================

def parse_human_input(user_input, env, player_index):
    """
    Parses human input. Valid commands:
      - "pass"
      - "single <card>"         (e.g., "single 3C" or "single Joker")
      - "pair <rank>"           (e.g., "pair 5")
      - "triplet <rank>"        (e.g., "triplet A")
      - "four <rank>"           (e.g., "four 7")
      - "sequence <suit> <length> <start>"   (e.g., "sequence H 3 3")
    Returns the corresponding discrete action index or None.
    """
    tokens = user_input.strip().split()
    if not tokens:
        return None
    cmd = tokens[0].lower()
    if cmd in ["pass", "p"]:
        return 0
    elif cmd in ["single", "s"]:
        if len(tokens) != 2:
            return None
        card = card_str_to_index(tokens[1])
        if card is None:
            return None
        return card + 1
    elif cmd in ["pair", "d", "double"]:
        if len(tokens) != 2:
            return None
        rank = tokens[1].upper()
        if rank not in RANKS:
            return None
        r = RANKS.index(rank)
        return 54 + r
    elif cmd in ["triplet", "t"]:
        if len(tokens) != 2:
            return None
        rank = tokens[1].upper()
        if rank not in RANKS:
            return None
        r = RANKS.index(rank)
        return 67 + r
    elif cmd in ["four"]:
        if len(tokens) != 2:
            return None
        rank = tokens[1].upper()
        if rank not in RANKS:
            return None
        r = RANKS.index(rank)
        return 80 + r
    elif cmd in ["sequence", "seq"]:
        if len(tokens) != 4:
            return None
        suit = tokens[1].upper()
        if suit not in SUITS:
            return None
        try:
            length = int(tokens[2])
        except:
            return None
        start_rank = tokens[3].upper()
        if start_rank not in RANKS:
            return None
        suit_index = SUITS.index(suit)
        start = RANKS.index(start_rank)
        for idx, tup in enumerate(env.sequence_actions):
            if tup == (length, suit_index, start):
                return 93 + idx
        return None
    return None

def human_select_action(env, player_index):
    """
    Prompts the human player for input and returns a valid discrete action index.
    Displays the player's hand and instructions.
    """
    while True:
        print("\nYour hand:")
        hand = env.hands[player_index]
        print(" ".join(index_to_card(c) for c in sorted(hand)))
        print("Valid commands:")
        print("  pass")
        print("  single <card>         (e.g., 'single 3C' or 'single Joker')")
        print("  pair <rank>           (e.g., 'pair 5')")
        print("  triplet <rank>        (e.g., 'triplet A')")
        print("  four <rank>           (e.g., 'four 7')")
        print("  sequence <suit> <length> <start>   (e.g., 'sequence H 3 3')")
        user_input = input("Enter your move: ")
        action = parse_human_input(user_input, env, player_index)
        if action is None:
            print("Invalid input. Please try again.")
            continue
        valid_actions = env.get_valid_action_indices(player_index)
        if action not in valid_actions:
            print("That move is not valid at this time. Valid moves are:", valid_actions)
            continue
        return action

# ===============================
# Action Selection and Training Functions
# ===============================

def select_action(model: nn.Module, state: np.ndarray, epsilon: float, valid_actions: list, device: torch.device):
    """
    Epsilon-greedy action selection over valid actions.
    """
    if random.random() < epsilon or not valid_actions:
        return random.choice(valid_actions) if valid_actions else 0
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = model(state_tensor).cpu().numpy()[0]
    mask = np.full(q_values.shape, -np.inf)
    for a in valid_actions:
        mask[a] = q_values[a]
    return int(np.argmax(mask))

def train_dqn(model: nn.Module, target_model: nn.Module, optimizer: optim.Optimizer,
              criterion: nn.Module, replay_buffer: ReplayBuffer, batch_size: int,
              gamma: float, device: torch.device) -> float:
    """
    Trains the DQN model using a batch of transitions.
    """
    if len(replay_buffer) < batch_size:
        return 0.0
    transitions = replay_buffer.sample(batch_size)
    batch = Transition(*zip(*transitions))
    state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32).to(device)
    action_batch = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1).to(device)
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1).to(device)
    next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(device)
    done_batch = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1).to(device)
    current_q = model(state_batch).gather(1, action_batch)
    with torch.no_grad():
        next_q = model(next_state_batch)
        next_actions = torch.argmax(next_q, dim=1, keepdim=True)
        next_q_target = target_model(next_state_batch).gather(1, next_actions)
        target_q = reward_batch + gamma * next_q_target * (1 - done_batch)
    loss = criterion(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# ===============================
# Main Training and Evaluation Loop
# ===============================

def main():
    # Mode Selection:
    print("Select mode:")
    print("1: Train AI vs AI")
    print("2: Train AI vs Human")
    print("3: Test AI vs Human")
    mode_input = input("Enter choice (1/2/3): ").strip()
    if mode_input == "1":
        TRAINING_MODE = True
        HUMAN_PLAYER = False
    elif mode_input == "2":
        TRAINING_MODE = True
        HUMAN_PLAYER = True
    elif mode_input == "3":
        TRAINING_MODE = False
        HUMAN_PLAYER = True
    else:
        print("Invalid input. Defaulting to Train AI vs AI.")
        TRAINING_MODE = True
        HUMAN_PLAYER = False

    VISUALIZE = True  # Show all cards face-up.
    NUM_PLAYERS = 5

    # Hyperparameters:
    NUM_EPISODES = 1000
    MAX_STEPS_PER_EPISODE = 1000
    BATCH_SIZE = 64
    GAMMA = 0.99
    LEARNING_RATE = 1e-3
    REPLAY_BUFFER_CAPACITY = 10000
    TARGET_UPDATE_FREQ = 500
    EPSILON_START = 1.0
    EPSILON_DECAY = 0.995
    EPSILON_END = 0.05
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = DaifugoEnv(num_players=NUM_PLAYERS, human_player=HUMAN_PLAYER, visualize=VISUALIZE,
                     training=TRAINING_MODE, use_revolution=True)
    input_dim = len(env.get_obs())
    output_dim = env.ACTION_SPACE_SIZE  # 213

    # Create agent models.
    agents = [DQN(input_dim, output_dim).to(DEVICE) for _ in range(NUM_PLAYERS)]
    target_agents = [DQN(input_dim, output_dim).to(DEVICE) for _ in range(NUM_PLAYERS)]
    for i in range(NUM_PLAYERS):
        target_agents[i].load_state_dict(agents[i].state_dict())
        target_agents[i].eval()

    # Ask the user whether to load saved models (if available)
    load_choice = input("Saved model files may exist. Do you want to load them? (y/n): ").strip().lower()
    if load_choice == "y":
        for i in range(NUM_PLAYERS):
            model_filename = f"dqn_player_{i}.pth"
            if os.path.exists(model_filename):
                checkpoint = torch.load(model_filename, map_location=DEVICE)
                agents[i].load_state_dict(checkpoint["state_dict"])
                target_agents[i].load_state_dict(checkpoint["state_dict"])
                loaded_ep = checkpoint.get("episode", "Unknown")
                print(f"Loaded model for player {i} from {model_filename} (saved at episode {loaded_ep}).")
            else:
                print(f"No saved model found for player {i}. Training from scratch.")

    optimizers = [optim.Adam(agents[i].parameters(), lr=LEARNING_RATE) for i in range(NUM_PLAYERS)]
    criterion = nn.MSELoss()
    replay_buffers = [ReplayBuffer(REPLAY_BUFFER_CAPACITY) for _ in range(NUM_PLAYERS)]
    epsilons = [EPSILON_START for _ in range(NUM_PLAYERS)]
    csv_filename = "training_log.csv"
    csv_file = open(csv_filename, "w", newline="")
    csv_writer = csv.writer(csv_file)
    # CSV header includes move frequencies.
    csv_writer.writerow(["Episode", "TotalReward", "AvgLoss", "pass", "single", "pair", "triplet", "four", "sequence"])
    
    # Setup matplotlib for realtime plotting (4 subplots):
    plt.ion()
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 16))
    episodes_list = []
    rewards_list = []
    losses_list = []
    # Record move frequency per move type over episodes.
    move_counts_over_episodes = {
        "pass": [],
        "single": [],
        "pair": [],
        "triplet": [],
        "four": [],
        "sequence": []
    }
    writer_tb = SummaryWriter('runs/daifugo')
    steps_done = 0

    last_episode = 0  # To record last episode number completed.
    try:
        for episode in range(1, NUM_EPISODES + 1):
            last_episode = episode
            state = env.reset(first_hand=(episode == 1))
            done = False
            step = 0
            total_rewards = [0 for _ in range(NUM_PLAYERS)]
            loss_records = [[] for _ in range(NUM_PLAYERS)]
            # Record move counts for current episode.
            episode_move_count = {"pass": 0, "single": 0, "pair": 0, "triplet": 0, "four": 0, "sequence": 0}
            while not done and step < MAX_STEPS_PER_EPISODE:
                step += 1
                current_player = env.current_player
                valid_actions = env.get_valid_action_indices(current_player)
                if HUMAN_PLAYER and current_player == 0:
                    action = human_select_action(env, current_player)
                else:
                    action = select_action(agents[current_player], state, epsilons[current_player], valid_actions, DEVICE)
                # Decode move for logging.
                current_move = env.decode_action(action, current_player)
                next_state, reward, done, info = env.step(action)
                total_rewards[current_player] += reward
                if reward != -1 and current_move is not None:
                    episode_move_count[current_move["type"]] += 1
                replay_buffers[current_player].push(state, action, reward, next_state, done)
                if TRAINING_MODE and (not (HUMAN_PLAYER and current_player == 0)):
                    loss = train_dqn(agents[current_player], target_agents[current_player],
                                     optimizers[current_player], criterion,
                                     replay_buffers[current_player], BATCH_SIZE, GAMMA, DEVICE)
                    if loss:
                        loss_records[current_player].append(loss)
                state = next_state
                steps_done += 1
                if steps_done % TARGET_UPDATE_FREQ == 0:
                    for i in range(NUM_PLAYERS):
                        target_agents[i].load_state_dict(agents[i].state_dict())
                if TRAINING_MODE and epsilons[current_player] > EPSILON_END and (not (HUMAN_PLAYER and current_player == 0)):
                    epsilons[current_player] = max(EPSILON_END, epsilons[current_player] * EPSILON_DECAY)
            avg_loss = np.mean([np.mean(loss_rec) if loss_rec else 0 for loss_rec in loss_records])
            episode_total_reward = sum(total_rewards)
            print(f"Episode {episode} finished. Total Reward: {episode_total_reward}, Avg Loss: {avg_loss:.4f}")
            writer_tb.add_scalar("Reward/episode", episode_total_reward, episode)
            writer_tb.add_scalar("Loss/episode", avg_loss, episode)
            # Write CSV row including move frequencies.
            csv_writer.writerow([episode, episode_total_reward, avg_loss,
                                 episode_move_count["pass"], episode_move_count["single"],
                                 episode_move_count["pair"], episode_move_count["triplet"],
                                 episode_move_count["four"], episode_move_count["sequence"]])
            csv_file.flush()
            episodes_list.append(episode)
            rewards_list.append(episode_total_reward)
            losses_list.append(avg_loss)
            # Append move counts for each move type.
            for move_type in move_counts_over_episodes.keys():
                move_counts_over_episodes[move_type].append(episode_move_count[move_type])
            # Update line plots:
            ax1.clear()
            ax1.plot(episodes_list, rewards_list, label="Total Reward")
            ax1.set_xlabel("Episode")
            ax1.set_ylabel("Reward")
            ax1.legend()
            ax2.clear()
            ax2.plot(episodes_list, losses_list, label="Avg Loss", color="red")
            ax2.set_xlabel("Episode")
            ax2.set_ylabel("Loss")
            ax2.legend()
            ax3.clear()
            for move_type, counts in move_counts_over_episodes.items():
                ax3.plot(episodes_list, counts, label=move_type)
            ax3.set_xlabel("Episode")
            ax3.set_ylabel("Cumulative Move Count")
            ax3.legend()
            # Update bar plot for current episode move frequencies.
            ax4.clear()
            move_types = list(episode_move_count.keys())
            counts = [episode_move_count[m] for m in move_types]
            ax4.bar(move_types, counts, color="green")
            ax4.set_xlabel("Move Type")
            ax4.set_ylabel("Frequency in Current Episode")
            ax4.set_title(f"Episode {episode} Move Frequency")
            fig.tight_layout()
            plt.pause(0.01)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving progress...")
    finally:
        # Save agent models with checkpoint dictionary.
        for i in range(NUM_PLAYERS):
            model_filename = f"dqn_player_{i}.pth"
            checkpoint = {"episode": last_episode, "state_dict": agents[i].state_dict()}
            torch.save(checkpoint, model_filename)
            print(f"Saved model for player {i} to {model_filename} (last episode: {last_episode}).")
        csv_file.close()
        writer_tb.close()
        # Move usage analysis.
        print("\nMove Usage Analysis:")
        for move_type, counts in move_counts_over_episodes.items():
            if counts:
                max_count = max(counts)
                max_ep = episodes_list[counts.index(max_count)]
                print(f"Move type '{move_type}' was used most in episode {max_ep} with count {max_count}.")
        plt.ioff()
        plt.show()
        if VISUALIZE:
            pygame.quit()

if __name__ == "__main__":
    main()
