import sys

try:
    import torch
    from game import ACTION_SPACE_SIZE, DQN, DaifugoEnv, ReplayBuffer, train_dqn
except ModuleNotFoundError as exc:
    torch = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


def _require_deps():
    if IMPORT_ERROR is not None:
        raise RuntimeError(f"Missing test dependency: {IMPORT_ERROR.name}") from IMPORT_ERROR


def test_pass_is_invalid_on_empty_table():
    _require_deps()
    env = DaifugoEnv(visualize=False)
    env.last_move = None
    env.first_move = True
    env.current_player = 0

    assert 0 not in env.get_valid_action_indices(0)

    _, reward, done, _ = env.step(0)
    assert reward == -1
    assert not done


def test_winner_order_stores_final_player():
    _require_deps()
    env = DaifugoEnv(visualize=False)
    env.hands = [[], [], [], [], [0]]
    env.winner_order = [0, 1, 2]
    env.current_player = 3
    env.first_move = False
    env.last_move = None

    _, _, done, info = env.step(0)
    assert not done

    env.hands[3] = [1]
    env.current_player = 3
    action = 2
    _, _, done, info = env.step(action)

    assert done
    assert env.winner_order == [0, 1, 2, 3, 4]
    assert info["winner_order"] == env.winner_order


def test_exchange_is_reciprocal_and_keeps_hand_sizes():
    _require_deps()
    env = DaifugoEnv(visualize=False)
    env.winner_order = [0, 1, 2, 3, 4]
    env.hands = [
        [0, 4, 8],
        [1, 5, 9],
        [2, 6, 10],
        [44, 48, 3],
        [52, 51, 7],
    ]
    before_sizes = [len(hand) for hand in env.hands]

    env.exchange_cards()

    assert [len(hand) for hand in env.hands] == before_sizes
    assert 52 in env.hands[0]
    assert 51 in env.hands[0]
    assert 0 in env.hands[4]
    assert 48 in env.hands[1]
    assert 1 in env.hands[3]


def test_observation_includes_last_move_state():
    _require_deps()
    env = DaifugoEnv(visualize=False)
    env.current_player = 0
    empty_obs = env.get_obs()
    env.last_move = {"type": "single", "cards": [12]}
    env.pass_count = 2
    env.revolution = True
    obs = env.get_obs()

    assert len(obs) == 53 + 53 + 5 + 4 + env.num_players * 2
    assert obs[53 + 12] == 1.0
    assert obs[53 + 53] == 1.0
    assert obs.sum() > empty_obs.sum()


def test_train_dqn_masks_invalid_next_actions():
    _require_deps()
    env = DaifugoEnv(visualize=False)
    input_dim = len(env.get_obs())
    model = DQN(input_dim, ACTION_SPACE_SIZE)
    target_model = DQN(input_dim, ACTION_SPACE_SIZE)
    target_model.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    buffer = ReplayBuffer(4)
    state = env.get_obs()

    buffer.push(state, 1, 0.0, state, False, [1])
    loss = train_dqn(
        model,
        target_model,
        optimizer,
        torch.nn.MSELoss(),
        buffer,
        batch_size=1,
        gamma=0.99,
        device=torch.device("cpu"),
    )

    assert loss >= 0.0


if __name__ == "__main__":
    if IMPORT_ERROR is not None:
        print(f"Missing test dependency: {IMPORT_ERROR.name}")
        sys.exit(1)
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            value()
    print("regression checks passed")
